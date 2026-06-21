# reserved — Resource Reservation Daemon

## TLDR

`reserved` is a single-machine, best-effort resource reservation system for physical hardware (GPUs, CPU, RAM). It succeeds or fails immediately — there is no queue.

1. **Reservation system** — processes claim a slice of physical hardware capacity (device type, VRAM, compute budget). The daemon grants or rejects immediately.
2. **Handle-based lifetime** — each reservation is backed by an open Unix Domain Socket connection. Multiple reservations can be acquired atomically in one request but released independently, each via its own socket. When a process dies for any reason, its sockets close and all held reservations are freed automatically. No cleanup code required.
3. **Optional SDK** — a Python SDK provides a `Reservation` context manager for acquiring reservations and a `check()` method developers should call at natural checkpoints (between epochs, between inferences). If the daemon has restarted and the reservation is gone, `check()` raises `ReservationExpiredError` so the program can exit gracefully.
4. **Daemon** — a lightweight process that holds the resource ledger in memory, grants/rejects reservation requests, and releases them when connections close. State is lost on restart.
5. **Optional status API** — an HTTP endpoint and `reserve-status` CLI for operators to inspect current device capacity and active reservations.
6. **Convenience wrapper** — a `reserve` shell script that wraps any non-reservation-aware program, acquiring a reservation and exec'ing the target command under it.

---

## What is reserved?

As GPU based AI applications become popular, our operating systems don't offer resource management tooling to help users run applications without interfering with each other. Reserved offers an opt-in mechanism for process resource coordination by "reserving" resources for a task, and then releasing them when finished.

A reservation is a claim on a slice of hardware — a specific device, an amount of VRAM, a compute budget. The daemon grants the claim if the hardware is available, or rejects it immediately if not. Each reservation is backed by an open Unix Domain Socket; if the process dies for any reason, the socket closes and the resources are freed automatically — no cleanup required.

Multiple reservations can be requested atomically in a single request (all granted or all rejected), but each carries its own socket handle and can be released independently. This allows callers to model independent resource lifetimes — a model checkpoint held for a session, LoRA weights freed after each inference — while still acquiring everything upfront in one atomic step.

Normal POSIX socket semantics (fork/exec, fd inheritance) apply, making it straightforward to wrap non-reservation-aware programs in the reservation system.

---

## Components

### Daemon (`reserved`)

The daemon process that runs on the machine. It loads a JSON config describing the available hardware, then listens on a Unix domain socket for reservation requests. It holds the resource ledger in memory, grants or rejects requests, and releases reservations automatically when a client connection closes.

Optionally serves an HTTP endpoint for monitoring (`--http PORT`).

### Scripts

**`reserve`** — requests a reservation and execs a command under it. Sets `CUDA_VISIBLE_DEVICES` and other env vars before handing off to the command. The reservation is released when the command exits.

```bash
reserve train type=cuda,model=a6000 -- python train.py
```

**`reserve-status`** — prints the current state of all devices and active reservations. Accepts `--json` for scripting.

```bash
reserve-status

Devices:
  cuda:0  cuda  rtx3090  available: 24576/24576 MB  100/100 compute
  cuda:1  cuda  rtx3090  available: 16384/24576 MB  50/100 compute
  cuda:2  cuda  a6000    available: 49152/49152 MB  100/100 compute

Reservations:
  a3f9c2  label=train  age=2m14s
    cuda:1  8192 MB  50 compute
```

### SDKs

**Python (`reserve`)** — a `Reservation` context manager that acquires a reservation on enter and releases it on exit. Suitable for notebooks, training scripts, or any Python process that needs to claim hardware for a block of work.

```python
from reserve import Reservation, DeviceConstraint

with Reservation(DeviceConstraint("cuda", ram_mb=16384), label="train") as r:
    model.to(r.devices[0].torch_device())
    run_training()
```

---

## Resource Config (`reserved.json`)

Defines the set of physical devices on this machine. The daemon loads this at startup and never re-reads it while running.

```json
{
  "devices": [
    { "name": "cuda:0", "type": "cuda",  "model": "rtx3090",      "ram_mb": 24576, "compute_units": 100 },
    { "name": "cuda:1", "type": "cuda",  "model": "rtx3090",      "ram_mb": 24576, "compute_units": 100 },
    { "name": "cuda:2", "type": "cuda",  "model": "a6000",        "ram_mb": 49152, "compute_units": 100 },
    { "name": "metal:0","type": "metal", "model": "m3-max",       "ram_mb": 36864, "compute_units": 100 },
    { "name": "cpu",    "type": "cpu",                             "ram_mb": 65536 }
  ]
}
```

Each device has:

- `name` — unique identifier, returned in responses and set as `RESERVED_DEVICE`
- `type` — `cuda`, `metal`, or `cpu`
- `model` — optional human label for the hardware class (e.g. `rtx3090`, `a6000`, `m3-max`).

Callers can filter on this to target specific hardware.

- `ram_mb` — total reservable RAM. For CUDA devices this is VRAM; for CPU this is

system RAM. Multiple reservations can share a device as long as combined usage fits.

- `compute_units` — optional opaque token budget. No physical unit is implied; the

meaning is a convention between the operator and callers.

---

## Reservation Spec

Requests follow a four-level hierarchy:

```
Request
└── Reservation  (one per independent lifetime)
    └── Slot     (one per physical device)
        └── Resource  (capacity claimed on that device)
```

- **Request** — the atomic unit. All reservations in a request are granted together or the entire request is rejected. One wire message.
- **Reservation** — a named grouping with its own handle (socket fd). Reservations within a request are granted atomically but released independently. The caller declares reservation boundaries explicitly based on intended lifetime — the daemon does not infer them.
- **Slot** — one physical device within a reservation. A reservation can hold multiple slots; all slots are released when the reservation handle closes.
- **Resource** — the capacity claimed on a slot's device (`ram_mb`, `compute_units`). Multiple slots can target the same physical device as long as combined capacity fits.

**Constraint fields** (per slot):

| Field | Effect |
| --- | --- |
| name | Pin to a specific device by name (e.g. `cuda:2`). Overrides type/model matching. |
| type | Device must be this type (`cuda`, `metal`, or `cpu`) |
| model | Device must have this model label |
| ram_mb | Device must have at least this much available RAM |
| compute_units | Device must have at least this many available compute units |

Unspecified fields are unconstrained. The daemon picks any satisfying device. Use `name` to pin when your orchestrator has already decided placement.

**Naive use — daemon picks placement:**

```
# Single reservation, one slot: any CUDA device
reservation(slot(type=cuda))

# Single reservation, one slot: needs 30GB VRAM
reservation(slot(type=cuda, ram_mb=30720))

# Single reservation, two slots: distributed job, two GPUs released together
reservation(slot(type=cuda), slot(type=cuda))

# Single reservation, two slots: GPU + CPU released together
reservation(slot(type=cuda, model=a6000), slot(type=cpu, ram_mb=32768))
```

**Multiple reservations — independent lifetimes:**

```
# Stable diffusion: checkpoint and LoRA acquired atomically, freed independently
request(
  reservation("checkpoint", slot(type=cuda, ram_mb=8192)),
  reservation("lora",       slot(type=cuda, ram_mb=2048)),
)

# TTS pipeline: 5 actors acquired atomically, freed one by one as each finishes
request(
  reservation("actor-1", slot(type=cuda, ram_mb=4096)),
  reservation("actor-2", slot(type=cuda, ram_mb=4096)),
  reservation("actor-3", slot(type=cuda, ram_mb=4096)),
  reservation("actor-4", slot(type=cuda, ram_mb=4096)),
  reservation("actor-5", slot(type=cuda, ram_mb=4096)),
)
```

**Orchestrator-pinned placement:**

When a sophisticated caller (e.g. a pipeline orchestrator) has already decided how to pack devices, it reads `status`, runs its own placement algorithm, and submits pinned slots using `name`. The daemon's solver is bypassed — it just does atomic check-and-subtract.

```
# Orchestrator has decided to consolidate onto cuda:0 for co-location with LLM stage
request(
  reservation("checkpoint", slot(name=cuda:0, ram_mb=8192)),
  reservation("lora",       slot(name=cuda:0, ram_mb=2048)),
)

# Orchestrator has decided to spread actors across two cards for throughput
request(
  reservation("actor-1", slot(name=cuda:0, ram_mb=4096)),
  reservation("actor-2", slot(name=cuda:0, ram_mb=4096)),
  reservation("actor-3", slot(name=cuda:1, ram_mb=4096)),
  reservation("actor-4", slot(name=cuda:1, ram_mb=4096)),
  reservation("actor-5", slot(name=cuda:1, ram_mb=4096)),
)
```

If a pinned request fails due to a race, the orchestrator re-reads status and re-plans. The daemon owns atomicity; the orchestrator owns strategy.

---

## Daemon (`reserved`)

### Startup

```
reserved --config reserved.json [--socket PATH] [--socket-group GROUP] [--http PORT]
```

- Loads config, builds in-memory ledger.
- Acquires the listening socket (see socket acquisition below).
- Writes a PID file alongside the socket.
- Accepts connections and processes requests one at a time per connection.
- If `--http PORT` is given, also starts an HTTP server on `0.0.0.0:PORT` (stdlib

`http.server`, no dependencies). Serves two routes:

- `GET /status` — returns the same payload as the UDS `status` op as JSON

(`Content-Type: application/json`).

- `GET /` — returns a self-contained HTML page with a human-readable table of

devices and active reservations. No JavaScript, no external assets; the daemon templates it directly. Suitable for a browser tab or a simple dashboard embed.

### Socket Acquisition

The daemon supports two socket startup paths and checks them in order:

**1. Systemd socket activation (preferred for system deployments)**

If the `LISTEN_FDS` environment variable is set, systemd has already created and bound the socket and passed it as an inherited file descriptor. The daemon calls `socket.fromfd(SD_LISTEN_FDS_START)` and skips `bind()` entirely. Socket path, permissions, and group are all declared in the systemd socket unit — the daemon does not need to know them.

**2. Self-bind (standalone / user deployments)**

If `LISTEN_FDS` is not set, the daemon creates and binds the socket itself:

- Path: `--socket PATH` flag, default `/var/run/reserved.sock`
- Permissions: `0660` by default
- Group: `--socket-group GROUP` sets the socket file's group, allowing non-root users

in that group to connect without making the socket world-writable

This two-path approach requires no systemd library — just a check on `LISTEN_FDS`.

### Ledger

The ledger is an in-memory mapping of each device to its total and currently available `(ram_mb, compute_units)`. Active reservations are kept in memory with their granted device assignments. No database. State is lost on daemon restart; all reservations are implicitly released.

### Request / Response Protocol

Line-delimited JSON over the UDS. Each message is one JSON object followed by `\n`.

**Reserve request:**

A single request contains one or more reservations. Each reservation has a label and one or more slots. Each slot is a device constraint.

```json
{
  "op": "reserve",
  "reservations": [
    {
      "label": "checkpoint",
      "slots": [
        { "type": "cuda", "ram_mb": 8192 }
      ]
    },
    {
      "label": "lora",
      "slots": [
        { "type": "cuda", "ram_mb": 2048 }
      ]
    }
  ]
}
```

All slot constraints across all reservations are resolved atomically. Slot constraint fields (`name`, `type`, `model`, `ram_mb`, `compute_units`) are all optional — omitted fields are unconstrained.

**Reserve response (success):**

The response contains one entry per reservation, each with its own `reservation_id` and assigned slots. `reservation_id` identifies the handle — closing the corresponding socket releases that reservation.

```json
{
  "ok": true,
  "reservations": [
    {
      "reservation_id": "a3f9c2",
      "label": "checkpoint",
      "slots": [
        { "device_name": "cuda:2", "device_index": 0 }
      ]
    },
    {
      "reservation_id": "b7e1d4",
      "label": "lora",
      "slots": [
        { "device_name": "cuda:2", "device_index": 0 }
      ]
    }
  ]
}
```

`slots` is parallel to the request `slots` array for that reservation. `device_index` is a remapped 0-based CUDA index scoped to this reservation (for use in `CUDA_VISIBLE_DEVICES`). Metal and CPU devices have `device_index: -1`.

Each reservation requires its own lease fd. The daemon creates these fds and delivers them to the client via `SCM_RIGHTS` (Unix domain socket ancillary data) alongside the JSON response. The client receives N open file descriptors — one per reservation — in the same message as the grant. Each fd is a lease: the daemon watches it for closure and releases that reservation's resources when it closes. No follow-up messages or additional connections are needed.

**Reserve response (failure):**

```json
{
  "ok": false,
  "error": "insufficient resources"
}
```

No resources are changed. No fds are sent.

**Status request / response** (for introspection):

```json
{ "op": "status" }

{
  "devices": [
    {
      "name": "cuda:0",
      "type": "cuda",
      "model": "rtx3090",
      "ram_mb": 24576,
      "available_ram_mb": 16384,
      "compute_units": 100,
      "available_compute_units": 50
    }
  ],
  "reservations": [
    {
      "id": "a3f9c2",
      "label": "checkpoint",
      "created_at": 1748000000.0,
      "slots": [
        { "device_name": "cuda:2", "ram_mb": 8192, "compute_units": 0 }
      ]
    },
    {
      "id": "b7e1d4",
      "label": "lora",
      "created_at": 1748000000.0,
      "slots": [
        { "device_name": "cuda:2", "ram_mb": 2048, "compute_units": 0 }
      ]
    }
  ]
}
```

### Behavior on Reserve

1. Collect all slot constraints across all reservations in the request. Resolve them atomically — earlier assignments reduce available capacity for later ones.
2. If any slot constraint cannot be satisfied: respond `ok: false`, no resources are changed.
3. If all slots are satisfied: record all reservations, subtract capacity from each assigned device, respond with the full assignment.
4. Hold each reservation's resources until its lease socket is closed.

### Lease Lifetime and Connection Model

Each reservation is represented by a separate open socket connection. The daemon releases that reservation's resources when its socket closes — whether by explicit release, process exit, or crash. Multiple reservations from the same request have independent sockets and independent lifetimes.

**Connection lifecycle:**

```
client opens a single transport socket, sends reserve request
  → daemon resolves all slots atomically
  → [if ok] daemon creates N lease fds (one per reservation)
            sends JSON response + N fds via SCM_RIGHTS in one message
  → client receives response and N fds
  → client closes the transport socket (it was only needed for the request)
  → each lease fd is held open independently — closing it releases that reservation
  → reservation released when all copies of its fd are closed
     (explicit release, process exit, crash, or last forked copy exits)
```

The transport socket is short-lived — connect, exchange one message, close. The lease fds are what the daemon watches. For a single-reservation request, one fd is returned. For N reservations, N fds are returned, each independently closeable.

The daemon multiplexes all open lease fds in a single thread using `select`. Status requests arrive on their own short-lived connections and are handled inline.

### Daemon does NOT:

- Queue or retry requests (callers decide when to retry).
- Persist state across restarts.
- Know anything about tasks, pipelines, or workers.

---

## Socket Semantics

The open socket connection *is* the reservation. Application developers and SDK users should manage their reservations by reasoning about standard POSIX file descriptor semantics — no reserved-specific lifecycle API is needed beyond acquiring and releasing the initial connection.

### fork()

`fork()` duplicates all open file descriptors into the child. Both parent and child hold the reservation. The reservation lives until the last copy of the fd is closed — so a process that forks worker children can exit itself and the reservation stays alive until all workers finish. This is the correct model for a process pool or multiprocessing job: acquire once in the parent, fork workers, let the parent exit; the reservation outlives it.

### exec()

`exec()` behavior depends on the `O_CLOEXEC` flag on the socket fd. If the flag is set, the fd is closed at exec and the reservation is released at that point — not when the exec'd program exits. To transfer the lease across an exec boundary, `O_CLOEXEC` must be cleared before calling exec.

**Python** sets `O_CLOEXEC` by default on all sockets. Clear it explicitly before exec:

```python
import os
os.set_inheritable(sock.fileno(), True)
os.execvp("child", ["child", ...])
```

**C/C++** does not set `O_CLOEXEC` by default. A socket created with `socket()` survives exec unless you opt out:

```c
// Default: fd survives exec — child inherits the reservation automatically
int fd = socket(AF_UNIX, SOCK_STREAM, 0);

// Opt out: close on exec
int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
// or after the fact:
fcntl(fd, F_SETFD, FD_CLOEXEC);
```

When a reservation fd is inherited across exec, the exec'd program holds the lease for its entire lifetime — including any children it forks — with no knowledge of `reserved`. This is how the `reserve` CLI works: it acquires the reservation, clears `FD_CLOEXEC` on the lease fd, sets environment variables, and execs the target command. The reservation ends when CMD and all its descendants have closed the fd.

### Passing a reservation through an exec boundary

Any program can pass its reservation to a child process it execs by ensuring the fd is inheritable (clear `O_CLOEXEC` in Python; leave it unset in C). The child need not speak the `reserved` protocol — it simply holds an open fd. The reservation lives until the last copy is closed across the entire process tree.

### Python SDK and reservation lifetime

The Python SDK's `Reservation` context manager keeps the socket open for the lifetime of the `with` block. If the process forks inside the block, both parent and child hold the reservation. If the process calls `release()` or exits the `with` block in the parent, the child still holds a copy of the fd and the reservation remains live until the child closes it.

```python
with Reservation(DeviceConstraint("cuda", ram_mb=16384)) as r:
    pid = os.fork()
    if pid == 0:
        # child: still holds the reservation via inherited fd
        run_worker(r.devices[0])
        os._exit(0)
    # parent: reservation still live while child runs
    os.waitpid(pid, 0)
# both parent and child have exited — reservation released
```

To pass a reservation to an exec'd child from Python:

```python
with Reservation(DeviceConstraint("cuda")) as r:
    os.set_inheritable(r._sock.fileno(), True)
    os.execvp("worker", ["worker"])
    # exec replaces this process; reservation held by worker until it exits
```

### Finding reservation owners

The daemon has no way to verify which process holds a reservation — `pid` fields in requests are advisory only and not tracked. To find what process owns a given socket connection, use `lsof`:

```bash
# List all processes with an open connection to the daemon socket
lsof /var/run/reserved.sock
```

Cross-reference the reservation IDs from `reserve-status` with socket file descriptors shown by `lsof` to identify the owning process.

---

## `reserve` CLI

Acquires a single reservation and runs a command under it.

```
reserve LABEL [SLOT ...] [--socket PATH] -- CMD [ARGS...]

Arguments:
  LABEL             Required. Human-readable name for this reservation (e.g. "train", "infer").
                    Shown in reserve-status output and daemon logs.
  SLOT              Device constraint for one physical slot. Comma-separated fields:
                    type=TYPE, name=NAME, model=MODEL, ram=MB, compute=N.
                    All fields optional. Repeat for multiple slots. Omit entirely for
                    no device constraint.

Options:
  --socket PATH     Daemon socket path [default: /var/run/reserved.sock]
```

All slots belong to the single reservation and are granted atomically — all or none.

```bash
# No device constraint
reserve train -- python infer.py

# Any CUDA device
reserve infer type=cuda -- python infer.py

# Metal device (Apple Silicon)
reserve infer type=metal -- python infer.py

# Needs 30GB VRAM (rules out 3090s)
reserve train type=cuda,ram=30720 -- python train.py

# Specifically a 3090
reserve infer type=cuda,model=rtx3090 -- python infer.py

# Pin to a specific device by name
reserve infer name=cuda:2 -- python infer.py

# Two CUDA slots (distributed job)
reserve distributed type=cuda type=cuda -- python distributed.py

# One A6000 slot + one CPU slot
reserve pipeline type=cuda,model=a6000 type=cpu,ram=32768 -- python pipeline.py
```

**What it does:**

1. Asks the daemon for the requested devices.
2. If the daemon is not running: prints an error to stderr and exits 2.
3. If the daemon rejects (insufficient resources): prints an error to stderr and exits 1.
4. If granted: sets environment variables, then replaces itself with CMD via `exec`.

CMD runs directly — no wrapper process sitting around it.

**Environment variables set before exec:**

| Variable | Value |
| --- | --- |
| CUDA_VISIBLE_DEVICES | Comma-separated remapped CUDA indices (CUDA devices only), e.g. `0` or `0,1` |
| RESERVED_DEVICE | Physical name of the first granted accelerator device, e.g. `cuda:2` or `metal:0` |
| RESERVED_DEVICES | Comma-separated physical names of all granted devices |
| RESERVED_ID | Reservation ID (opaque string) |
| RESERVED_FD | File descriptor number of the open lease socket, e.g. `3` |

`RESERVED_FD` is not needed by typical programs (they simply inherit the fd and hold the lease by staying alive). It is useful for wrapper scripts that need to explicitly close the fd early, debugging tools that want to identify the lease fd, or fd-preserving child launchers that need to pass it along.

CUDA device indices are always remapped to start from 0 within this reservation regardless of which physical devices were assigned. A process granted `cuda:2` and `cuda:3` sees them as `CUDA_VISIBLE_DEVICES=0,1`.

Metal does not have an equivalent device-selection environment variable — there is only one logical MPS device per system. `reserved` coordinates access to the Metal device via the reservation mechanism; the exec'd process selects it in code (e.g. `torch.device("mps")`).

**Why `exec` instead of running CMD as a subprocess:** `exec` replaces `reserve` with CMD entirely — no wrapper process lingers waiting for CMD to finish. `reserve` clears `FD_CLOEXEC` on the lease fd before exec, so CMD inherits it and holds the lease for its own lifetime. The reservation ends when CMD (and any descendants) exit, not when `reserve` would have exited.

---

## Python SDK

```python
from reserve import Request, Reservation, DeviceConstraint, ReservationError
```

**Single reservation — use `Reservation` directly as a context manager:**

```python
# Any CUDA device
with Reservation("infer", DeviceConstraint("cuda")) as r:
    model.to(r.devices[0].torch_device())
    run_model()

# Metal device (Apple Silicon)
with Reservation("infer", DeviceConstraint("metal")) as r:
    model.to(r.devices[0].torch_device())   # torch.device("mps")
    run_model()

# Needs 30GB VRAM — rules out 3090s
with Reservation("infer", DeviceConstraint("cuda", ram_mb=30720)) as r:
    model.to(r.devices[0].torch_device())
    ...

# Two CUDA devices in one reservation — released together
with Reservation("distributed", DeviceConstraint("cuda"), DeviceConstraint("cuda")) as r:
    devices = r.torch_devices()   # [torch.device("cuda:0"), torch.device("cuda:1")]
    run_distributed(devices)
```

**Multiple reservations — use `Request` for atomic acquire, independent release:**

`Request.acquire()` returns a tuple of `Reservation` objects in the same order as declared. All are granted atomically or the call raises. Each is a context manager and can be released independently via `with` or `.release()`.

```python
# Stable diffusion: checkpoint held for the session, LoRA freed after each inference
req = Request(
    Reservation("checkpoint", DeviceConstraint("cuda", ram_mb=8192)),
    Reservation("lora",       DeviceConstraint("cuda", ram_mb=2048)),
)
checkpoint, lora = req.acquire()

with checkpoint:
    load_checkpoint(checkpoint.devices[0].torch_device())
    with lora:
        load_lora(lora.devices[0].torch_device())
        run_inference()
    # lora released here, VRAM returned to pool
# checkpoint released here

# TTS pipeline: 5 actors acquired atomically, freed one by one as each finishes
actors = Request(*[
    Reservation(f"actor-{i}", DeviceConstraint("cuda", ram_mb=4096))
    for i in range(5)
]).acquire()

for actor, line in zip(actors, script):
    render(actor, line)
    actor.release()   # frees this actor's VRAM immediately
```

**`Reservation(label, *constraints)`**

Label is required and positional. One or more `DeviceConstraint` objects follow. All constraints in a single `Reservation` share the same lease fd and are released together.

**`DeviceConstraint(type, *, name=None, model=None, ram_mb=0, compute_units=0)`**

All fields except `type` are optional. Omitting a field means unconstrained on that dimension. `type` may also be omitted entirely (pass `None`) to match any device type. Pass `name` to pin to a specific device by name — useful when an orchestrator has already decided placement.

```python
# Orchestrator-pinned: consolidate onto cuda:0 for co-location
req = Request(
    Reservation("checkpoint", DeviceConstraint(name="cuda:0", ram_mb=8192)),
    Reservation("lora",       DeviceConstraint(name="cuda:0", ram_mb=2048)),
)
```

**`Reservation` fields and methods after `acquire()`:**

- `reservation_id: str`
- `label: str`
- `devices: list[GrantedDevice]` — parallel to the constraints passed at construction
- `torch_devices() -> list[torch.device]` — convenience wrapper over `[d.torch_device() for d in self.devices]`
- `release()` — closes the lease fd, releasing all slots in this reservation
- `check()` — see below

**`GrantedDevice` fields and methods:**

- `device_name: str` — physical device name, e.g. `"cuda:2"` or `"metal:0"`
- `device_index: int` — remapped 0-based CUDA index scoped to this reservation. Metal and CPU devices have `device_index = -1`.
- `torch_device() -> torch.device` — `torch.device("cuda:N")` for CUDA, `torch.device("mps")` for Metal, `torch.device("cpu")` for CPU.

**`check()`**

Call at natural checkpoints between units of work — between training epochs, between inference calls, between actor lines. Performs a non-blocking read on the lease fd. If the daemon has restarted (EOF), raises `ReservationExpiredError`. Returns immediately if healthy.

```python
for epoch in range(num_epochs):
    r.check()   # raises ReservationExpiredError if daemon restarted
    train_epoch(model, data)
```

**Errors:**

- `ReservationError` — raised by `acquire()` when the daemon rejects the request.
- `DaemonNotRunningError` — raised when the daemon socket does not exist or connection is refused.
- `ReservationExpiredError` — raised by `check()` when the lease fd has been closed by the daemon (daemon restarted). Treat as a signal to abort and exit.

**Connection:** `acquire()` connects to the daemon, sends the reserve request, and receives back one lease fd per reservation via `SCM_RIGHTS`. The transport socket is then closed — the lease fds are what the daemon watches. `release()` / `__exit__` closes the lease fd; no message is sent. If the fd has been inherited by a forked child or passed across exec, closing it in the parent does not release the reservation — the child's copy keeps it alive.

**SDK configuration** (in priority order):

1. Constructor kwargs (`socket_path=`, `label=`)
2. Environment variable `RESERVED_SOCKET` for socket path
3. Default: `/var/run/reserved.sock`

---

## Status API

The status API is optional operator tooling for inspecting device capacity and active reservations. It has two interfaces: a CLI for interactive use and scripting, and an HTTP endpoint for dashboards and monitoring systems.

## `reserve-status` CLI

Connects to the UDS, sends `{"op": "status"}`, and prints to stdout. No HTTP server required.

```
reserve-status [--socket PATH] [--json]
```

Default output is human-readable text:

```
Devices:
  cuda:0  cuda  rtx3090  available:  8192/24576 MB  100/100 compute
  cuda:1  cuda  rtx3090  available:  4096/24576 MB  100/100 compute
  cuda:2  cuda  a6000    available: 49152/49152 MB  100/100 compute
  cpu     cpu            available: 65536/65536 MB

Reservations:
  a3f9c2  label=actor-1  age=3m12s
    cuda:0  4096 MB
  b7e1d4  label=actor-2  age=3m12s
    cuda:0  4096 MB
  c2a8f1  label=actor-3  age=3m12s
    cuda:0  4096 MB
  d9b3e7  label=actor-4  age=3m12s
    cuda:1  4096 MB
  e4f6c9  label=actor-5  age=3m12s
    cuda:1  4096 MB
```

Three actors are packed onto `cuda:0`, two onto `cuda:1`. As each actor finishes its lines, its reservation is freed and that VRAM returns to the pool — the other actors are unaffected.

With `--json`, prints the raw status JSON for use in scripts:

```bash
reserve-status --json | jq '.reservations | length'
```

---

## Deployment

### System daemon (systemd, root)

Suitable for shared machines where multiple users need access to the same GPUs. The daemon runs as a dedicated service account; a unix group controls who can reserve.

```ini
# /etc/systemd/system/reserved.socket
[Unit]
Description=reserved resource reservation daemon socket

[Socket]
ListenStream=/var/run/reserved.sock
SocketGroup=gpu-users
SocketMode=0660

[Install]
WantedBy=sockets.target
```

```ini
# /etc/systemd/system/reserved.service
[Unit]
Description=reserved resource reservation daemon
Requires=reserved.socket
After=reserved.socket

[Service]
ExecStart=/usr/bin/reserved --config /etc/reserved.json
User=reserved
Group=reserved
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now reserved.socket
```

Users in the `gpu-users` group can immediately run `reserve` with no additional configuration. The socket exists as soon as the socket unit is active, even before the daemon has fully started.

### User daemon (systemd user, no root)

Suitable for a single-user workstation or a developer machine where root access is not available. The daemon runs as the logged-in user; the socket lives in the user runtime directory.

```ini
# ~/.config/systemd/user/reserved.socket
[Unit]
Description=reserved resource reservation daemon socket

[Socket]
ListenStream=%t/reserved.sock

[Install]
WantedBy=sockets.target
```

```ini
# ~/.config/systemd/user/reserved.service
[Unit]
Description=reserved resource reservation daemon
Requires=reserved.socket
After=reserved.socket

[Service]
ExecStart=/usr/bin/reserved --config %h/.config/reserved.json
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now reserved.socket
export RESERVED_SOCKET="${XDG_RUNTIME_DIR}/reserved.sock"
```

`%t` expands to `$XDG_RUNTIME_DIR` (e.g. `/var/run/user/1000`). Set `RESERVED_SOCKET` in the shell profile so `reserve` and the SDK find the socket automatically.

### Standalone (no systemd)

Run the daemon directly. Suitable for macOS, containers, or environments without systemd.

```bash
reserved --config /etc/reserved.json --socket /var/run/reserved.sock --socket-group gpu-users
```

The daemon binds the socket itself, sets group and permissions, and runs in the foreground. Use a process supervisor (launchd on macOS, runit, s6, etc.) for auto-restart.

---

## What `reserved` does NOT include

- Task queuing or retry logic (callers poll and retry if they want to wait).
- Pipeline / DAG concepts.
- Worker process spawning.
- Persistence (restart = clean slate).
- Multi-box / distributed operation.

---

## Daemon Restart

When the daemon restarts, all open client connections are closed and the in-memory ledger is lost. There is no way to restore reservations — the connection *is* the reservation, and the connection is gone.

Clients holding a reservation at the time of restart are left with a broken socket. They will not know their reservation was lost until they next attempt to communicate with the daemon.

Python SDK users should call `r.check()` at natural checkpoints (between epochs, between inference calls). `check()` performs a non-blocking read; EOF means the daemon has restarted and the reservation is gone, and `ReservationExpiredError` is raised. Raw protocol users can detect this the same way: a non-blocking `recv()` returning 0 bytes signals EOF.

The daemon logs all active reservations at shutdown (`SIGTERM`) so operators can see what work will be disrupted before restarting.

---

## Implementation Architecture

### Module Structure

```
reserved/
  core.py       # pure functions, no IO — the functional core
  daemon.py     # imperative shell — owns the mutable ledger reference, drives the event loop
  protocol.py   # JSON serialization/deserialization for the wire format
  sdk/
    client.py   # Reservation, Request, DeviceConstraint
    errors.py   # ReservationError, DaemonNotRunningError, ReservationExpiredError
```

### Functional Core (`core.py`)

All business logic lives here as pure functions. No IO, no sockets, no global state. Every function takes a `Ledger` and returns a new `Ledger` plus a result. The ledger is an immutable frozen dataclass — functions never mutate it, they produce a replacement.

```python
@dataclass(frozen=True)
class Device:
    name: str
    type: str
    model: str | None
    ram_mb: int
    compute_units: int

@dataclass(frozen=True)
class DeviceAvailability:
    device: Device
    available_ram_mb: int
    available_compute_units: int

@dataclass(frozen=True)
class Ledger:
    availability: tuple[DeviceAvailability, ...]
    reservations: tuple[ActiveReservation, ...]

def load_ledger(config: dict) -> Ledger: ...

def grant(ledger: Ledger, request: ReserveRequest) -> tuple[Ledger, GrantResult]: ...
def release(ledger: Ledger, reservation_id: str) -> Ledger: ...
def status(ledger: Ledger) -> StatusResult: ...
```

`grant()` resolves all slot constraints atomically. If any slot cannot be satisfied, it returns the original ledger unchanged and a failure result. If all slots are satisfied, it returns a new ledger with capacity subtracted and a success result containing the assignments.

This design means the core is fully testable as plain function calls — no daemon, no sockets, no hardware required.

```python
# example test
ledger = load_ledger({"devices": [{"name": "cuda:0", "type": "cuda", "ram_mb": 24576, ...}]})
ledger2, result = grant(ledger, request)
assert result.ok
assert ledger2.availability[0].available_ram_mb == 24576 - 8192
# original ledger is unchanged
assert ledger.availability[0].available_ram_mb == 24576
```

### Imperative Shell (`daemon.py`)

The shell owns the single mutable reference to the current ledger and drives the `select` loop. It calls core functions and swaps the ledger reference with the result.

```python
ledger: Ledger = core.load_ledger(config)

# on reserve request received:
ledger, result = core.grant(ledger, request)
# send result + lease fds to client via SCM_RIGHTS

# on lease fd closed (detected via select):
ledger = core.release(ledger, reservation_id)

# on status request:
result = core.status(ledger)
# send result to client
```

The shell contains no allocation logic — it only handles IO, fd lifecycle, and ledger swapping. Integration tests for the shell run a real daemon subprocess and connect to it over a real socket.
