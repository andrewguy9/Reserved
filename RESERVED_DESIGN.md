# reserved — Resource Reservation Daemon

## TLDR

`reserved` is a single-machine, best-effort resource reservation system for physical hardware (GPUs, CPU, RAM). It succeeds or fails immediately — there is no queue.

1. **Reservation system** — processes claim a slice of physical hardware capacity (device type, VRAM, compute budget). The daemon grants or rejects immediately.
2. **Handle-based lifetime** — each reservation is backed by a lease fd (one end of a socketpair created by the daemon). Multiple reservations can be acquired atomically in one request but released independently, each via its own lease fd. When a process dies for any reason, its fds close and all held reservations are freed automatically. No cleanup code required.
3. **Optional SDK** — a Python SDK provides a `Reservation` context manager for acquiring reservations and a `check()` method developers should call at natural checkpoints (between epochs, between inferences). If the daemon has restarted and the reservation is gone, `check()` raises `ReservationExpiredError` so the program can exit gracefully.
4. **Daemon** — a lightweight process that holds the resource ledger in memory, grants/rejects reservation requests, and releases them when connections close. State is lost on restart.
5. **Optional status API** — an HTTP endpoint and `reserve-status` CLI for operators to inspect current device capacity and active reservations.
6. **Convenience wrapper** — a `reserve` command that wraps any non-reservation-aware program, acquiring a reservation and exec'ing the target command under it.
7. **Config generator** — `reserved-generate-config` writes the hardware config from what torch detects, so CUDA cards are named by GPU UUID without hand-editing.

---

## What is reserved?

As GPU based AI applications become popular, our operating systems don't offer resource management tooling to help users run applications without interfering with each other. Reserved offers an opt-in mechanism for process resource coordination by "reserving" resources for a task, and then releasing them when finished.

A reservation is a claim on a slice of hardware — a specific device, an amount of VRAM, a compute budget. The daemon grants the claim if the hardware is available, or rejects it immediately if not. Each reservation is backed by a lease fd — one end of a socketpair created by the daemon at grant time. If the process dies for any reason, the fd closes and the resources are freed automatically — no cleanup required.

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
reserve train type=cuda,ram=30720 -- python train.py
```

**`reserved-generate-config`** — detects the hardware torch can see and prints a `reserved.json` for it. CUDA cards are named by GPU UUID with their memory and multiprocessor count filled in.

**`reserve-status`** — prints the current state of all devices and active reservations. Accepts `--json` for scripting.

```bash
reserve-status

Devices:
  cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780  cuda  available: 24576/24576 MB  100/100 compute
  cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781  cuda  available: 16384/24576 MB  50/100 compute
  cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782  cuda  available: 49152/49152 MB  100/100 compute

Reservations:
  a3f9c2  label=train  age=2m14s
    cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781  8192 MB  50 compute
```

### SDKs

**Python (`reserved.sdk`)** — a `Reservation` context manager that acquires a reservation on enter and releases it on exit. Suitable for notebooks, training scripts, or any Python process that needs to claim hardware for a block of work.

```python
from reserved.sdk import Reservation, DeviceConstraint

with Reservation("train", DeviceConstraint("cuda", ram_mb=16384)) as r:
    model.to(r.torch_devices()[0])
    run_training()
```

---

## Resource Config (`reserved.json`)

Defines the set of physical devices on this machine. The daemon loads this at startup and never re-reads it while running.

```json
{
  "devices": [
    { "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780", "type": "cuda",  "ram_mb": 24576, "compute_units": 82 },
    { "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781", "type": "cuda",  "ram_mb": 24576, "compute_units": 82 },
    { "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782", "type": "cuda",  "ram_mb": 49152, "compute_units": 84 },
    { "name": "metal:0", "type": "metal", "ram_mb": 36864, "compute_units": 100 },
    { "name": "cpu", "type": "cpu", "ram_mb": 65536 }
  ]
}
```

Each device has:

- `name` — unique physical identifier, returned in grants and set as `RESERVED_DEVICE`
- `type` — `cuda`, `metal`, or `cpu`
- `ram_mb` — total reservable RAM. For CUDA devices this is VRAM; for Metal this is unified memory; for CPU this is system RAM. Multiple reservations can share a device as long as combined usage fits.
- `compute_units` — optional opaque budget. No physical unit is implied; the meaning is a convention between the operator and callers. The generator fills in the GPU's streaming multiprocessor count.

A device is exactly these fields. There is no free-form label: nothing a caller could ask for is left implicit.

**CUDA device naming:** must be `cuda:GPU-<uuid>`, where the UUID is the lowercase one `nvidia-smi -L` prints for the card. The UUID names one specific board regardless of the driver's enumeration order, `CUDA_VISIBLE_DEVICES` accepts it directly, and torch reports it for every visible device. The daemon validates the form at startup. `reserved-generate-config` writes these names for you.

**Metal device naming:** Apple Silicon has a single unified GPU. Use `metal:0` by convention. There is no standard index-based selection for Metal — `reserved` coordinates access via the reservation mechanism; processes select the device in code with `torch.device("mps")`.

A macOS config for an Apple Silicon machine:

```json
{
  "devices": [
    { "name": "metal:0", "type": "metal", "ram_mb": 36864, "compute_units": 100 },
    { "name": "cpu", "type": "cpu", "ram_mb": 65536 }
  ]
}
```

---

## Two Languages: Explicit and Constraint

`reserved` uses two distinct languages for describing reservations, operating at different layers of the stack.

### Explicit Language (daemon protocol)

The daemon speaks only in explicit terms. Every allocation in a reserve request names a specific device and states exactly how much capacity to claim:

```
allocation: name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780  ram_mb=8192  compute_units=50
allocation: name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781  ram_mb=4096  compute_units=25
```

This record, `DeviceAllocation`, is the only device record in the system. The client requests it, the daemon grants it back unchanged, the ledger holds it, and status reports it.

No wildcards. No type matching. The daemon does not search or infer — it checks whether the named device has sufficient available capacity and subtracts if so. This keeps the daemon simple, correct, and easy to test.

### Constraint Language (SDK and CLI)

Users and applications typically don't know or care which physical device they get — they just need a CUDA device with enough VRAM. The constraint language expresses this:

```
constraint: type=cuda  ram_mb=8192
constraint: type=cuda  compute_units=40
constraint: name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780  ram_mb=8192   # a name pins one physical device
```

A constraint has exactly four fields: `type`, `name`, `ram_mb`, and `compute_units`. Every field given narrows the choice. The constraint language is resolved by a **constraint solver** before the request is submitted to the daemon. The solver reads the current status, finds devices satisfying the constraints, and produces an explicit request. The SDK and CLI accept only constraints; pinning a device is a constraint with a `name`.

### Constraint Solvers

A constraint solver is any function that takes a set of device constraints and the current device status, and returns an explicit request. The SDK ships two:

- **`ConsolidateSolver`** — packs allocations onto as few physical devices as possible. Leaves the maximum amount of contiguous free capacity for other workloads. Good for co-location (e.g. checkpoint and LoRA on the same card). **Default for `Reservation`, `Request`, and the `reserve` CLI.**
- **`SpreadSolver`** — distributes allocations across as many physical devices as possible. Good for maximizing parallel throughput (e.g. TTS actors spread across cards).

Both solvers express clear user intent. The choice between them is a statement about how the caller wants to share the machine with other workloads.

An orchestrator that runs its own placement expresses the result as `name` constraints, or speaks the explicit protocol to the daemon directly.

```
Constraint spec  →  Solver (reads status)  →  Explicit request  →  Daemon (check-and-subtract)
                                                        ↑
                                         Orchestrators enter here
```

The daemon is unaware of solvers. New solvers can be added to the SDK or written by callers without any daemon changes.

---

## Reservation Spec

Requests follow a four-level hierarchy:

```
Request
└── Reservation  (one per independent lifetime)
    └── Allocation  (one per physical device)
        └── Resource  (capacity claimed on that device)
```

- **Request** — the atomic unit. All reservations in a request are granted together or the entire request is rejected. One wire message.
- **Reservation** — a named grouping with its own handle (socket fd). Reservations within a request are granted atomically but released independently. The caller declares reservation boundaries explicitly based on intended lifetime — the daemon does not infer them.
- **Allocation** — one physical device within a reservation and the capacity claimed on it. A reservation can hold multiple allocations; all are released when the reservation handle closes.
- **Resource** — the capacity claimed on an allocation's device (`ram_mb`, `compute_units`). Multiple allocations can target the same physical device as long as combined capacity fits.

### Allocation fields (daemon protocol)

The daemon only accepts explicit allocations. Every allocation names a specific device:

| Field | Description |
| --- | --- |
| name | Required. Physical device name (e.g. `cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780`, `metal:0`, `cpu`) |
| ram_mb | Amount of RAM to claim on this device |
| compute_units | Number of compute units to claim on this device |

### Constraint fields (SDK and CLI)

The SDK and CLI accept device constraints, which a solver resolves to allocations before submitting to the daemon:

| Field | Effect |
| --- | --- |
| type | Device must be this type (`cuda`, `metal`, or `cpu`) |
| name | Device must be this physical device |
| ram_mb | Device must have at least this much available RAM |
| compute_units | Device must have at least this many available compute units |

**Constraint-based (SDK resolves before submitting):**

```
# Single reservation, one allocation: any CUDA device
reservation(constraint(type=cuda))

# Single reservation, two allocations: distributed job, released together
reservation(constraint(type=cuda), constraint(type=cuda))

# Multiple reservations: stable diffusion, freed independently
request(
  reservation("checkpoint", constraint(type=cuda, ram_mb=8192)),
  reservation("lora",       constraint(type=cuda, ram_mb=2048)),
)

# Multiple reservations: TTS actors, freed one by one
request(
  reservation("actor-1", constraint(type=cuda, ram_mb=4096)),
  reservation("actor-2", constraint(type=cuda, ram_mb=4096)),
  reservation("actor-3", constraint(type=cuda, ram_mb=4096)),
  reservation("actor-4", constraint(type=cuda, ram_mb=4096)),
  reservation("actor-5", constraint(type=cuda, ram_mb=4096)),
)
```

**Pinned (orchestrator has already chosen the cards):**

```
# Orchestrator consolidates onto one card for co-location
request(
  reservation("checkpoint", constraint(name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780, ram_mb=8192)),
  reservation("lora",       constraint(name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780, ram_mb=2048)),
)

# Orchestrator spreads actors across two cards for throughput
request(
  reservation("actor-1", constraint(name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780, ram_mb=4096)),
  reservation("actor-2", constraint(name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780, ram_mb=4096)),
  reservation("actor-3", constraint(name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781, ram_mb=4096)),
  reservation("actor-4", constraint(name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781, ram_mb=4096)),
  reservation("actor-5", constraint(name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781, ram_mb=4096)),
)
```

A name constraint resolves to exactly that device if the capacity fits. If the request then fails at the daemon due to a race, the orchestrator re-reads status and re-plans. The daemon owns atomicity; the orchestrator owns strategy.

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

- Path: `--socket PATH` flag, with [shared platform-aware discovery](INSTALL.md#socket-and-config-discovery)
- Permissions: `0660` by default
- Group: `--socket-group GROUP` sets the socket file's group, allowing non-root users

in that group to connect without making the socket world-writable

This two-path approach requires no systemd library — just a check on `LISTEN_FDS`.

### Ledger

The ledger is an in-memory mapping of each device to its total and currently available `(ram_mb, compute_units)`. Active reservations are kept in memory with their granted device assignments. No database. State is lost on daemon restart; all reservations are implicitly released.

### Request / Response Protocol

Line-delimited JSON over the UDS. Each message is one JSON object followed by `\n`.

**Reserve request:**

A single request contains one or more reservations. Each reservation has a label and zero or more allocations. Each allocation names a device and the capacity to claim on it.

```json
{
  "op": "reserve",
  "reservations": [
    {
      "label": "checkpoint",
      "allocations": [
        { "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782", "ram_mb": 8192, "compute_units": 50 }
      ]
    },
    {
      "label": "lora",
      "allocations": [
        { "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782", "ram_mb": 2048, "compute_units": 0 }
      ]
    }
  ]
}
```

The daemon protocol is fully explicit — every allocation names a specific device. Constraint resolution happens in the SDK or CLI before this request is submitted. All allocations across all reservations are checked atomically.

**Reserve response (success):**

The response contains one entry per reservation, each with its own `reservation_id` and its allocations. `reservation_id` identifies the handle — closing the corresponding socket releases that reservation.

```json
{
  "ok": true,
  "reservations": [
    {
      "reservation_id": "a3f9c2",
      "label": "checkpoint",
      "allocations": [
        { "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782", "ram_mb": 8192, "compute_units": 50 }
      ]
    },
    {
      "reservation_id": "b7e1d4",
      "label": "lora",
      "allocations": [
        { "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782", "ram_mb": 2048, "compute_units": 0 }
      ]
    }
  ]
}
```

Each granted allocation is the requested record echoed back unchanged. The grant carries nothing the request did not: the device name is the device's whole identity, and for CUDA it already contains the UUID that `CUDA_VISIBLE_DEVICES` and torch understand.

Each reservation requires its own lease fd. For each granted reservation the daemon creates a `socketpair()` — two connected ends. The daemon retains one end and sends the other to the client via `SCM_RIGHTS` (Unix domain socket ancillary data) alongside the JSON response. The client receives N file descriptors — one per reservation — in the same message as the grant.

The daemon watches its end of each socketpair with `select`. When the client closes its end (explicit release, process exit, or crash), the daemon's end becomes readable with EOF — this is the release signal. No messages are exchanged over the lease fd; it exists solely as a close-detectable handle. The transport socket is closed after the exchange and plays no further role.

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
      "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780",
      "type": "cuda",
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
      "allocations": [
        { "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782", "ram_mb": 8192, "compute_units": 0 }
      ]
    },
    {
      "id": "b7e1d4",
      "label": "lora",
      "created_at": 1748000000.0,
      "allocations": [
        { "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782", "ram_mb": 2048, "compute_units": 0 }
      ]
    }
  ]
}
```

### Behavior on Reserve

All allocations across all reservations are checked atomically. The daemon does no constraint solving — every allocation names a specific device.

1. For each allocation, check that the named device exists and has sufficient available `ram_mb` and `compute_units`.
2. If any allocation cannot be satisfied: respond `ok: false`, no resources are changed.
3. If all allocations are satisfied: subtract capacity from each named device, record all reservations, respond with the same allocations.
4. Hold each reservation's resources until its lease socketpair fd is closed.

### Lease Lifetime and Connection Model

Each reservation is represented by a separate open socket connection. The daemon releases that reservation's resources when its socket closes — whether by explicit release, process exit, or crash. Multiple reservations from the same request have independent sockets and independent lifetimes.

**Connection lifecycle:**

```
client opens a single transport socket, sends reserve request
  → daemon checks all allocations atomically
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
with Reservation("worker", DeviceConstraint("cuda", ram_mb=16384)) as r:
    pid = os.fork()
    if pid == 0:
        # child: still holds the reservation via inherited fd
        run_worker(r.torch_devices()[0])
        os._exit(0)
    # parent: reservation still live while child runs
    os.waitpid(pid, 0)
# both parent and child have exited — reservation released
```

To start a worker that holds a copy of the reservation from Python:

```python
with Reservation("worker", DeviceConstraint("cuda")) as r:
    worker = r.popen(["/path/to/worker-env/bin/python", "-m", "worker"])
    worker.wait()
    # the parent still holds its copy; the lease ends when the block exits
```

`popen()` is `subprocess.Popen` with the lease fd added to `pass_fds` and the same environment variables `reserve` sets, so the worker can call `adopt_inherited_lease()`. `popen_python(env_dir, module, *args)` builds on it for a Python worker in another environment: it runs that environment's `bin/python -P -m module`, with the environment's `bin` first on `PATH`, `VIRTUAL_ENV` set, and `PYTHONPATH` and `PYTHONHOME` dropped, so the worker's imports come only from its own environment. `lease_fd` and `child_env()` expose those two pieces for other launchers, including a plain exec: make `lease_fd` inheritable and `execvpe` the worker with `child_env()`. The lease socket attribute itself is private.

### Finding reservation owners

The daemon has no way to verify which process holds a reservation. Lease fds are socketpair ends held by the client process — they are not connections to the daemon socket and will not appear in `lsof /var/run/reserved.sock`. To find which process holds a lease fd, use `reserve-status --json` to get the reservation ID, then search for the fd numerically:

```bash
# Find all processes with an open fd matching the lease
lsof -p <PID> | grep -i sock
```

Alternatively, cross-reference `reserve-status` reservation ages and labels with known process start times to identify the owner.

---

## `reserve` CLI

Acquires a single reservation and runs a command under it.

```
reserve LABEL [DEVICE ...] [--socket PATH] -- CMD [ARGS...]

Arguments:
  LABEL             Required. Human-readable name for this reservation (e.g. "train", "infer").
                    Shown in reserve-status output and daemon logs.
  DEVICE            Constraint for one allocation. Comma-separated fields:
                    type=TYPE, name=NAME, ram=MB, compute=N.
                    All fields optional. Repeat for multiple allocations. Omit entirely for
                    no device constraint.

Options:
  --solver NAME     Constraint solver to use: consolidate (default) or spread.
  --socket PATH     Daemon socket path [shared platform-aware default; see INSTALL.md]
```

All allocations belong to the single reservation and are granted atomically — all or none.

```bash
# No device constraint
reserve train -- python infer.py

# Any CUDA device
reserve infer type=cuda -- python infer.py

# Metal device (Apple Silicon)
reserve infer type=metal -- python infer.py

# Needs 30GB VRAM (rules out 3090s)
reserve train type=cuda,ram=30720 -- python train.py

# Pin to a specific device by name
reserve infer name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782 -- python infer.py

# Two CUDA allocations (distributed job)
reserve distributed type=cuda type=cuda -- python distributed.py

# One CUDA allocation with 40 compute units + one CPU allocation
reserve pipeline type=cuda,compute=40 type=cpu,ram=32768 -- python pipeline.py
```

**What it does:**

1. Runs the constraint solver against current device status to produce an explicit request, then submits it to the daemon.
2. If the daemon is not running: prints an error to stderr and exits 2.
3. If the daemon rejects (insufficient resources): prints an error to stderr and exits 1.
4. If granted: sets environment variables, then replaces itself with CMD via `exec`.

CMD runs directly — no wrapper process sitting around it.

**Environment variables set before exec:**

| Variable | Value |
| --- | --- |
| CUDA_VISIBLE_DEVICES | Distinct GPU UUIDs of the granted CUDA devices in first-grant order, e.g. `GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782` |
| RESERVED_DEVICE | Physical name of the first granted accelerator device, e.g. `cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782` or `metal:0` |
| RESERVED_DEVICES | Comma-separated physical names of all granted devices |
| RESERVED_GRANTED_ALLOCATIONS | JSON array of the granted allocations (`name`, `ram_mb`, `compute_units`) for SDK adoption |
| RESERVED_ID | Reservation ID (opaque string) |
| RESERVED_FD | File descriptor number of the lease socketpair end for this reservation, e.g. `3` |

`RESERVED_FD` is one fd regardless of how many allocations were granted — allocations share a reservation and a single lease fd. It is not needed by typical programs (they simply inherit the fd and hold the lease by staying alive). It is useful for wrapper scripts that need to explicitly close the fd early, debugging tools that want to identify the lease fd, or fd-preserving child launchers that need to pass it along. The SDK's `Reservation.popen()` and `child_env()` produce the same variables from the same implementation, so a worker adopts a lease identically whether `reserve` or a Python parent launched it.

`CUDA_VISIBLE_DEVICES` is an NVIDIA-defined environment variable that the CUDA runtime, PyTorch, and other frameworks read at startup to determine which physical GPUs are visible to the process. By setting it before exec, `reserve` hooks into this existing operator infrastructure — the exec'd process sees only the devices it was granted, with no awareness of `reserved` or the full physical device layout.

Two distinct concepts:
- **Physical identity** — the GPU UUID in the device name (`cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782`). This is what goes into `CUDA_VISIBLE_DEVICES`; the driver accepts UUIDs there directly.
- **Logical index** — the 0-based position within what `CUDA_VISIBLE_DEVICES` exposes to the process. This is what `torch.device("cuda:0")` refers to inside the child.

For example, if the daemon grants `cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782`, `reserve` sets `CUDA_VISIBLE_DEVICES=GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782`. Inside the process the CUDA runtime exposes one device, so `torch.device("cuda:0")` targets that card. A process granted cards 2 and 3 sees both UUIDs in the mask, in grant order; inside the process `torch.device("cuda:0")` is card 2 and `torch.device("cuda:1")` is card 3. The SDK's `torch_device()` does this lookup by asking torch for each visible device's UUID, so it needs no knowledge of the mask.

Metal does not have an equivalent device-selection environment variable — there is only one logical MPS device per system. `reserved` coordinates access to the Metal device via the reservation mechanism; the exec'd process selects it in code (e.g. `torch.device("mps")`).

**Why `exec` instead of running CMD as a subprocess:** `exec` replaces `reserve` with CMD entirely — no wrapper process lingers waiting for CMD to finish. `reserve` clears `FD_CLOEXEC` on the lease fd before exec, so CMD inherits it and holds the lease for its own lifetime. The reservation ends when CMD (and any descendants) exit, not when `reserve` would have exited.

---

## Python SDK

```python
from reserved.sdk import Request, Reservation, DeviceConstraint, ReservationError
```

**Single reservation — use `Reservation` directly as a context manager:**

```python
# Any CUDA device
with Reservation("infer", DeviceConstraint("cuda")) as r:
    model.to(r.torch_devices()[0])
    run_model()

# Metal device (Apple Silicon)
with Reservation("infer", DeviceConstraint("metal")) as r:
    model.to(r.torch_devices()[0])   # torch.device("mps")
    run_model()

# Needs 30GB VRAM
with Reservation("infer", DeviceConstraint("cuda", ram_mb=30720)) as r:
    model.to(r.torch_devices()[0])
    ...

# Two CUDA devices in one reservation — released together
with Reservation("distributed", DeviceConstraint("cuda"), DeviceConstraint("cuda")) as r:
    devices = r.torch_devices()   # one process-local torch.device per allocation, in constraint order
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
    (ckpt_device,) = checkpoint.torch_devices()
    load_checkpoint(ckpt_device)
    with lora:
        (lora_device,) = lora.torch_devices()   # may be the same physical device as checkpoint
        load_lora(lora_device)
        run_inference()
    # lora released here, VRAM returned to pool
# checkpoint released here

# TTS pipeline: 5 actors acquired atomically, freed one by one as each finishes
actors = Request(*[
    Reservation(f"actor-{i}", DeviceConstraint("cuda", ram_mb=4096))
    for i in range(5)
]).acquire()

for actor, line in zip(actors, script):
    (device,) = actor.torch_devices()
    render(device, line)
    actor.release()   # frees this actor's VRAM immediately
```

**One input language: constraints**

Every argument to `Reservation` is a `DeviceConstraint`. The SDK runs a solver against current device status to resolve each constraint to a `DeviceAllocation` before submitting to the daemon. `ConsolidateSolver` is the default. An orchestrator that has already decided placement pins each card with a `name` constraint; the solver then only checks that the capacity fits.

```python
# Solver picks the device
with Reservation("infer", DeviceConstraint("cuda", ram_mb=8192)) as r:
    ...

# Explicit solver choice
with Reservation("infer", DeviceConstraint("cuda", ram_mb=8192), solver=SpreadSolver()) as r:
    ...

# Orchestrator pins a card
with Reservation("infer", DeviceConstraint(name="cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780", ram_mb=8192)) as r:
    ...

# Orchestrator controls packing across reservations
req = Request(
    Reservation("checkpoint", DeviceConstraint(name="cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780", ram_mb=8192)),
    Reservation("lora",       DeviceConstraint(name="cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780", ram_mb=2048)),
)
checkpoint, lora = req.acquire()
```

**`Reservation(label, *constraints, solver=ConsolidateSolver())`**

Label is required and positional. Every other positional argument is a `DeviceConstraint`; anything else is a `TypeError`. All allocations share the same lease fd and are released together.

**`DeviceConstraint(type, *, name=None, ram_mb=0, compute_units=0)`**

Every field is optional. `type` may be omitted (`None`) to match any device type; `name` pins one physical device; `ram_mb` and `compute_units` must fit in the device's available capacity.

**`Reservation` fields and methods after `acquire()`:**

- `reservation_id: str`
- `label: str`
- `devices: list[DeviceAllocation]` — the granted allocations, parallel to the constraints passed at construction
- `torch_devices() -> list[torch.device]` — `[torch_device(a) for a in self.devices]`
- `release()` — closes the lease fd, releasing all allocations in this reservation
- `check()` — see below

**`torch_device(allocation) -> torch.device`**

Resolves one granted allocation to the process-local PyTorch device. For CUDA it asks torch for each visible device's UUID and returns the index whose UUID matches the allocation's name, so the result is right whether `CUDA_VISIBLE_DEVICES` is unset, set by `reserve`, or set by a scheduler; the SDK never reads or writes the mask. Metal resolves to `torch.device("mps")` and CPU to `torch.device("cpu")`. PyTorch is expected to be installed in the calling process; the package does not declare it, and no other SDK operation imports it. Resolving a CUDA grant initializes CUDA, so call it in the process that will use the GPU.

**`check()`**

Call at natural checkpoints between units of work — between training epochs, between inference calls, between actor lines. Performs a non-blocking read on the client's end of the lease socketpair. If the daemon has closed its end (restart or fatal error), the read returns EOF and `ReservationExpiredError` is raised. Returns immediately if the lease is healthy.

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

1. Constructor kwargs (`socket_path=`)
2. Environment variable `RESERVED_SOCKET` for socket path
3. `$XDG_RUNTIME_DIR/reserved.sock` when set.
4. `/tmp/reserved-<uid>.sock` on macOS, otherwise `/var/run/reserved.sock`.

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
  cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780  cuda  available:  8192/24576 MB  100/100 compute
  cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781  cuda  available:  4096/24576 MB  100/100 compute
  cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782  cuda  available: 49152/49152 MB  100/100 compute
  cpu  cpu  available: 65536/65536 MB

Reservations:
  a3f9c2  label=actor-1  age=3m12s
    cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780  4096 MB
  b7e1d4  label=actor-2  age=3m12s
    cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780  4096 MB
  c2a8f1  label=actor-3  age=3m12s
    cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780  4096 MB
  d9b3e7  label=actor-4  age=3m12s
    cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781  4096 MB
  e4f6c9  label=actor-5  age=3m12s
    cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781  4096 MB
```

Three actors are packed onto the first card, two onto the second. As each actor finishes its lines, its reservation is freed and that VRAM returns to the pool — the other actors are unaffected.

With `--json`, prints the raw status JSON for use in scripts:

```bash
reserve-status --json | jq '.reservations | length'
```

---

## Deployment

The authoritative installation and operations guide is [INSTALL.md](INSTALL.md).
It includes Linux system and user systemd services, a macOS launchd user agent,
source installation into virtual environments, client setup, verification,
upgrades, and removal. Service definitions live in [deploy/](deploy/).

The daemon and clients share the socket discovery rules documented in
[the installation guide](INSTALL.md#socket-and-config-discovery). A daemon
started by systemd adopts its supplied listener. Explicit config selections
must exist; an invalid explicit path does not fall back to another config.

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

### Language Boundaries

The constraint language never crosses a process boundary. `DeviceConstraint` and the solver classes are in-process Python objects only — they are never serialized. Only explicit requests of `DeviceAllocation` records hit the wire.

The wire format is single-line JSON over the Unix domain socket. Each message is one JSON object terminated by `\n`. `protocol.py` owns all serialization and deserialization of explicit requests and responses. Constraint types never appear in `protocol.py`.

```
In-process (SDK):          DeviceConstraint → Solver → ReserveRequest of DeviceAllocations
                                                              ↓
Wire (protocol.py):                               single-line JSON \n
                                                              ↓
In-process (daemon):                              core.grant(ledger, request)
```

### Implementation Language

All components — daemon, CLI tools, and SDK — are implemented in Python. The `reserve`, `reserve-status`, and `reserved-generate-config` CLIs are Python scripts installed as executables. Python is required for SCM_RIGHTS fd passing, socketpair management, and the constraint solver logic.

### Module Structure

```
reserved/
  core.py          # pure functions, no IO — the functional core; DeviceAllocation, cuda_uuid()
  daemon_state.py  # pure state machine: DaemonState + Event -> DaemonState + Effects
  daemon.py        # IO shell — sockets, select loop, SCM_RIGHTS, effect application
  protocol.py   # serialization of explicit requests/responses to single-line JSON only
  sdk/
    client.py   # Reservation, Request, DeviceConstraint, torch_device()
    solvers.py  # ConsolidateSolver, SpreadSolver
    errors.py   # ReservationError, DaemonNotRunningError, ReservationExpiredError
  bin/
    reserve.py          # CLI: constraint solver + exec wrapper
    reserve_status.py   # CLI: status query
    generate_config.py  # CLI: config from the hardware torch detects
```

### Functional Core (`core.py`)

All business logic lives here as pure functions. No IO, no sockets, no global state. Every function takes a `Ledger` and returns a new `Ledger` plus a result. The ledger is an immutable frozen dataclass — functions never mutate it, they produce a replacement.

```python
@dataclass(frozen=True)
class Device:
    name: str
    type: str
    ram_mb: int
    compute_units: int

@dataclass(frozen=True)
class DeviceAllocation:
    name: str
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

`grant()` checks all allocations atomically. If any named device lacks sufficient capacity, it returns the original ledger unchanged and a failure result. If all allocations are satisfied, it returns a new ledger with capacity subtracted and a success result carrying the same allocations.

This design means the core is fully testable as plain function calls — no daemon, no sockets, no hardware required.

```python
# example test
ledger = load_ledger({"devices": [{"name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780", "type": "cuda", "ram_mb": 24576, ...}]})
ledger2, result = grant(ledger, request)
assert result.ok
assert ledger2.availability[0].available_ram_mb == 24576 - 8192
# original ledger is unchanged
assert ledger.availability[0].available_ram_mb == 24576
```

### State Layer (`daemon_state.py`) and IO Shell (`daemon.py`)

Between the core and the sockets sits a pure state machine. One daemon step takes the logical daemon state and one observed event, and returns the new state plus the effects the IO shell must perform:

```text
DaemonState + Event -> DaemonState + Effects
```

The state holds the ledger and the open transports and leases by logical id. It never holds a socket or a file descriptor. The IO shell (`daemon.py`) owns the real sockets, runs the `select` loop, translates readable descriptors into events, calls `daemon_state.step()`, and applies the effects: creating lease socketpairs, sending lease fds with `SCM_RIGHTS`, writing responses, and closing sockets.

```python
state = daemon_state.initial_state(core.load_ledger(config))

# on any socket activity, translated to an event:
state, effects = daemon_state.step(state, event)
apply(effects)   # the only place sockets are touched
```

The shell contains no allocation logic and no state transitions — only IO, descriptor lifecycle, and effect application. The state layer is tested as plain function calls with no sockets; the shell is tested against a real daemon process. See [ARCHITECTURE.md](ARCHITECTURE.md#daemon-internals) for the events, effects, and flows.
