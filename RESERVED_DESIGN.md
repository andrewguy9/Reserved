# reserved — Resource Reservation Daemon

## TLDR

`reserved` is a single-machine, best-effort resource reservation system for physical hardware (GPUs, CPU, RAM). It succeeds or fails immediately — there is no queue.

1. **Reservation system** — processes claim a slice of physical hardware capacity (device type, VRAM, compute budget). The daemon grants or rejects immediately.
2. **Handle-based lifetime** — a reservation is an open Unix Domain Socket connection. When the process dies for any reason, the socket closes and resources are freed automatically. No cleanup code required.
3. **Optional SDK** — a Python SDK provides a `Reservation` context manager for acquiring reservations and a `check()` method developers should call at natural checkpoints (between epochs, between inferences). If the daemon has restarted and the reservation is gone, `check()` raises `ReservationExpiredError` so the program can exit gracefully.
4. **Daemon** — a lightweight process that holds the resource ledger in memory, grants/rejects reservation requests, and releases them when connections close. State is lost on restart.
5. **Optional status API** — an HTTP endpoint and `reserve-status` CLI for operators to inspect current device capacity and active reservations.
6. **Convenience wrapper** — a `reserve` shell script that wraps any non-reservation-aware program, acquiring a reservation and exec'ing the target command under it.

---

## What is reserved?

As GPU based AI applications become popular, our operating systems don't offer resource management tooling to help users run applications without interfering with each other. Reserved offers an opt-in mechanism for process resource coordination by "reserving" resources for a task, and then releasing them when finished.

A reservation is a claim on a slice of hardware — a specific device model, an amount of VRAM, a compute budget. The daemon grants the claim if the hardware is available, or rejects it immediately if not. The claim is backed by an open Unix Domain Socket between the `reserved` daemon and the task program, so that in the event of a crash the resources are released. This also means that we can use normal socket semantics with fork/exec to allow non-reserved aware programs to leverage the same reservation system. The claim is held for exactly as long as the socket is held open. If the process dies for any reason, the socket is released automatically — no cleanup required.

---

## Components

### Daemon (`reserved`)

The daemon process that runs on the machine. It loads a JSON config describing the available hardware, then listens on a Unix domain socket for reservation requests. It holds the resource ledger in memory, grants or rejects requests, and releases reservations automatically when a client connection closes.

Optionally serves an HTTP endpoint for monitoring (`--http PORT`).

### Scripts

**`reserve`** — requests a reservation and execs a command under it. Sets `CUDA_VISIBLE_DEVICES` and other env vars before handing off to the command. The reservation is released when the command exits.

```bash
reserve --device cuda:model=a6000 -- python train.py
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

## Constraint-Based Matching

A reservation request specifies one or more **device constraints** — one per physical device needed. Each constraint is a set of filters; the daemon finds a device satisfying all of them. Unspecified fields are unconstrained.

| Constraint field | Effect |
| --- | --- |
| type | Device must be this type (`cuda`, `metal`, or `cpu`) |
| model | Device must have this model label |
| ram_mb | Device must have at least this much available RAM |
| compute_units | Device must have at least this many available compute units |

All constraints in a request are resolved atomically — either every device constraint is satisfied simultaneously, or the whole request is rejected.

**Examples:**

```
# Any CUDA device, don't care which
type=cuda

# Needs 30GB VRAM — rules out 3090s (24GB), matches A6000 (48GB)
type=cuda  ram_mb=30720

# Specifically a 3090, claim the whole thing
type=cuda  model=rtx3090

# A 3090 with partial capacity
type=cuda  model=rtx3090  ram_mb=8192  compute_units=30

# Two CUDA devices of any kind (e.g. a distributed training job)
[type=cuda, type=cuda]

# One A6000 for training + one CPU allocation
[type=cuda model=a6000, type=cpu ram_mb=32768]
```

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

```json
{
  "op": "reserve",
  "label": "my-training-run",
  "devices": [
    { "type": "cuda", "model": "a6000", "ram_mb": 16384, "compute_units": 50 },
    { "type": "cpu",  "ram_mb": 32768 }
  ]
}
```

Each entry in `devices` is a constraint — all fields are optional. An empty object `{}` matches any device with any available capacity. All constraints are resolved atomically.

**Reserve response (success):**

```json
{
  "ok": true,
  "reservation_id": "a3f9c2",
  "devices": [
    { "device_name": "cuda:2", "device_index": 0 },
    { "device_name": "cpu",    "device_index": -1 }
  ]
}
```

`devices` is parallel to the request `devices` array — `devices[i]` is the physical device assigned for constraint `i`. `device_index` is a remapped 0-based index within this reservation's CUDA devices (e.g. for use in `CUDA_VISIBLE_DEVICES`). Metal and CPU devices have `device_index: -1` — there is no index-based selection for those types.

**Reserve response (failure):**

```json
{
  "ok": false,
  "error": "insufficient resources"
}
```

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
      "label": "my-training-run",
      "created_at": 1748000000.0,
      "devices": [
        { "device_name": "cuda:2", "ram_mb": 16384, "compute_units": 50 },
        { "device_name": "cpu",    "ram_mb": 32768 }
      ]
    }
  ]
}
```

### Behavior on Reserve

1. For each constraint in the request, find a device satisfying all specified fields

and having sufficient available capacity. All constraints are evaluated together — earlier assignments reduce available capacity for later ones.

1. If any constraint cannot be satisfied: respond `ok: false`, no resources are changed.
2. If all constraints are satisfied: record the reservation, subtract capacity from each

assigned device, respond with the assignment list.

1. Keep the client connection open. The open connection *is* the lease.

### Lease Lifetime and Connection Model

The reservation is represented by an open file descriptor to the daemon socket. The
daemon releases the reservation when that file descriptor is closed — whether the client
called `release`, the process exited, or it crashed. No PID tracking, no TTL polling,
no platform-specific monitoring code.

**Connection lifecycle:**

```
client connects
  → sends reserve request
  → daemon responds (ok / error)
  → [if ok] fd stays open — this is the lease
  → reservation ends when all copies of the fd are closed
     (explicit release, process exit, crash, or last forked worker exits)
```

The daemon multiplexes all open connections in a single thread using `select`. Each
connection is either a live lease (awaiting close) or a transient operation (status).

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

Acquires a reservation and runs a command under it.

```
reserve [options] -- CMD [ARGS...]

Options:
  --device [TYPE][:model=MODEL][:ram=MB][:compute=N]
                    Constraint for one physical device. Repeat for multiple devices.
                    All fields are optional; unspecified fields are unconstrained.
  --label LABEL     Human-readable label [default: reserve]
  --socket PATH     Daemon socket path [default: /var/run/reserved.sock]
```

All `--device` constraints are resolved atomically — all are granted or none are.

```bash
# Any CUDA device
reserve --device cuda -- python infer.py

# Metal device (Apple Silicon)
reserve --device metal -- python infer.py

# Needs 30GB VRAM (rules out 3090s)
reserve --device cuda:ram=30720 -- python train.py

# Specifically a 3090
reserve --device cuda:model=rtx3090 -- python infer.py

# Two CUDA devices of any kind
reserve --device cuda --device cuda -- python distributed.py

# One A6000 + CPU RAM
reserve --device cuda:model=a6000 --device cpu:ram=32768 -- python pipeline.py
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
from reserve import Reservation, DeviceConstraint, ReservationError

# Any CUDA device
with Reservation(DeviceConstraint("cuda"), label="infer") as r:
    model.to(r.devices[0].torch_device())
    run_model()

# Metal device (Apple Silicon)
with Reservation(DeviceConstraint("metal"), label="infer") as r:
    model.to(r.devices[0].torch_device())   # torch.device("mps")
    run_model()

# Needs 30GB VRAM — rules out 3090s
with Reservation(DeviceConstraint("cuda", ram_mb=30720)) as r:
    model.to(r.devices[0].torch_device())
    ...

# Specifically a 3090 with partial capacity
with Reservation(DeviceConstraint("cuda", model="rtx3090", ram_mb=8192, compute_units=30)) as r:
    ...

# Two CUDA devices, granted atomically
with Reservation(
    DeviceConstraint("cuda"),
    DeviceConstraint("cuda"),
    label="distributed",
) as r:
    devices = r.torch_devices()   # [torch.device("cuda:0"), torch.device("cuda:1")]
    run_distributed(devices)

# One A6000 + CPU RAM
with Reservation(
    DeviceConstraint("cuda", model="a6000"),
    DeviceConstraint("cpu", ram_mb=32768),
) as r:
    gpu, cpu = r.torch_devices()   # torch.device("cuda:0"), torch.device("cpu")
    run_pipeline(gpu, cpu)

```

**`DeviceConstraint(type, *, model=None, ram_mb=0, compute_units=0)`**

All fields except `type` are optional. Omitting a field means unconstrained on that dimension. `type` may also be omitted entirely (pass `None`) to match any device type.

**`Reservation` fields and methods after `acquire()`:**

- `reservation_id: str`
- `devices: list[GrantedDevice]` — parallel to the constraints passed at construction
- `torch_devices() -> list[torch.device]` — returns a `torch.device` for each granted device, in constraint order. Convenience wrapper over `[d.torch_device() for d in self.devices]`.

**`GrantedDevice` fields and methods:**

- `device_name: str` — physical device name, e.g. `"cuda:2"` or `"metal:0"`
- `device_index: int` — remapped 0-based CUDA device index for use with `CUDA_VISIBLE_DEVICES`. Metal and CPU devices have `device_index = -1`.
- `torch_device() -> torch.device` — returns the appropriate `torch.device` for this grant: `torch.device("cuda:N")` for CUDA (using the remapped index), `torch.device("mps")` for Metal, `torch.device("cpu")` for CPU.

**`check()`**

A best-practice method to call at natural checkpoints in long-running work — between training epochs, between inference calls, etc. Performs a non-blocking read on the lease socket. If the daemon has restarted (EOF), raises `ReservationExpiredError`. If the connection is healthy, returns immediately. The SDK cannot poll in the background, so this is the recommended pattern for detecting daemon restarts before committing to another unit of work.

```python
for epoch in range(num_epochs):
    r.check()   # raises ReservationExpiredError if daemon restarted
    train_epoch(model, data)
```

**Errors:**

- `ReservationError` — raised on `__enter__` when the daemon rejects the request.
- `DaemonNotRunningError` — raised when the socket does not exist or connection is refused.
- `ReservationExpiredError` — raised by `check()` when the lease socket has been closed by the daemon (e.g. daemon restarted). The program should treat this as a signal to exit or abort the current work.

**Connection:** `acquire()` opens a connection to the daemon socket and keeps it open for the lifetime of the reservation. The open connection is the lease — the daemon releases the resource when the last copy of the fd is closed, whether by `release()` / `__exit__`, process exit, or crash. `release()` / `__exit__` simply closes the fd; no message is sent to the daemon. If the fd has been inherited by a forked child or passed across exec, closing it in the parent does not release the reservation — the child's copy keeps it alive.

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
  cuda:0  cuda  rtx3090  available: 16384/24576 MB  50/100 compute
  cuda:1  cuda  rtx3090  available: 24576/24576 MB  100/100 compute
  cuda:2  cuda  a6000    available: 49152/49152 MB  100/100 compute
  cpu     cpu            available: 65536/65536 MB

Reservations:
  a3f9c2  label=encode  age=4m32s
    cuda:0  8192 MB  50 compute
```

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
