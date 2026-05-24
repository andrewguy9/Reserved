# reserved — Resource Reservation Daemon

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
  a3f9c2  pid=12345  label=train  age=2m14s
    cuda:1  8192 MB  50 compute
```

### SDKs

**Python (`reserve`)** — a `Reservation` context manager that acquires a reservation on enter and releases it on exit. Suitable for notebooks, training scripts, or any Python process that needs to claim hardware for a block of work.

```python
from reserve import Reservation, DeviceConstraint

with Reservation(DeviceConstraint("cuda", ram_mb=16384), label="train") as r:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(r.devices[0].device_index)
    run_training()
```

---

## Resource Config (`reserved.json`)

Defines the set of physical devices on this machine. The daemon loads this at startup and never re-reads it while running.

```json
{
  "devices": [
    { "name": "cuda:0", "type": "cuda", "model": "rtx3090", "ram_mb": 24576, "compute_units": 100 },
    { "name": "cuda:1", "type": "cuda", "model": "rtx3090", "ram_mb": 24576, "compute_units": 100 },
    { "name": "cuda:2", "type": "cuda", "model": "a6000",   "ram_mb": 49152, "compute_units": 100 },
    { "name": "cpu",    "type": "cpu",                       "ram_mb": 65536 }
  ]
}
```

Each device has:

- `name` — unique identifier, returned in responses and set as `RESERVED_DEVICE`
- `type` — `cuda` or `cpu`
- `model` — optional human label for the hardware class (e.g. `rtx3090`, `a6000`).

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
| type | Device must be this type (cuda or cpu) |
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
  "pid": 12345,
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

`devices` is parallel to the request `devices` array — `devices[i]` is the physical device assigned for constraint `i`. `device_index` is the remapped CUDA index for `CUDA_VISIBLE_DEVICES` (0-based, counting only CUDA devices in this reservation). Non-CUDA devices have `device_index: -1`.

**Reserve response (failure):**

```json
{
  "ok": false,
  "error": "insufficient resources"
}
```

**Release request:**

```json
{
  "op": "release",
  "reservation_id": "a3f9c2"
}
```

**Release response:**

```json
{
  "ok": true
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
      "pid": 12345,
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

**File descriptor inheritance follows standard POSIX rules:**

`fork()` duplicates all open file descriptors into the child. Both parent and child hold
the reservation. The reservation lives until the last copy of the fd is closed — so a
process that forks worker children can exit itself and the reservation stays alive until
all workers finish.

`exec()` behavior depends on the `O_CLOEXEC` flag. Python sets this by default on all
sockets. `reserve` calls `os.set_inheritable(fd, True)` on the socket before exec so
the child process inherits it. The exec'd program holds the reservation for its entire
lifetime — including any children it forks — with no knowledge of `reserved`. It just
runs with `CUDA_VISIBLE_DEVICES` already set.

For programs that want to pass a reservation through their own exec boundary, the same
pattern applies: call `os.set_inheritable(fd, True)` before exec and the child inherits
the open socket. For deeply nested process trees where multiple workers each hold a copy
of the fd, the reservation lives until the last one exits.

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
2. If the daemon rejects: prints an error to stderr and exits 1.
3. If granted: sets environment variables, then replaces itself with CMD via `exec`.

CMD runs directly — no wrapper process sitting around it.

**Environment variables set before exec:**

| Variable | Value |
| --- | --- |
| CUDA_VISIBLE_DEVICES | Comma-separated remapped CUDA indices, e.g. 0 or 0,1 |
| RESERVED_DEVICE | Physical name of the first granted CUDA device, e.g. cuda:2 |
| RESERVED_DEVICES | Comma-separated physical names of all granted devices |
| RESERVED_ID | Reservation ID (opaque string) |

`CUDA_VISIBLE_DEVICES` is always remapped to start from 0 regardless of which physical GPUs were assigned. A process granted `cuda:2` and `cuda:3` sees them as `0` and `1`.

**Why `exec` instead of running CMD as a subprocess:**`exec` replaces `reserve` with CMD entirely. The socket connection drops immediately (closed by `exec`), releasing the reservation the moment CMD exits. No wrapper lingers.

---

## Python SDK

```python
from reserve import Reservation, DeviceConstraint, ReservationError

# Any CUDA device
with Reservation(DeviceConstraint("cuda"), label="infer") as r:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(r.devices[0].device_index)
    run_model()

# Needs 30GB VRAM — rules out 3090s
with Reservation(DeviceConstraint("cuda", ram_mb=30720)) as r:
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
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(d.device_index) for d in r.devices if d.device_index >= 0
    )
    run_distributed()

# One A6000 + CPU RAM
with Reservation(
    DeviceConstraint("cuda", model="a6000"),
    DeviceConstraint("cpu", ram_mb=32768),
) as r:
    gpu = r.devices[0].device_name   # e.g. "cuda:2"
    cpu = r.devices[1].device_name   # "cpu"

# Non-context usage
r = Reservation(DeviceConstraint("cuda", ram_mb=8192))
r.acquire()   # raises ReservationError if unavailable
try:
    ...
finally:
    r.release()
```

**`DeviceConstraint(type, *, model=None, ram_mb=0, compute_units=0)`**

All fields except `type` are optional. Omitting a field means unconstrained on that dimension. `type` may also be omitted entirely (pass `None`) to match any device type.

**`Reservation` fields after `acquire()`:**

- `reservation_id: str`
- `devices: list[GrantedDevice]` — parallel to the constraints passed at construction

**`GrantedDevice` fields:**

- `device_name: str` — physical device name, e.g. `"cuda:2"`
- `device_index: int` — remapped CUDA index (0-based within this reservation's CUDA

devices). Non-CUDA devices have `device_index = -1`.

**Errors:**

- `ReservationError` — raised by `acquire()` when the daemon rejects the request.
- `DaemonNotRunningError` — raised when the socket does not exist or connection refused.

**Connection:**`acquire()` opens a connection to the daemon socket and keeps it open for the lifetime of the reservation. The open connection is the lease — the daemon releases the resource when the connection closes. `release()` / `__exit__` sends the release request and then closes the connection. If the process dies before calling `release()`, the kernel closes the socket and the daemon releases automatically.

**SDK configuration** (in priority order):

1. Constructor kwargs (`socket_path=`, `label=`)
2. Environment variable `RESERVED_SOCKET` for socket path
3. Default: `/var/run/reserved.sock`

---

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
  a3f9c2  pid=12345  label=encode  age=4m32s
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

Clients holding a reservation at the time of restart are left with a broken socket. They will not know their reservation was lost until they next attempt to write to it (at `release()` time), at which point they receive `EPIPE`.

Well-designed clients should periodically check that their reservation is still valid by attempting a read on the socket. A closed connection returns EOF immediately, signalling that the daemon has restarted and the reservation is gone. The client can then fail fast rather than continue using hardware it no longer holds.

The daemon logs all active reservations at shutdown (`SIGTERM`) so operators can see what work will be disrupted before restarting.
