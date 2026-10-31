# reserved

**Stop playing musical chairs with your GPUs. `reserved` prevents over-allocation crashes by letting jobs reserve hardware before they run.**

A single-machine, best-effort resource reservation daemon for physical hardware (GPUs, CPU, RAM). Processes claim hardware capacity before running, and release it automatically when they exit — even on crash.

`reserved` is opt-in coordination between willing processes. It does not enforce anything against programs that don't participate. It succeeds or fails immediately — there is no queue.

`reserved` is experimental operator tooling. Treat it as cooperative coordination, not a security boundary, quota enforcer, scheduler, or replacement for OS/container isolation.

**New clients and agents: start with [Getting started](GETTING_STARTED.md)** for
installation from GitHub, a walkthrough requiring no GPU, and guidance
on integrating an application or model worker.

For installation and service management, see [INSTALL.md](INSTALL.md). For the detailed design, see [ARCHITECTURE.md](ARCHITECTURE.md). The original proposal is in [RESERVED_DESIGN.md](RESERVED_DESIGN.md).

---

## Motivation

On a multi-GPU workstation, it is common to run several model workloads at the same time: training jobs, long inference runs, local services, notebooks, experiments, and one-off scripts. The operating system can show that processes exist, and GPU tools can show current utilization, but they do not provide a shared application-level plan for who intends to use which resources for how long.

That makes collisions easy. Two programs can start on the same GPU, overcommit VRAM, and crash. In the worst case, a collision ruins a long training job or inference run that had already spent real time getting to that point.

For reservation-aware applications, there is a second problem: planning. If an application wants to dynamically scale the number of model instances it runs, it needs machine-wide metadata about available GPUs, memory budgets, and currently held reservations. That metadata needs to work across applications, not just inside one process.

`reserved` exists to provide that missing coordination layer. The daemon gives the machine a shared resource ledger. The SDK lets applications participate directly and make scaling decisions from current status. The `reserve` CLI lets existing, non-reservation-aware programs participate by wrapping them in a lease-backed reservation.

---

## Install

Requires Python 3.11+. Install from this repository in a virtual environment:

```bash
git clone https://github.com/andrewguy9/Reserved.git
cd Reserved
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The public PyPI package named `reserved` is an unrelated project. Use this
checkout as the package source. For persistent installations, follow the
[Linux systemd or macOS launchd guide](INSTALL.md), which provides fixed paths
and matching service definitions.

For SDK installation into an existing environment, you can also install directly
from GitHub (requires Git):

```bash
python -m pip install "git+https://github.com/andrewguy9/Reserved.git"
```

---

## Architecture at a glance

The `reserved` package has three main parts:

1. **Daemon (`reserved`)** — runs on the machine, tracks configured hardware capacity, grants or rejects reservation requests, and releases resources when reservation lease fds close.
2. **Python SDK (`reserved.sdk`)** — lets reservation-aware applications request resources directly, manage reservation lifetimes with context managers, and check whether a daemon restart invalidated a lease.
3. **Command line wrapper (`reserve`)** — lets non-reservation-aware programs run under a reservation by acquiring resources, setting environment variables, and `exec`-ing the target command.

There is also a small status tool, `reserve-status`, for operators and scripts.

The daemon only accepts explicit device assignments. The SDK and CLI can accept higher-level constraints such as `type=cuda,ram=8192`; they resolve those constraints against daemon status before submitting an explicit reservation request.

```
reservation-aware app ── Python SDK ─┐
                                     ├─ explicit reserve request ── reserved daemon
ordinary command ─────── reserve CLI ┘                              │
                                                                    └─ lease fd keeps reservation alive
```

The daemon keeps all state in memory. A reservation is alive while its lease fd is open; closing the fd, exiting the process, or crashing releases the claimed resources.

---

## Development and testing

Install test tooling:

```bash
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
make test
```

`make check` runs the same suite.

Generate coverage reports:

```bash
make coverage
make htmlcov
```

The HTML report is written to `htmlcov/index.html`. Tests cover the core, protocol,
solvers, daemon state transitions and startup, shared socket discovery, SDK, and
CLI helpers. Service-manager installation and boot/login behavior need platform
verification using [INSTALL.md](INSTALL.md).

---

## Quick start

**1. Create a hardware config** describing the devices on your machine:

```json
{
  "devices": [
    { "name": "cuda:0", "type": "cuda", "model": "rtx3090", "ram_mb": 24576, "compute_units": 100 },
    { "name": "cuda:1", "type": "cuda", "model": "rtx3090", "ram_mb": 24576, "compute_units": 100 },
    { "name": "cpu",    "type": "cpu",                       "ram_mb": 65536 }
  ]
}
```

Create `~/.config/reserved` with `mkdir -p ~/.config/reserved` and save the JSON
as `reserved.json` there (or pass `--config PATH`). Adapt the devices and budgets
to your hardware. For a CPU-only check, copy [examples/cpu.json](examples/cpu.json).

**2. Start the daemon:**

```bash
reserved
```

The daemon runs in the foreground. On Linux it uses
`$XDG_RUNTIME_DIR/reserved.sock` when available; on macOS its fallback is
`/tmp/reserved-<uid>.sock`. Clients use the same discovery rules. On Linux
without `XDG_RUNTIME_DIR`, choose a writable path explicitly with
`export RESERVED_SOCKET="/tmp/reserved-$(id -u)-test.sock"` in both shells, or use the
[system service](INSTALL.md#linux-system-service).

**3. Check status:**

In a second shell, activate the same virtual environment first. If you selected
a custom socket, export the same `RESERVED_SOCKET` there:

```bash
reserve-status
```

---

## Command line wrapper: `reserve`

`reserve` wraps any command under a reservation. Use it for programs that do not know about the SDK. It acquires a single reservation, sets environment variables for the child process, then replaces itself with the target command.

```
reserve LABEL [SLOT ...] [--solver consolidate|spread] [--socket PATH] -- CMD [ARGS...]
```

Each `SLOT` is a comma-separated set of constraints:

| Field | Effect |
| --- | --- |
| `type=TYPE` | Device type: `cuda`, `metal`, or `cpu` |
| `model=MODEL` | Device model label (e.g. `rtx3090`) |
| `ram=MB` | Minimum available RAM in MB |
| `compute=N` | Minimum available compute units |
| `name=NAME` | Pin to a specific device by name |

```bash
# Any CUDA device
reserve infer type=cuda -- python infer.py

# Apple Silicon (Metal/MPS)
reserve infer type=metal -- python infer.py

# Needs 30GB VRAM
reserve train type=cuda,ram=30720 -- python train.py

# Specific model
reserve infer type=cuda,model=rtx3090 -- python infer.py

# Pin to a specific device
reserve infer name=cuda:2 -- python infer.py

# Two CUDA devices (distributed job)
reserve distributed type=cuda type=cuda -- python distributed.py

# Spread across cards for throughput
reserve pipeline --solver spread type=cuda type=cuda -- python pipeline.py
```

**Environment variables set before exec:**

| Variable | Value |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | Physical CUDA ordinals of granted devices (e.g. `2` or `2,3`) |
| `RESERVED_DEVICE` | Physical name of the first granted accelerator |
| `RESERVED_DEVICES` | Comma-separated physical names of all granted devices |
| `RESERVED_ID` | Reservation ID |
| `RESERVED_FD` | File descriptor number of the lease fd |

---

## Python SDK: `reserved.sdk`

Use the SDK when an application can participate in reservations directly. The SDK talks to the daemon, holds the lease fd for the lifetime of each `Reservation`, and exposes granted devices to application code.

```python
from reserved.sdk.client import Reservation, Request, DeviceConstraint, ExplicitSlot
from reserved.sdk.errors import ReservationError, DaemonNotRunningError, ReservationExpiredError
```

### Single reservation

```python
# Any CUDA device
with Reservation("infer", DeviceConstraint("cuda")) as r:
    model.to(r.devices[0].torch_device())
    run_model()

# Metal (Apple Silicon)
with Reservation("infer", DeviceConstraint("metal")) as r:
    model.to(r.devices[0].torch_device())   # torch.device("mps")
    run_model()

# Needs 30GB VRAM
with Reservation("infer", DeviceConstraint("cuda", ram_mb=30720)) as r:
    model.to(r.devices[0].torch_device())
    ...

# Two CUDA devices, released together
with Reservation("distributed", DeviceConstraint("cuda"), DeviceConstraint("cuda")) as r:
    devices = r.torch_devices()
    run_distributed(devices)
```

### Multiple reservations — atomic acquire, independent release

`Request.acquire()` returns reservations that are already acquired. Release them
in `finally` blocks. The current `Reservation.__enter__()` attempts a fresh
acquisition, so do not put these returned objects in `with` statements.

```python
# Runs against the CPU config in the getting-started walkthrough.
first, second = Request(
    Reservation("first", DeviceConstraint("cpu", ram_mb=256)),
    Reservation("second", DeviceConstraint("cpu", ram_mb=256)),
).acquire()

try:
    first.check()
    second.check()
    first.release()
    second.check()  # second remains alive after first is released
finally:
    first.release()  # repeated release is safe
    second.release()
```

Releasing a reservation returns its accounting capacity. Unload any model using
that capacity before releasing it; Reserved does not free GPU memory itself.

### Explicit placement (orchestrator)

```python
# Pin directly to named devices — bypasses solver
with Reservation("infer", ExplicitSlot("cuda:2", ram_mb=8192)) as r:
    model.to(r.devices[0].torch_device())
    run_model()
```

### Checking for daemon restart

Call `check()` at natural checkpoints. Raises `ReservationExpiredError` if the daemon has restarted and the lease is gone.

```python
with Reservation("train", DeviceConstraint("cuda", ram_mb=16384)) as r:
    for epoch in range(num_epochs):
        r.check()
        train_epoch(model, data)
```

### Errors

- `ReservationError` — raised by `acquire()` when the request is rejected (insufficient resources).
- `DaemonNotRunningError` — raised when the daemon socket is not found.
- `ReservationExpiredError` — raised by `check()` when the daemon has restarted.

---

## Operator status: `reserve-status`

```bash
reserve-status [--socket PATH] [--json]
```

```
Devices:
  cuda:0  cuda  rtx3090  available: 16384/24576 MB  100/100 compute
  cuda:1  cuda  rtx3090  available: 24576/24576 MB  100/100 compute

Reservations:
  a3f9c2  label=train  age=2m14s
    cuda:0  8192 MB
```

Use `--json` for scripting:

```bash
reserve-status --json | jq '.reservations | length'
```

---

## Resource config

Devices have the following fields:

| Field | Description |
| --- | --- |
| `name` | Unique identifier. CUDA: must be `cuda:N` matching `nvidia-smi` index. Metal: use `metal:0`. |
| `type` | `cuda`, `metal`, or `cpu` |
| `model` | Optional label for the hardware class |
| `ram_mb` | Total reservable RAM (VRAM for CUDA, unified memory for Metal, system RAM for CPU) |
| `compute_units` | Optional opaque budget — convention between operator and callers |

**Apple Silicon (macOS) config:**

```json
{
  "devices": [
    { "name": "metal:0", "type": "metal", "model": "m3-max", "ram_mb": 36864, "compute_units": 100 },
    { "name": "cpu",     "type": "cpu",                       "ram_mb": 65536 }
  ]
}
```

Config is discovered in this order:
1. `--config PATH` flag
2. `RESERVED_CONFIG` env var
3. `~/.config/reserved/reserved.json`
4. `/etc/reserved/reserved.json`

An explicitly selected missing config is an error rather than a reason to use
a different file. See [socket discovery](INSTALL.md#socket-and-config-discovery)
for daemon and client socket defaults and overrides.

See [examples/](examples/) for small runnable configs and scripts, including an Apple Silicon Metal example.

---

## Common failure modes

- **Daemon socket not found** — start the daemon, or point clients at it with `RESERVED_SOCKET` or `--socket`.
- **Socket path mismatch** — make sure `reserved`, `reserve`, `reserve-status`, and SDK callers are using the same socket path.
- **Connection refused** — the socket file may be stale after a crash. Starting the daemon again probes and replaces stale sockets.
- **Reservation rejected** — resources are insufficient right now. `reserved` does not queue; callers decide whether to retry.
- **Daemon restarted** — all in-memory reservations are lost. SDK users should call `check()` at natural checkpoints.
- **Other processes still use the GPU** — only participating processes honor reservations. `reserved` does not constrain non-participants.

---

## Deployment

See [INSTALL.md](INSTALL.md) for complete, source-based installation recipes:

- [Linux system service](INSTALL.md#linux-system-service): dedicated service account, shared group, and systemd socket activation.
- [Linux user service](INSTALL.md#linux-user-service): per-user socket activation and optional operation after logout.
- [macOS user agent](INSTALL.md#macos-user-agent): generated launchd plist with absolute virtual environment paths.

The guide covers configuration, client setup, verification, logs, restarts,
upgrades, rollback, and removal. Ready-to-install systemd units and the launchd
plist renderer are in [deploy/](deploy/).

---

## What reserved does NOT do

- Queue requests — callers retry if they want to wait
- Persist state — restart means a clean slate
- Enforce against non-participating processes
- Multi-machine coordination
