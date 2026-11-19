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
`/tmp/reserved-<uid>.sock`. Clients try the runtime socket first, then the platform
default if it is missing or refuses connections. Explicit `--socket` and
`RESERVED_SOCKET` settings disable fallback. On Linux
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
| `name=NAME` | Pin to a physical device by its configured name |

```bash
# Any CUDA device
reserve infer type=cuda -- python infer.py

# Apple Silicon (Metal/MPS)
reserve infer type=metal -- python infer.py

# Needs 30GB VRAM
reserve train type=cuda,ram=30720 -- python train.py

# Specific model
reserve infer type=cuda,model=rtx3090 -- python infer.py

# Pin to a physical device
reserve infer name=cuda:2 -- python infer.py

# Two distinct CUDA devices (distributed job)
reserve distributed name=cuda:0,ram=1024 name=cuda:1,ram=1024 -- python distributed.py

# Prefer spreading across cards when capacity allows
reserve pipeline type=cuda type=cuda --solver spread -- python pipeline.py
```

See the [runnable NVIDIA CUDA examples](examples/README.md#nvidia-cuda) for
configs and a PyTorch worker for one or two cards.

**Environment variables set before exec:**

| Variable | Value |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | Distinct physical CUDA ordinals in first-grant order, defining the child's CUDA view |
| `RESERVED_DEVICE` | Physical name of the first granted accelerator, or the first device for a CPU-only grant |
| `RESERVED_DEVICES` | Comma-separated physical device names in grant order |
| `RESERVED_GRANTED_SLOTS` | JSON array of physical grant metadata for SDK adoption |
| `RESERVED_ID` | Reservation ID |
| `RESERVED_FD` | File descriptor number of the lease fd |

For `reserve infer name=cuda:2 -- python infer.py`, the daemon grants physical
`cuda:2`. The worker sees `RESERVED_DEVICE=cuda:2`, `RESERVED_DEVICES=cuda:2`,
and `CUDA_VISIBLE_DEVICES=2`. PyTorch uses logical `cuda:0` in that process.

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

# Two distinct CUDA devices, released together
with Reservation("distributed", ExplicitSlot("cuda:0", ram_mb=1024),
                 ExplicitSlot("cuda:1", ram_mb=1024)) as r:
    devices = r.torch_devices()
    run_distributed(devices)
```

### Multiple reservations — atomic acquire, independent release

`Request.acquire()` returns reservations that are already acquired. Release them
in `finally` blocks. Entering a context with one of those objects attempts a
fresh acquisition.

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
# Pin directly to a physical device
with Reservation("infer", ExplicitSlot("cuda:2", ram_mb=8192)) as r:
    model.to(r.devices[0].torch_device())
    run_model()
```

### Solve placement separately from acquisition

SDK `ReservationSpec` records describe requirements; concrete `ReserveRequest`
records contain named device assignments. Given `capacity`, the `devices` list
from a daemon status response, solving is pure:

```python
from reserved.sdk import (
    DeviceConstraint, NoPlacement, ReservationSpec, acquire_reservations,
    solve_reservations_spread,
)

specifications = [
    ReservationSpec("worker-0", (DeviceConstraint("cpu", ram_mb=256),)),
    ReservationSpec("worker-1", (DeviceConstraint("cpu", ram_mb=256),)),
]
placement = solve_reservations_spread(capacity, specifications)
if isinstance(placement, NoPlacement):
    print(placement.reason)  # The application decides whether to retry.
else:
    reservations = acquire_reservations(placement)
    try:
        for reservation in reservations:
            reservation.check()
            # Start and supervise the worker while retaining this handle.
    finally:
        for reservation in reservations:
            reservation.release()
```

Each policy attempts every specification and returns one complete candidate or
`NoPlacement`; failure does not return a partial batch. Labels are carried as
metadata, and inputs remain unchanged. These are greedy policies, not exhaustive
searches. Their explicit assignments are passed through for daemon validation.

Acquisition does no solving or status query. The daemon can reject a stale
placement with `ReservationError`. Returned handles are already acquired and
release independently. For the combined workflow, use
`Request(..., solver=solve_reservations_spread).acquire()`.

Custom solvers are ordinary functions with signature
`(capacity, reservation_specs) -> ReserveRequest | NoPlacement`. This replaces
the previous `ConsolidateSolver`/`SpreadSolver` objects and their `.solve()`
method. Choose the solver on `Request` for a batch; a member's explicitly chosen
standalone solver must agree with that choice.

### Adopt a lease from `reserve`

A Python worker launched by `reserve` can manage the lease it already holds.
Adopt it at startup and keep the context open while the worker uses the granted
capacity.

```python
from reserved import sdk

with sdk.adopt_inherited_lease() as reservation:
    reservation.check()
    (device,) = reservation.torch_devices()
    run_worker(device)
```

Adoption takes ownership of `RESERVED_FD` once and removes it from the
environment. Context exit or explicit `release()` closes the worker's copy of
the lease. After adoption, the descriptor closes on a later `exec` so another
executable cannot accidentally keep the grant alive. A forked child still
gets a copy, so capacity returns when the last copy closes.

The adopted reservation exposes `reservation_id` from `RESERVED_ID` and
physical grants from `RESERVED_GRANTED_SLOTS`. Update the `reserve` launcher
and the worker SDK together. Adoption requires this grant metadata and
accepts an empty array for a reservation with zero slots.
See the [getting-started walkthrough](GETTING_STARTED.md#adopt-the-lease-in-a-python-worker).

### PyTorch device selection

For either acquisition path, `reservation.devices` holds physical grant
metadata for placement and diagnostics. Pass the result of
`reservation.torch_devices()` to PyTorch. It returns one process-local device
per granted slot, in grant order. For example, with `CUDA_VISIBLE_DEVICES=5,2`,
physical `cuda:2` becomes logical `cuda:1`. When the mask is unset, the
helper uses the physical ordinal as the PyTorch index.

Set CUDA visibility before CUDA initializes. The SDK accepts masks made of
distinct nonnegative decimal ordinals. An empty mask or `-1` hides all CUDA
devices. Hidden grants, malformed masks, UUID masks, and MIG masks raise
`ReservationError`.

For PyTorch device selection, use `cpu` or a `cpu:` name for CPU grants and
`metal:0` for Metal grants. Unsupported device names raise `ReservationError`.

### Checking for daemon restart

Call `check()` at natural checkpoints. Raises `ReservationExpiredError` if the daemon has restarted and the lease is gone.

```python
with Reservation("train", DeviceConstraint("cuda", ram_mb=16384)) as r:
    for epoch in range(num_epochs):
        r.check()
        train_epoch(model, data)
```

### Errors

- `ReservationError` — raised for rejected requests, invalid inherited handoffs, or PyTorch device mapping errors.
- `DaemonNotRunningError` — raised when the daemon socket is not found.
- `ReservationExpiredError` — raised by `check()` when the daemon has restarted.
- `InheritedLeaseAlreadyAdoptedError` — raised if the handoff has already been claimed, even after release.

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
| `name` | Unique physical identifier. CUDA uses `cuda:N` with the physical ordinal. Metal uses `metal:0`. |
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
