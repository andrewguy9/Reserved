# reserved

**Stop playing musical chairs with your GPUs. `reserved` prevents over-allocation crashes by letting jobs reserve hardware before they run.**

A single-machine, best-effort resource reservation daemon for physical hardware (GPUs, CPU, RAM). Processes claim hardware capacity before running, and release it automatically when they exit — even on crash.

`reserved` is opt-in coordination between willing processes. It does not enforce anything against programs that don't participate. It succeeds or fails immediately — there is no queue.

`reserved` is experimental operator tooling. Treat it as cooperative coordination, not a security boundary, quota enforcer, scheduler, or replacement for OS/container isolation.

**New clients and agents: start with [Getting started](GETTING_STARTED.md)** for
installation from GitHub, a walkthrough requiring no GPU, and guidance
on integrating an application or model worker.

For installation and service management, see [INSTALL.md](INSTALL.md). For the implementation reference, see [ARCHITECTURE.md](ARCHITECTURE.md). For the design and its rationale, see [RESERVED_DESIGN.md](RESERVED_DESIGN.md).

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
    { "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780", "type": "cuda", "ram_mb": 24576, "compute_units": 100 },
    { "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781", "type": "cuda", "ram_mb": 24576, "compute_units": 100 },
    { "name": "cpu", "type": "cpu", "ram_mb": 65536 }
  ]
}
```

CUDA devices are named by GPU UUID, exactly as `nvidia-smi -L` prints it, so a
name identifies one specific card regardless of driver enumeration order.
`reserved-generate-config` writes this config for the CUDA cards torch can see,
with each card's memory as `ram_mb` and its multiprocessor count as `compute_units`.

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
reserve LABEL [DEVICE ...] [--solver consolidate|spread] [--socket PATH] -- CMD [ARGS...]
```

Each `DEVICE` is one allocation in the reservation, written as comma-separated fields:

| Field | Effect |
| --- | --- |
| `type=TYPE` | Device type: `cuda`, `metal`, or `cpu` |
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

# Pin to a physical device
reserve infer name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782 -- python infer.py

# Two distinct CUDA devices (distributed job)
reserve distributed name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780,ram=1024 name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781,ram=1024 -- python distributed.py

# Prefer spreading across cards when capacity allows
reserve pipeline type=cuda type=cuda --solver spread -- python pipeline.py
```

See the [runnable NVIDIA CUDA examples](examples/README.md#nvidia-cuda) for
configs and a PyTorch worker for one or two cards.

**Environment variables set before exec:**

| Variable | Value |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | Distinct GPU UUIDs of the granted CUDA devices in first-grant order, defining the child's CUDA view |
| `RESERVED_DEVICE` | Physical name of the first granted accelerator, or the first device for a CPU-only grant |
| `RESERVED_DEVICES` | Comma-separated physical device names in grant order |
| `RESERVED_GRANTED_ALLOCATIONS` | JSON array of granted allocations (`name`, `ram_mb`, `compute_units`) for SDK adoption |
| `RESERVED_ID` | Reservation ID |
| `RESERVED_FD` | File descriptor number of the lease fd |

`Reservation.popen()` in the SDK sets the same variables for a worker it
starts; see [Launch a worker under a reservation](#launch-a-worker-under-a-reservation).

For `reserve infer name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782 -- python infer.py`, the daemon
grants that card. The worker sees `RESERVED_DEVICE` and `RESERVED_DEVICES` set
to `cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782` and `CUDA_VISIBLE_DEVICES=GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782`.
PyTorch uses logical `cuda:0` in that process.

---

## Python SDK: `reserved.sdk`

Use the SDK when an application can participate in reservations directly. The SDK talks to the daemon, holds the lease fd for the lifetime of each `Reservation`, and exposes granted devices to application code.

```python
from reserved.sdk import Reservation, Request, DeviceConstraint
from reserved.sdk.errors import ReservationError, DaemonNotRunningError, ReservationExpiredError
```

### Single reservation

```python
# Any CUDA device
with Reservation("infer", DeviceConstraint("cuda")) as r:
    model.to(r.torch_devices()[0])
    run_model()

# Metal (Apple Silicon)
with Reservation("infer", DeviceConstraint("metal")) as r:
    model.to(r.torch_devices()[0])   # torch.device("mps")
    run_model()

# Needs 30GB VRAM
with Reservation("infer", DeviceConstraint("cuda", ram_mb=30720)) as r:
    model.to(r.torch_devices()[0])
    ...

# Two distinct CUDA devices, released together
with Reservation("distributed", DeviceConstraint(name="cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780", ram_mb=1024),
                 DeviceConstraint(name="cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781", ram_mb=1024)) as r:
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

A constraint with `name=` pins one physical device. The solver only checks
that the capacity fits, and the daemon checks again when it grants.

```python
# Pin directly to a physical device
with Reservation("infer", DeviceConstraint(name="cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782", ram_mb=8192)) as r:
    model.to(r.torch_devices()[0])
    run_model()
```

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
physical grants from `RESERVED_GRANTED_ALLOCATIONS`. Update the `reserve` launcher
and the worker SDK together. Adoption requires this grant metadata and
accepts an empty array for a reservation with zero allocations.
See the [getting-started walkthrough](GETTING_STARTED.md#adopt-the-lease-in-a-python-worker).

### Launch a worker under a reservation

An orchestrating process can acquire capacity with the SDK and start a worker
that holds a copy of the lease, the same handoff `reserve` performs. `popen()`
wraps `subprocess.Popen`: it passes the lease descriptor to the child and sets
the [launcher's environment variables](#command-line-wrapper-reserve) so the
worker can call `adopt_inherited_lease()`. Other descriptors stay closed unless
listed in `pass_fds`. Remaining keyword arguments go to `subprocess.Popen`.

```python
from reserved.sdk import Reservation, DeviceConstraint

with Reservation("tts", DeviceConstraint("cuda", ram_mb=8192)) as r:
    worker = r.popen(["/path/to/tts-env/bin/python", "-m", "tts_worker"])
    worker.wait()
```

The parent keeps its own copy of the lease. The daemon reclaims capacity after
the last copy closes, so the reservation above ends once the worker has exited
and the `with` block closes the parent's copy. Keep the parent's copy open to
relaunch a crashed worker under the same grant, or release it to let the
worker's lifetime end the reservation. A worker that exits does not return
capacity while the parent still holds the lease.

For a Python worker in another environment, `popen_python()` takes the
environment directory and the module to run. It starts that environment's
`bin/python` with `-P -m module`, puts the environment's `bin` first on `PATH`,
sets `VIRTUAL_ENV`, and drops `PYTHONPATH` and `PYTHONHOME`, so the worker
imports only from its own environment. Further positional arguments are the
module's arguments; keyword arguments go to `subprocess.Popen`.

```python
with Reservation("tts", DeviceConstraint("cuda", ram_mb=8192)) as r:
    worker = r.popen_python("/path/to/tts-env", "tts_worker", "--port", "9000")
    worker.wait()
```

`child_env(base_env)` returns the variables without starting a process, and
`lease_fd` is the descriptor to pass, for other launchers such as
`asyncio.create_subprocess_exec(..., pass_fds=[r.lease_fd], env=r.child_env())`.
All of these raise `ReservationExpiredError` once the reservation is released.
For a worker in another virtual environment, including the version the worker
must install, see the getting-started guide's
[Work with several virtual environments](GETTING_STARTED.md#work-with-several-virtual-environments).

### PyTorch device selection

For either acquisition path, `reservation.devices` holds the granted physical
device names for placement and diagnostics. Pass the result of
`reservation.torch_devices()` to PyTorch. It returns one process-local device
per granted allocation, in grant order. A CUDA grant resolves to the device
torch reports with the same GPU UUID, so the mapping is correct whether
`CUDA_VISIBLE_DEVICES` is unset, set by the `reserve` launcher, or set by a
scheduler. For example, with `CUDA_VISIBLE_DEVICES=GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781,GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782`,
a grant of `cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782` becomes logical `cuda:1`.

`torch_devices()` expects PyTorch to be installed in the calling process; the
package does not declare or install it. Acquiring, checking, and releasing
reservations never import torch. Resolving a CUDA grant initializes CUDA in the
calling process, so call `torch_devices()` in the process that will use the
GPU. A grant that torch cannot see, and a torch build that does not report
device UUIDs, raise `ReservationError`.

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
  cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780  cuda  available: 16384/24576 MB  100/100 compute
  cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781  cuda  available: 24576/24576 MB  100/100 compute

Reservations:
  a3f9c2  label=train  age=2m14s
    cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780  8192 MB
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
| `name` | Unique physical identifier. CUDA uses `cuda:GPU-<uuid>` with the lowercase UUID printed by `nvidia-smi -L`. Metal uses `metal:0`. |
| `type` | `cuda`, `metal`, or `cpu` |
| `ram_mb` | Total reservable RAM (VRAM for CUDA, unified memory for Metal, system RAM for CPU) |
| `compute_units` | Optional opaque budget — convention between operator and callers |

**Apple Silicon (macOS) config:**

```json
{
  "devices": [
    { "name": "metal:0", "type": "metal", "ram_mb": 36864, "compute_units": 100 },
    { "name": "cpu", "type": "cpu", "ram_mb": 65536 }
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
