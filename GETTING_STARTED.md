# Getting started with Reserved

Use Reserved to coordinate hardware capacity between cooperating processes on
one machine. This walkthrough installs Reserved from GitHub, runs a reservation
without a GPU, and explains how to integrate an application or model worker.

Reserved grants or rejects immediately. Your application owns its work queue,
batching, retries, model loading, and worker processes.

## Install from GitHub

Requires Python 3.11 or later and Git on macOS or Linux. The transport uses Unix
sockets and file descriptor passing. Pip installs `docopt-ng` for the command-line
tools. Install PyTorch separately when using `torch_device()` or `torch_devices()`.

Activate your application's existing Python environment. For a standalone demo,
you can create one instead:

```bash
python3 -m venv "$HOME/.venvs/reserved"
. "$HOME/.venvs/reserved/bin/activate"
```

Install directly from GitHub:

```bash
python -m pip install "git+https://github.com/andrewguy9/Reserved.git"
python -c 'from reserved.sdk import Reservation; print("Reserved SDK available")'
```

This installs the SDK and the `reserved`, `reserve`, and `reserve-status`
commands. Pip fetches and builds the project from GitHub; you do not need to
manage a source checkout to use them. Install Reserved into each environment
that imports its SDK. The daemon can run in a separate environment; clients
coordinate through the same socket.

For development against a local checkout, use an editable install instead:

```bash
python -m pip install -e /absolute/path/to/Reserved
```

Replace the path with your checkout. An editable install follows changes in that
checkout.

## Try a reservation without a GPU

The walkthrough reads example files from a source checkout. Keep the installed
environment active and get the examples below. If you already have a checkout,
change to its directory and skip cloning:

```bash
git clone https://github.com/andrewguy9/Reserved.git
cd Reserved
```

The [CPU config](examples/cpu.json) declares a 1024 MB accounting budget. It
neither allocates memory nor discovers the machine's actual capacity.

### Terminal one starts the daemon

```bash
export RESERVED_SOCKET=/tmp/reserved-getting-started.sock
reserved --config examples/cpu.json --socket "$RESERVED_SOCKET"
```

Leave the daemon running in the foreground. Use this demo socket separately from
your normal daemon. If another demo is already using it, choose a different path
and use that path in both terminals.

### Terminal two runs the client

Open another terminal at the same checkout and activate the same environment.
For the standalone environment above:

```bash
. "$HOME/.venvs/reserved/bin/activate"
export RESERVED_SOCKET=/tmp/reserved-getting-started.sock
reserve-status
python examples/cpu_job.py
reserve-status
```

Each terminal needs its own environment activation and socket setting. The first
status should show `1024/1024 MB` available and no reservations. The example uses
the SDK to claim 256 MB, checks its lease, and runs the status CLI while holding
the claim. That output should show `768/1024 MB` and a `cpu-example` reservation.
After it exits, the daemon observes the lease closing and returns availability
to `1024/1024 MB`. If the final status races with cleanup, query again until the
reservation disappears.

### Wrap a command that knows nothing about Reserved

Run this against the same demo daemon:

```bash
reserve cli-example type=cpu,ram=256 -- sleep 5
reserve-status --json
```

`sleep` holds the lease simply by keeping the inherited descriptor open. Run
`reserve-status` from another terminal with the same environment and socket
while it sleeps to see `cli-example` and 768 MB available. After `sleep` exits,
availability returns to 1024 MB and the `reservations` array becomes empty.
When a command hands work off to another process, that process must keep a
lease copy open if it will continue using the reserved capacity after the
original command exits.

### Adopt the lease in a Python worker

Launch the [example worker](examples/inherited_lease_job.py) against the same
daemon:

```bash
reserve inherited-example type=cpu,ram=256 -- python examples/inherited_lease_job.py
```

The worker adopts the lease at startup and prints its reservation ID at each
batch checkpoint. It releases its copy when it leaves the `with` block. A
missing or invalid inherited descriptor raises `ReservationError` before the
worker reaches its first batch.

A Python process can perform the same handoff itself. The
[launch example](examples/launch_worker.py) acquires a reservation with the SDK
and starts the same worker under it with `Reservation.popen()`:

```bash
python examples/launch_worker.py
```

The parent prints when it starts and when the worker exits; the worker prints
the same reservation ID at each checkpoint.

Stop the demo daemon with Ctrl-C in terminal one when finished. All commands in
the rest of this guide assume a daemon configured for your real hardware; see
the [resource config reference](README.md#resource-config).

## Choose the integration path

| Use case | Starting point | Lifetime owner |
| --- | --- | --- |
| Python worker that can check its lease between batches | `Reservation` context manager | The worker |
| Existing command that needs a reservation and CUDA masking | `reserve LABEL DEVICE -- CMD` | The command and any inherited lease copies |
| Python worker launched by `reserve` that checks or releases its lease | `adopt_inherited_lease()` | The worker and any inherited lease copies |
| Python orchestrator that starts workers in other environments | `Reservation.popen(...)` | The orchestrator and each worker's lease copy |
| Several reservations that must be granted together | `Request(...).acquire()` | The caller, with explicit `release()` cleanup |
| Coordinator inspecting capacity | `reserve-status --json` | Status is a snapshot and holds no capacity |

The [CPU example](examples/cpu_job.py) is a complete SDK entry point. The
[README SDK reference](README.md#python-sdk-reservedsdk) covers constraints,
explicit placement, and acquiring several reservations together. Use public
imports from `reserved.sdk`; transport helpers and lease socket attributes are
private.

After `Request.acquire()`, its returned reservations are already acquired. Use
`try`/`finally` and `release()` as in the README. Entering a context with one of
those objects attempts a fresh acquisition.

## Hold capacity for a model worker

Acquire or adopt the lease inside the worker before loading the model. Keep
that reservation for as long as the model occupies the device, including idle
time between batches.
Budget for the model plus peak batch memory and other allocations. Supply
nonnegative amounts: omitted RAM and compute amounts default to zero, which
claims none of that capacity. Device selection alone does not reserve its RAM.

Call `reservation.check()` before each batch or at other natural checkpoints.
It raises `ReservationExpiredError` when the daemon's lease endpoint closes.
On lease loss, stop submitting inference, stop or unload the worker's model,
and let application recovery decide what to retry. The daemon starts with an
empty ledger after restart and does not restore previous reservations.

Before releasing a live worker's reservation, unload the resources it covers.
`release()` only returns accounting capacity; it does not unload a model or
free GPU memory. If a worker exits or crashes, the daemon reclaims capacity
after the last client lease copy closes. Forked children or deliberately
inherited copies can extend that lifetime.

For a command in another virtual environment, use that environment's Python
executable after `--`. For example, if your application provides `model_worker`:

```bash
reserve model-worker type=cuda,ram=8192 -- /absolute/path/to/model-env/bin/python -m model_worker
```

`model_worker` belongs to the application; Reserved does not provide it. The
8192 MB budget is illustrative and must be replaced with your measured needs.
The CLI preserves the lease across exec. Inside `model_worker`, adopt the
inherited lease and call `check()` at checkpoints to detect daemon shutdown.

An orchestrating Python process can perform the same handoff without the CLI:

```python
from reserved import sdk

with sdk.Reservation("model-worker", sdk.DeviceConstraint("cuda", ram_mb=8192)) as r:
    worker = r.popen(["/absolute/path/to/model-env/bin/python", "-m", "model_worker"])
    worker.wait()
```

The orchestrator keeps its own lease copy, so a worker that crashes does not
return capacity until the orchestrator releases or relaunches. Install the same
Reserved version in the orchestrator's and each worker's environment, since the
worker's SDK reads the variables the orchestrator's SDK writes. See
[Work with several virtual environments](#work-with-several-virtual-environments)
for installation and activation details.

### CUDA device indices

The CLI sets `CUDA_VISIBLE_DEVICES` to the granted GPU UUIDs. A worker
launched with `reserve model-worker name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782 -- python -m model_worker`
sees `RESERVED_DEVICE` and `RESERVED_DEVICES` set to that name and
`CUDA_VISIBLE_DEVICES=GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782`.

```python
from reserved import sdk

with sdk.adopt_inherited_lease() as reservation:
    (device,) = reservation.torch_devices()  # torch.device("cuda:0")
    run_worker(device)
```

Use `torch_devices()` the same way with an SDK-acquired reservation. The helper
asks torch which visible device carries the granted UUID and returns PyTorch
devices in grant order, so it does not depend on how the process's CUDA
visibility was set. `reservation.devices` holds the physical grant names for
diagnostics.

| Physical grant in `reservation.devices` | CUDA devices torch can see | Logical PyTorch device |
| --- | --- | --- |
| `cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782` | All cards, unmasked; this is the third | `cuda:2` |
| `cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782` | `GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782` only | `cuda:0` |
| `cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782` | `GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781`, then `GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782` | `cuda:1` |

Resolving a CUDA grant initializes CUDA, so call `torch_devices()` in the
process that will use the GPU.
See [CUDA device indexing](ARCHITECTURE.md#cuda-device-indexing).

## Work with several virtual environments

Model families often need dependency sets that cannot share one environment.
A common layout is an orchestrator in one virtual environment that starts each
model worker from the worker's own environment. The handoff is a file
descriptor and a few environment variables, so it does not depend on which
interpreter runs the worker, or whether the worker is Python at all. This
section covers the three things that do matter: installing the SDK in each
environment that uses it, choosing the worker's interpreter, and the
activation state the worker inherits.

### Install Reserved in every environment that uses the SDK

A Python worker that adopts its lease with `adopt_inherited_lease()` imports
`reserved.sdk` from its own environment, so Reserved must be installed there
as well as in the orchestrator's. The SDK uses only the standard library;
`docopt-ng` is needed only by the command-line tools. A worker that only
adopts a lease can install without dependencies:

```bash
/absolute/path/to/model-env/bin/python -m pip install --no-deps \
  git+https://github.com/andrewguy9/Reserved.git
```

Install the same version everywhere. Adoption reads the variables the
launching SDK writes, so a worker on a different version fails at startup with
`ReservationError` naming `RESERVED_GRANTED_ALLOCATIONS` or `RESERVED_FD`.
Reinstall the orchestrator and every worker environment from the same commit.
Before version 1.0, the handoff variables and the daemon's socket messages may
change between releases without compatibility with earlier versions.

Do not start workers with a Reserved source checkout as their working
directory. `python -m` puts the working directory first on the import path,
so the worker would import the checkout instead of its own installed copy.
Pass `cwd=` to `popen()` when the orchestrator itself runs from the checkout.

### Start the worker with its own interpreter

The interpreter decides which site-packages the worker sees. Give `popen()`
the absolute path to the worker environment's Python:

```python
from reserved import sdk

venv = "/absolute/path/to/model-env"

with sdk.Reservation("model-worker", sdk.DeviceConstraint("cuda", ram_mb=8192)) as r:
    worker = r.popen([f"{venv}/bin/python", "-m", "model_worker"])
    worker.wait()
```

Inside the worker, `sys.prefix` is the worker environment and
`adopt_inherited_lease()` works exactly as it does under `reserve`.

### Pass activation state, not just the interpreter

`popen()` copies the orchestrator's environment to the worker. If the
orchestrator runs inside an activated virtual environment, the worker inherits
`VIRTUAL_ENV` and a `PATH` whose first entry is the orchestrator's `bin`. The
worker's interpreter and imports are still correct, but anything the worker
runs by bare name, such as `python`, a console script, or a helper tool
installed in its environment, resolves into the orchestrator's environment
instead.

Activation is two variables, so the orchestrator can set them itself and skip
the shell:

```python
import os
from reserved import sdk


def activated(venv: str) -> dict[str, str]:
    env = {**os.environ, "VIRTUAL_ENV": venv, "PATH": f"{venv}/bin:{os.environ['PATH']}"}
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


venv = "/absolute/path/to/model-env"

with sdk.Reservation("model-worker", sdk.DeviceConstraint("cuda", ram_mb=8192)) as r:
    worker = r.popen(["python", "-m", "model_worker"], env=activated(venv))
    worker.wait()
```

With `PATH` set this way the bare `python` is the worker environment's
interpreter. Dropping `PYTHONPATH` and `PYTHONHOME` keeps the orchestrator's
import configuration out of the worker.

A shell that sources the environment's `activate` script gives the same
result. The lease descriptor passes through the shell to the worker:

```python
worker = r.popen(["bash", "-c", f"source '{venv}/bin/activate' && exec python -m model_worker"])
```

Use `exec` so the shell is replaced by the worker and signals and exit codes
reach the worker directly. Without `exec` the shell stays as the worker's
parent; the handoff still works, because shells do not close inherited
descriptors, but the orchestrator then supervises the shell rather than the
worker.

### Workers that are not Python

A worker in another language needs no Reserved install. It holds the
descriptor named by `RESERVED_FD` and keeps the lease by staying alive; the
daemon reclaims the capacity when the last copy of the descriptor closes. The
descriptor becomes readable with end-of-file when the daemon restarts, so the
worker can poll it for the same signal `check()` provides. The granted devices
are in `RESERVED_DEVICES` and `RESERVED_GRANTED_ALLOCATIONS`, and CUDA
visibility is already masked. See the
[environment variable table](README.md#command-line-wrapper-reserve).

## Handle failures

| Observation | Application action |
| --- | --- |
| `DaemonNotRunningError`, or a CLI daemon-connection error with exit code 2 | Start or locate the daemon and check the socket setting. |
| `ReservationError` during acquisition, or `reserve` exit code 1 for a rejected request | Inspect the reason and current status. Fix invalid requests; retry capacity shortages according to application policy. |
| `ReservationExpiredError` from `check()` | Stop using the invalid lease and recover the work through the application. |
| `ReservationError` from `torch_devices()` for a hidden CUDA grant or invalid mask | Check the physical grant name and the process's CUDA visibility. |
| Capacity remains claimed after a worker exits | Check whether a child or another process still holds a lease copy. |
| `ReservationError` naming `RESERVED_GRANTED_ALLOCATIONS` or `RESERVED_FD` when a worker adopts | The launching and worker environments run different Reserved versions; reinstall both from the same commit. |

`DaemonNotRunningError` and `ReservationExpiredError` inherit from
`ReservationError`; catch them first if handling the cases separately. Transport
and protocol failures can also raise standard Python exceptions. The current
SDK has no configurable communication timeout, so applications needing bounded
supervision must provide it at the worker/process level.

Use one explicit socket path throughout setup. SDK and CLI callers use their
socket argument, then `RESERVED_SOCKET`, then `$XDG_RUNTIME_DIR/reserved.sock`
when set, then `/tmp/reserved-<uid>.sock` on macOS or `/var/run/reserved.sock`
on Linux. The standalone daemon uses the same discovery rules. A systemd
service adopts its supplied listener. For persistent services and client
configuration, follow the [installation guide](INSTALL.md).

## Continue from here

- [README](README.md): resource configuration, API examples, status, and deployment.
- [Installation guide](INSTALL.md): Linux systemd and macOS launchd setup, verification, and service management.
- [Architecture](ARCHITECTURE.md): current protocol, device indices, and lease mechanics.
- [Examples](examples/README.md): CPU walkthrough and Apple Silicon configuration.
- [Design](RESERVED_DESIGN.md): the design and its rationale, kept current with the code.
- [Contributor guide](AGENTS.md): implementation and verification rules when changing Reserved.
