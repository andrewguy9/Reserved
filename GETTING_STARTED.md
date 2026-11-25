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
return capacity until the orchestrator releases or relaunches. When the worker
runs from another virtual environment, follow
[Work with several virtual environments](#work-with-several-virtual-environments).

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
The usual layout is an orchestrator in one virtual environment that starts each
model worker from the worker's own environment. The lease handoff is a file
descriptor and a few environment variables, so it works across that boundary.

The rule behind every step here: **choose the worker's packages by choosing
its interpreter, never by adjusting its import path.** Python searches the
working directory first (for `python -m`), then `PYTHONPATH`, then the
interpreter's own environment, which includes editable installs. A path
override therefore loses silently to the working directory, and falls through
silently to the installed copy when it names a directory that does not contain
the package. The interpreter's environment is the one part of that search the
orchestrator can fix in advance.

### 1. Install the same version of Reserved in each environment

The orchestrator needs Reserved to acquire and launch. A Python worker that
adopts its lease needs Reserved to import `reserved.sdk`. Install the same
version in both, since the worker's SDK reads the variables the orchestrator's
SDK writes. The SDK uses only the standard library, so a worker environment
can install without dependencies:

```bash
/absolute/path/to/model-env/bin/python -m pip install --no-deps \
  "git+https://github.com/andrewguy9/Reserved.git"
```

### 2. Launch the worker with its own interpreter and environment

Run the worker environment's Python by absolute path, and give the worker an
environment that points at its own installation. `popen()` otherwise copies
the orchestrator's environment, including the orchestrator's own activation
state.

```python
import os
from reserved import sdk


def worker_env(venv: str) -> dict[str, str]:
    """Environment for a worker running from the virtual environment at venv."""
    env = {**os.environ, "VIRTUAL_ENV": venv, "PATH": f"{venv}/bin:{os.environ['PATH']}"}
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


venv = "/absolute/path/to/model-env"

with sdk.Reservation("model-worker", sdk.DeviceConstraint("cuda", ram_mb=8192)) as r:
    worker = r.popen([f"{venv}/bin/python", "-P", "-m", "model_worker"], env=worker_env(venv))
    worker.wait()
```

The interpreter path is the choice that matters. The two variables do what the
environment's `activate` script does, so programs the worker runs by name, such
as `python` or a tool installed in its environment, resolve into the worker's
environment. `PYTHONPATH` is removed because it belongs to the orchestrator:
it would put the orchestrator's trees ahead of the worker's installed packages.
`PYTHONHOME` is removed because it would repoint the worker's standard library.
Do not launch through a shell that sources `activate`; shell startup files can
change the environment in ways the orchestrator cannot see.

The `-P` flag keeps the worker's working directory off its import path. Without
it, `python -m` puts the current directory first, ahead of `PYTHONPATH` and of
every installed package, and a worker started in a source checkout imports
that checkout. In a printed `sys.path` that entry appears as `''`, which is
easy to read past. `PYTHONSAFEPATH=1` in the environment does the same as
`-P`. Passing `cwd=` to `popen()` to start the worker in a neutral directory
also works, and is the option when the worker needs its working directory on
the path.

Before relying on a launch, ask the interpreter what it will import, invoked
the way the worker is invoked: same interpreter, same flags, same working
directory, same environment:

```bash
/absolute/path/to/model-env/bin/python -P -c "import model_worker; print(model_worker.__file__)"
```

The printed path is the tree the worker will run. If it is not the one you
expect, fix the environment; no launch option will override it.

The `reserve` command gives the same result when the worker environment's
interpreter is named after `--`. The command passes its own environment through
unchanged, so clear `PYTHONPATH` first:

```bash
env -u PYTHONPATH reserve model-worker type=cuda,ram=8192 -- /absolute/path/to/model-env/bin/python -P -m model_worker
```

### 3. Adopt the lease in the worker as usual

Nothing changes inside the worker. It calls `adopt_inherited_lease()` at
startup and `check()` between units of work, exactly as under `reserve`.

### If a step is skipped

| Symptom | Cause | Fix |
| --- | --- | --- |
| Worker raises `ReservationError` naming `RESERVED_GRANTED_ALLOCATIONS` or `RESERVED_FD` at startup | The two environments have different Reserved versions | Reinstall both from the same commit (step 1) |
| `ModuleNotFoundError: reserved` in the worker | Reserved is not installed in the worker environment | Install it there (step 1) |
| A tool the worker runs by name comes from the orchestrator's environment | The worker inherited the orchestrator's `PATH` and `VIRTUAL_ENV` | Pass `env=worker_env(venv)` (step 2) |
| A worker started with `-m` or `-c` imports from the orchestrator's working directory, with no error, and `sys.path` shows `''` first | The working directory is searched before `PYTHONPATH` and before installed packages | Run the worker with `-P`, or pass `cwd=` to a neutral directory (step 2) |
| `PYTHONPATH` names a tree and the worker imports a different one, with no error | The named directory does not contain the package, so the search falls through to the installed copy | Do not select a tree with `PYTHONPATH`; run the interpreter of the environment that has the tree installed (step 2) |

### Workers that are not Python

A worker in another language needs no Reserved install. It holds the
descriptor named by `RESERVED_FD` and keeps the lease by staying alive; the
daemon reclaims the capacity when the last copy of the descriptor closes. The
descriptor becomes readable with end-of-file when the daemon restarts, which is
the same signal `check()` reports. The granted devices are in
`RESERVED_DEVICES` and `RESERVED_GRANTED_ALLOCATIONS`, and CUDA visibility is
already masked. See the
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
