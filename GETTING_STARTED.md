# Getting started with Reserved

Use Reserved to coordinate hardware capacity between cooperating processes on
one machine. This walkthrough installs Reserved from GitHub, runs a reservation
without a GPU, and explains how to integrate an application or model worker.

Reserved grants or rejects immediately. Your application owns its work queue,
batching, retries, model loading, and worker processes.

## Install from GitHub

Requires Python 3.11 or later and Git on macOS or Linux. The transport uses Unix
sockets and file descriptor passing. The package has no runtime dependencies;
PyTorch is imported only if you call `torch_device()` or `torch_devices()`.

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

Try the CLI wrapper against the same daemon:

```bash
reserve cli-example type=cpu,ram=256 -- python -m reserved.bin.reserve_status
reserve-status --json
```

The wrapped status command should see its own `cli-example` reservation and
768 MB available. After it exits, availability returns to 1024 MB and the
`reservations` array becomes empty. Reserved sets the lease up before replacing
the wrapper with the command, so the command needs no SDK integration.

Stop the demo daemon with Ctrl-C in terminal one when finished. All commands in
the rest of this guide assume a daemon configured for your real hardware; see
the [resource config reference](README.md#resource-config).

## Choose the integration path

| Use case | Starting point | Lifetime owner |
| --- | --- | --- |
| Python worker that can check its lease between batches | `Reservation` context manager | The worker |
| Existing command that needs a reservation and CUDA masking | `reserve LABEL SLOT -- CMD` | The command and any inherited lease copies |
| Several reservations that must be granted together | `Request(...).acquire()` | The caller, with explicit `release()` cleanup |
| Coordinator inspecting capacity | `reserve-status --json` | Status is a snapshot and holds no capacity |

The [CPU example](examples/cpu_job.py) is a complete SDK entry point. The
[README SDK reference](README.md#python-sdk-reservedsdk) covers constraints,
explicit placement, and acquiring several reservations together. Use public
imports from `reserved.sdk`; transport helpers and lease socket attributes are
private. There is currently no public SDK API for transferring or adopting a
reservation from another process. A new SDK worker can acquire its own lease.

After `Request.acquire()`, its returned reservations are already acquired. Use
`try`/`finally` and `release()` as in the README. The current implementation of
`with reservation:` attempts a fresh acquisition, including for an already
acquired reservation.

## Hold capacity for a model worker

Acquire inside the worker before loading the model. Keep that reservation for
as long as the model occupies the device, including idle time between batches.
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
The CLI preserves the lease across exec, but does not monitor or stop the
command when the daemon dies. Use worker-side SDK acquisition and `check()`
when you need the public Python lease-loss interface.

### CUDA device indices

The CLI sets `CUDA_VISIBLE_DEVICES` to granted physical ordinals. A worker
granted physical `cuda:2` sees `CUDA_VISIBLE_DEVICES=2` and uses logical
`cuda:0` inside PyTorch. `RESERVED_DEVICE` still names the physical device.

The SDK leaves `CUDA_VISIBLE_DEVICES` unchanged. Its `torch_device()` helper
returns the physical ordinal. Use that directly only when the worker has an
unmasked CUDA view; an existing mask needs an explicit physical-to-logical
mapping in the application. See [CUDA device indexing](ARCHITECTURE.md#cuda-device-indexing).

## Handle failures

| Observation | Application action |
| --- | --- |
| `DaemonNotRunningError`, or a CLI daemon-connection error with exit code 2 | Start or locate the daemon and check the socket setting. |
| `ReservationError` during acquisition, or `reserve` exit code 1 for a rejected request | Inspect the reason and current status. Fix invalid requests; retry capacity shortages according to application policy. |
| `ReservationExpiredError` from `check()` | Stop using the invalid lease and recover the work through the application. |
| Capacity remains claimed after a worker exits | Check whether a child or another process still holds a lease copy. |

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
- [Contributor guide](AGENTS.md): implementation and verification rules when changing Reserved.

`RESERVED_DESIGN.md` and `DAEMON_REFACTOR_PLAN.md` are historical context. Use
the current implementation and the references above when integrating a client.
