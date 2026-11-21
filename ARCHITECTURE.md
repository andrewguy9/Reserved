# reserved — Architecture

## Design principles

`reserved` is a single-machine, best-effort resource reservation system. It succeeds or fails immediately — there is no queue.

1. **Reservation system** — processes claim a slice of physical hardware capacity (device type, VRAM, compute budget). The daemon grants or rejects immediately.
2. **Handle-based lifetime** — each reservation is backed by a lease fd (one end of a socketpair created by the daemon). Multiple reservations can be acquired atomically in one request but released independently. When a process dies for any reason, its fds close and all held reservations are freed automatically.
3. **Optional SDK** — a Python SDK provides a `Reservation` context manager and a `check()` method for daemon restart detection.
4. **Daemon** — a lightweight process holding the resource ledger in memory. State is lost on restart.
5. **Optional status API** — HTTP endpoint and `reserve-status` CLI for operator visibility.
6. **Convenience wrapper** — `reserve` CLI wraps non-reservation-aware programs.

---

## Two languages: explicit and constraint

`reserved` uses two distinct languages at different layers of the stack.

### Explicit language (daemon protocol)

The daemon speaks only in explicit terms. Every allocation names a physical device and states exactly how much capacity to claim:

```
allocation: name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780  ram_mb=8192  compute_units=50
```

No wildcards. No type matching. The daemon checks whether the named device has sufficient capacity and subtracts if so.

### Constraint language (SDK and CLI)

Users typically don't know or care which physical device they get — they just need a CUDA device with enough VRAM:

```
constraint: type=cuda  ram_mb=8192
constraint: name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782  ram_mb=8192
```

Every field narrows the choice: `type` matches the device config, `name` pins one physical device, and `ram_mb` and `compute_units` must fit its available capacity. A **constraint solver** resolves constraints against current device status and produces an explicit request before submitting to the daemon. The SDK and CLI accept only constraints; a pinned device is a constraint with a `name`.

### Constraint solvers

| Solver | Behaviour | Default |
| --- | --- | --- |
| `ConsolidateSolver` | Packs allocations onto as few devices as possible | Yes |
| `SpreadSolver` | Distributes allocations across as many devices as possible | No |

An orchestrator that runs its own placement expresses the result as `name` constraints, or speaks the explicit protocol to the daemon directly.

```
Constraint spec  →  Solver (reads status)  →  Explicit request  →  Daemon (check-and-subtract)
                                                      ↑
                                       Orchestrators enter here
```

The daemon is unaware of solvers. New solvers can be added without any daemon changes.

---

## Reservation hierarchy

```
Request
└── Reservation  (one per independent lifetime)
    └── Allocation  (one per physical device)
        └── Resource  (ram_mb, compute_units claimed)
```

- **Request** — atomic unit. All reservations granted together or all rejected. One wire message.
- **Reservation** — named grouping with its own lease fd. Granted atomically, released independently. The caller declares boundaries based on intended lifetime.
- **Allocation** — one physical device and the capacity claimed on it. All allocations in a reservation are released when its lease fd closes. The same `DeviceAllocation` record is what the client requests, the daemon grants, the ledger holds, and status reports.
- **Resource** — capacity claimed on the allocation's device. Multiple allocations can share a physical device as long as combined capacity fits.

---

## Wire protocol

Line-delimited JSON over a Unix domain socket. Each message is one JSON object terminated by `\n`.

**Reserve request with explicit physical device names**

```json
{
  "op": "reserve",
  "reservations": [
    {
      "label": "checkpoint",
      "allocations": [{ "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782", "ram_mb": 8192, "compute_units": 50 }]
    },
    {
      "label": "lora",
      "allocations": [{ "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782", "ram_mb": 2048, "compute_units": 0 }]
    }
  ]
}
```

**Reserve response (success):**

```json
{
  "ok": true,
  "reservations": [
    {
      "reservation_id": "a3f9c2",
      "label": "checkpoint",
      "allocations": [{ "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782", "ram_mb": 8192, "compute_units": 50 }]
    },
    {
      "reservation_id": "b7e1d4",
      "label": "lora",
      "allocations": [{ "name": "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782", "ram_mb": 2048, "compute_units": 0 }]
    }
  ]
}
```

Each granted allocation is the requested record echoed back: the same
`name`, `ram_mb`, and `compute_units`. The name is the physical identity of
the device. For CUDA it is `cuda:GPU-<uuid>`, the UUID printed by
`nvidia-smi -L`, which `CUDA_VISIBLE_DEVICES` accepts directly and torch
reports for each visible device. Nothing else identifies a device.

**Reserve response (failure):**

```json
{ "ok": false, "error": "insufficient resources" }
```

**Status request / response:**

```json
{ "op": "status" }
```

Response contains device availability and active reservations.

---

## Lease fd mechanics

Each granted reservation is backed by a `socketpair()`:

- The daemon retains one end (the watch fd)
- The client receives the other end via `SCM_RIGHTS` alongside the JSON response
- The transport socket is closed immediately after the exchange
- The daemon watches its ends with `select` — EOF on its end = client closed theirs = release

```
client opens transport socket, sends reserve request
  → daemon grants atomically
  → daemon creates N socketpairs (one per reservation)
  → sends JSON response + N lease fds via SCM_RIGHTS
  → client receives response and fds
  → client closes transport socket
  → each lease fd held open = reservation alive
  → fd closed (release, exit, crash) → daemon detects EOF → resources freed
```

`check()` in the SDK performs a non-blocking read on the client's lease fd. EOF means the daemon closed its end (restart), raising `ReservationExpiredError`.

---

## Socket semantics (fork/exec)

The lease fd is a normal file descriptor. Standard POSIX semantics apply.

**fork()** — duplicates all fds into the child. Both parent and child hold the reservation. The reservation lives until the last copy of the fd is closed.

**exec()** — a lease fd must be inheritable to survive exec. The `reserve` CLI
sets this up, sets environment variables, and replaces itself with CMD:

```bash
reserve worker type=cuda,ram=8192 -- /path/to/model-env/bin/python worker.py
```

CMD inherits the fd and holds the lease while the fd remains open. A Python
worker can call `adopt_inherited_lease()` once to manage that same lease with
the SDK. Adoption removes `RESERVED_FD` from the worker's environment and from
the environment inherited by later children. The descriptor closes on a later
`exec`, so an unrelated executable cannot extend the lease. A child that only
forks still holds a descriptor copy. To hand off a lease across `exec`, do so
before adopting it in the process that will manage its lifetime.
See [Getting started](GETTING_STARTED.md#hold-capacity-for-a-model-worker).

---

## CUDA device indexing

Physical device names identify hardware in the config, CLI `name=` specs,
solver, ledger, protocol, and SDK grant metadata. A CUDA name is
`cuda:GPU-<uuid>`. The UUID is the one `nvidia-smi -L` prints, so it names one
specific board regardless of the driver's enumeration order, and
`CUDA_VISIBLE_DEVICES` accepts it directly. For example,
`reserve infer name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782 -- python infer.py`
claims that card. The CLI exports the name in `RESERVED_DEVICE` and
`RESERVED_DEVICES`, passes the granted allocations in
`RESERVED_GRANTED_ALLOCATIONS` for SDK adoption, and sets
`CUDA_VISIBLE_DEVICES=GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782` so the worker sees only
the granted GPU. Multiple CUDA allocations contribute distinct UUIDs in first-grant
order.

`torch_device(allocation)` is the conversion boundary; `reservation.devices` holds the granted `DeviceAllocation` records it takes. It enumerates the
CUDA devices torch can see, reads each one's UUID from its device properties,
and returns the index whose UUID matches the grant. The worker above gets
`torch.device("cuda:0")`. A process started with
`CUDA_VISIBLE_DEVICES=GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781,GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782`
gets `torch.device("cuda:1")` for the same grant, and a process with no mask
gets the card's position in torch's full enumeration. `Reservation.torch_devices()`
applies the conversion in grant order. Multiple allocations on the same GPU resolve
to the same logical index.

The SDK never reads or writes `CUDA_VISIBLE_DEVICES`; it only reports the mask
in error messages. Resolving a CUDA grant initializes CUDA in the calling
process, so it belongs in the process that will use the GPU. A grant torch
cannot see and a torch build that does not report device UUIDs raise
`ReservationError`.

---

## Daemon internals

### Functional core (`core.py`)

All business logic as pure functions. No IO, no global state. Every function takes a `Ledger` and returns a new `Ledger` plus a result. The ledger is an immutable frozen dataclass.

```python
@dataclass(frozen=True)
class Ledger:
    availability: tuple[DeviceAvailability, ...]
    reservations: tuple[ActiveReservation, ...]

def grant(ledger, request) -> tuple[Ledger, GrantResult]: ...
def release(ledger, reservation_id) -> Ledger: ...
def status(ledger) -> StatusResult: ...
```

`grant()` does a two-pass atomic check-and-subtract. If any allocation fails, the original ledger is returned unchanged.

Fully testable as plain function calls — no daemon, no sockets, no hardware required.

### State layer (`daemon_state.py`)

Between the core and the sockets sits a pure state machine. One daemon step is a function of logical state and one event, returning new state and the effects the IO shell must perform:

```text
DaemonState + Event -> DaemonState + Effects
```

`DaemonState` holds the ledger, the open transports (connection id and buffered bytes), and the open leases (lease id and reservation id). It never holds a socket or a file descriptor: descriptors are process-local, recyclable integers, so the state layer identifies connections and leases by logical ids and the shell keeps the mapping to real sockets in a handle table.

Events are what the shell observed: `Accepted`, `TransportReceived`, `TransportClosed`, `LeaseClosed`. Effects are what the shell must do: `SendBytes`, `CreateLease`, `SendLeaseFds`, `CloseTransport`, `CloseLease`.

The three flows:

- **Reserve.** Parse the request, call `core.grant()`. On failure the ledger is unchanged and the effects are a failure response and `CloseTransport`. On success the state carries the new ledger and one lease record per granted reservation, and the effects are `CreateLease` per lease, one `SendLeaseFds` with the response, and `CloseTransport`.
- **Status.** Call `core.status()`, emit the response and `CloseTransport`. Ledger and leases unchanged.
- **Lease close.** Look up the lease, `core.release()` its reservation, drop the lease record, emit `CloseLease`. This is the lifecycle invariant: a reservation exists exactly while its lease is open.

A request that arrives without its terminating newline only extends the transport's buffer; nothing executes until the line is complete. Malformed input produces a failure response and no ledger change.

`tests/test_daemon_state.py` exercises all of this as plain function calls, with no sockets.

### IO shell (`daemon.py`)

The shell owns the real sockets and the `select` loop. It translates readable descriptors into events, calls `daemon_state.step()`, and applies the returned effects: `CreateLease` makes a `socketpair()` and files the daemon end under the lease id, `SendLeaseFds` sends the client ends with `SCM_RIGHTS`, `SendBytes` writes a response, and the close effects close sockets and update the handle table. It also owns socket acquisition, signals, and the optional HTTP status server.

No allocation logic and no state transitions live in the shell — only IO, descriptor lifecycle, and effect application. `tests/test_daemon_startup.py` and the inherited-lease tests exercise it against a real daemon process.

### Module structure

```
reserved/
  core.py          # pure functions, immutable Ledger
  daemon_state.py  # pure state machine: DaemonState + Event -> DaemonState + Effects
  daemon.py        # IO shell: sockets, select loop, SCM_RIGHTS, effect application
  paths.py       # socket discovery shared by daemon, SDK, and CLIs
  protocol.py    # single-line JSON serialization (explicit types only)
  sdk/
    client.py    # Reservation, Request, DeviceConstraint, torch_device
    solvers.py   # ConsolidateSolver, SpreadSolver
    errors.py    # ReservationError, DaemonNotRunningError, ReservationExpiredError
  bin/
    reserve.py       # CLI: constraint solver + exec wrapper
    reserve_status.py  # CLI: status query
    generate_config.py # reserved-generate-config: config from detected hardware
```

### Language boundaries

The constraint language never crosses a process boundary. `DeviceConstraint` and the solver classes are in-process Python objects only — never serialized. Only explicit requests of `DeviceAllocation` records hit the wire. `protocol.py` never touches constraint types.

---

## Daemon restart

When the daemon restarts, all lease socketpairs are closed. Clients detect this via `check()` (EOF on their lease fd → `ReservationExpiredError`). The ledger starts clean — no attempt is made to restore prior reservations.

The daemon logs all active reservations at `SIGTERM` so operators can see what work will be disrupted before restarting.

---

## What reserved does NOT do

- Queue requests
- Persist state across restarts
- Do constraint solving (that's the SDK/CLI layer)
- Know anything about tasks, pipelines, or workers
- Enforce against non-participating processes
- Multi-machine coordination
