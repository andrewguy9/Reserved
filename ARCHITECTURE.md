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

The daemon speaks only in explicit terms. Every slot names a physical device and states exactly how much capacity to claim:

```
slot: name=cuda:0  ram_mb=8192  compute_units=50
```

No wildcards. No type matching. The daemon checks whether the named device has sufficient capacity and subtracts if so.

### Constraint language (SDK and CLI)

Users typically don't know or care which physical device they get — they just need a CUDA device with enough VRAM:

```
slot: type=cuda  ram_mb=8192
slot: type=cuda  model=rtx3090
```

A **constraint solver** is an ordinary pure function taking a supplied capacity
snapshot and SDK `ReservationSpec` records. Each specification contains a label
and device requirements. The function carries labels unchanged and returns a
concrete `ReserveRequest`, or `NoPlacement(reason)` when its policy finds no
complete placement. An empty specification batch is a successful empty request.
There is no IO or input mutation in solving.

`acquire_reservations(request)` submits an already concrete batch without
querying status or invoking a solver. `Request(..., solver=...)` is the convenience
workflow that obtains status, calls the chosen function, checks the returned
reservation count and order, and acquires the batch. It chooses one solver for
the batch; conflicting standalone solver choices on member reservations are
rejected. Even explicit declarations passed through this convenience workflow
use a status snapshot; callers with concrete requests can acquire them directly.

### Constraint solvers

| Solver | Behaviour | Default |
| --- | --- | --- |
| `solve_reservations_consolidated` | Greedily prefers eligible devices with less free RAM | Yes |
| `solve_reservations_spread` | Greedily prefers eligible devices with more free RAM | No |

Both policies attempt all specifications in order and produce one complete
candidate, never a partial batch. They do not enumerate alternatives or prove
infeasibility. Explicit assignments retain passthrough behavior; the daemon
checks whether their named devices and amounts can actually be acquired.
Best-effort placement and richer optimization semantics are outside this API.

Any caller can bypass solvers by submitting an explicit request directly to the daemon — this is what sophisticated orchestrators do after running their own placement algorithm.

```
Capacity + SDK specifications → Pure solver → Concrete request → Acquisition → Daemon
Orchestrator-provided concrete request ────────────────────────→ Acquisition → Daemon
```

The daemon is unaware of solvers. New solvers can be added without any daemon changes.

---

## Reservation hierarchy

```
Request
└── Reservation  (one per independent lifetime)
    └── Slot     (one per physical device)
        └── Resource  (ram_mb, compute_units claimed)
```

- **Request** — atomic unit. All reservations granted together or all rejected. One wire message.
- **Reservation** — named grouping with its own lease fd. Granted atomically, released independently. The caller declares boundaries based on intended lifetime.
- **Slot** — one physical device. All slots in a reservation are released when its lease fd closes.
- **Resource** — capacity claimed on the slot's device. Multiple slots can share a physical device as long as combined capacity fits.

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
      "slots": [{ "name": "cuda:2", "ram_mb": 8192, "compute_units": 50 }]
    },
    {
      "label": "lora",
      "slots": [{ "name": "cuda:2", "ram_mb": 2048, "compute_units": 0 }]
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
      "slots": [{ "device_name": "cuda:2", "cuda_ordinal": 2 }]
    },
    {
      "reservation_id": "b7e1d4",
      "label": "lora",
      "slots": [{ "device_name": "cuda:2", "cuda_ordinal": 2 }]
    }
  ]
}
```

`device_name` identifies the physical grant. `cuda_ordinal` is its physical
CUDA ordinal (for example, `2` for `cuda:2`) or `-1` for a non-CUDA grant.

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

Physical device names identify hardware in the config, CLI `name=` slots,
solver, ledger, protocol, and SDK grant metadata. For example,
`reserve infer name=cuda:2 -- python infer.py` claims physical `cuda:2`.
The CLI exports that name in `RESERVED_DEVICE` and `RESERVED_DEVICES`, and
passes the physical grant metadata in `RESERVED_GRANTED_SLOTS` for SDK
adoption. `CUDA_VISIBLE_DEVICES=2` exposes the granted GPU to the worker.
Multiple CUDA slots contribute distinct physical ordinals in first-grant order.

`GrantedDevice.torch_device()` is the conversion boundary. It maps the
physical ordinal to its zero-based position in the current process's CUDA
mask. The worker above gets `torch.device("cuda:0")`. A process with
`CUDA_VISIBLE_DEVICES=5,2` gets `torch.device("cuda:1")` for the same physical
grant. `Reservation.torch_devices()` applies the conversion in grant order.
Multiple slots on the same GPU resolve to the same logical index.

An unset mask uses the physical ordinal as the logical index, assuming the
unmasked CUDA order matches the configured physical ordinals. Configure
visibility before CUDA initializes. The SDK checks that the mask is a list of
distinct nonnegative decimal ordinals and that the grant appears in it. Empty
and `-1` masks hide all CUDA devices. Hidden grants and unsupported masks
raise `ReservationError`.

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

`grant()` does a two-pass atomic check-and-subtract. If any slot fails, the original ledger is returned unchanged.

Fully testable as plain function calls — no daemon, no sockets, no hardware required.

### Imperative shell (`daemon.py`)

Owns the single mutable ledger reference and the `select` loop. Calls core functions and swaps the ledger reference with the result.

```python
ledger = core.load_ledger(config)

# on reserve request:
ledger, result = core.grant(ledger, request)

# on lease fd EOF:
ledger = core.release(ledger, reservation_id)
```

No allocation logic in the shell — only IO, fd lifecycle, and ledger swapping.

### Module structure

```
reserved/
  core.py        # pure functions, immutable Ledger
  daemon.py      # imperative shell, select loop, SCM_RIGHTS
  paths.py       # socket discovery shared by daemon, SDK, and CLIs
  protocol.py    # single-line JSON serialization (explicit types only)
  sdk/
    client.py    # Reservation, Request, DeviceConstraint, ExplicitSlot
    solvers.py   # pure placement functions and NoPlacement
    errors.py    # ReservationError, DaemonNotRunningError, ReservationExpiredError
  bin/
    reserve.py       # CLI: constraint solver + exec wrapper
    reserve_status.py  # CLI: status query
```

### Language boundaries

The constraint language never crosses a process boundary. `DeviceConstraint`,
`ExplicitSlot`, and `ReservationSpec` are SDK objects. Only concrete requests
hit the wire; `protocol.py` never touches constraint types. The parser requires
a physical device name for each slot but currently ignores additional fields.

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
