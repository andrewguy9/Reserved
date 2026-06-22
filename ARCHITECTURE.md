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

The daemon speaks only in explicit terms. Every slot names a specific device and states exactly how much capacity to claim:

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

A **constraint solver** resolves constraints against current device status and produces an explicit request before submitting to the daemon.

### Constraint solvers

| Solver | Behaviour | Default |
| --- | --- | --- |
| `ConsolidateSolver` | Packs slots onto as few devices as possible | Yes |
| `SpreadSolver` | Distributes slots across as many devices as possible | No |

Any caller can bypass solvers by submitting an explicit request directly to the daemon — this is what sophisticated orchestrators do after running their own placement algorithm.

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

**Reserve request (explicit — all slots name specific devices):**

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

`cuda_ordinal` is the physical CUDA ordinal from the device name (e.g. `cuda:2` → `2`). Used directly in `CUDA_VISIBLE_DEVICES` by the CLI. `-1` for non-CUDA devices.

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

**exec()** — Python sets `O_CLOEXEC` by default. Clear it explicitly to pass the lease across exec:

```python
os.set_inheritable(r._sock.fileno(), True)
os.execvp("worker", ["worker"])
```

This is how the `reserve` CLI works: acquires the lease, clears `FD_CLOEXEC`, sets env vars, execs CMD. CMD inherits the fd and holds the lease for its lifetime without knowing about `reserved`.

---

## CUDA device indexing

Two distinct concepts:

- **Physical ordinal** — the integer in the device name (`cuda:2` → `2`). Goes into `CUDA_VISIBLE_DEVICES`.
- **Logical index** — 0-based position within what `CUDA_VISIBLE_DEVICES` exposes. What `torch.device("cuda:N")` refers to inside the child process.

The CLI sets `CUDA_VISIBLE_DEVICES` to physical ordinals (deduplicated for multi-slot same-device grants). The child process addresses devices as `cuda:0`, `cuda:1`, etc.

The SDK does **not** set `CUDA_VISIBLE_DEVICES`. `torch_device()` returns the physical ordinal directly (e.g. `torch.device("cuda:2")`). Use the CLI if you need process-level GPU masking.

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
  protocol.py    # single-line JSON serialization (explicit types only)
  sdk/
    client.py    # Reservation, Request, DeviceConstraint, ExplicitSlot
    solvers.py   # ConsolidateSolver, SpreadSolver
    errors.py    # ReservationError, DaemonNotRunningError, ReservationExpiredError
  bin/
    reserve.py       # CLI: constraint solver + exec wrapper
    reserve_status.py  # CLI: status query
```

### Language boundaries

The constraint language never crosses a process boundary. `DeviceConstraint`, `ExplicitSlot`, and solver classes are in-process Python objects only — never serialized. Only explicit requests hit the wire. `protocol.py` never touches constraint types.

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
