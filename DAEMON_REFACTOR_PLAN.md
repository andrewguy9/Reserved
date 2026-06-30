# Daemon Refactor Plan

## Goal

Make the daemon testable at the right layer: a single atomic pass over daemon
state, driven by explicit events, with observable effects returned to the IO
shell.

The current daemon is single-threaded and low-volume. It serves one request at a
time, and each request should behave like a state transition:

```text
DaemonState + Event -> DaemonState + Effects
```

This matches the existing functional core:

```text
Ledger + ReserveRequest -> Ledger + GrantResult
```

The refactor should preserve that architecture instead of relying primarily on
threaded or subprocess integration tests.

## Target Shape

The daemon should become a three-layer cake:

```text
1. State transition layer
   DaemonState + Event -> DaemonState + Effects

2. Event loop layer
   poll/select readable handles, translate them into Events, apply one step

3. Serve-forever lifecycle
   socket acquisition, signal setup, optional HTTP server, shutdown cleanup
```

Most tests should target layer 1. A few tests can cover layer 2. Layer 3 should
stay thin enough to need minimal direct testing.

## Do Not Put Raw FDs In State

Raw file descriptors are process-local, recyclable integers. They are necessary
for the IO shell, but they should not be the semantic identity inside the state
machine.

Use logical handle IDs in daemon state:

```python
ConnectionId = str
LeaseId = str
ReservationId = str

@dataclass(frozen=True)
class TransportRecord:
    connection_id: ConnectionId
    buffer: bytes

@dataclass(frozen=True)
class LeaseRecord:
    lease_id: LeaseId
    reservation_id: ReservationId

@dataclass(frozen=True)
class DaemonState:
    ledger: Ledger
    transports: dict[ConnectionId, TransportRecord]
    leases: dict[LeaseId, LeaseRecord]
    running: bool = True
```

Keep sockets and file descriptors in an outer handle table owned by the
imperative shell:

```python
@dataclass
class HandleTable:
    transports: dict[ConnectionId, socket.socket]
    leases: dict[LeaseId, socket.socket]
    fd_to_connection: dict[int, ConnectionId]
    fd_to_lease: dict[int, LeaseId]
```

The state layer should be testable without constructing real sockets.

## Events

The IO layer should translate socket activity into semantic events:

```python
@dataclass(frozen=True)
class Accepted:
    connection_id: ConnectionId

@dataclass(frozen=True)
class TransportReceived:
    connection_id: ConnectionId
    data: bytes

@dataclass(frozen=True)
class TransportClosed:
    connection_id: ConnectionId

@dataclass(frozen=True)
class LeaseClosed:
    lease_id: LeaseId
```

The poll/select layer owns the mapping from readable fd to event type. The state
layer only sees events.

## Effects

The state layer should return effects for the IO shell to perform:

```python
@dataclass(frozen=True)
class SendBytes:
    connection_id: ConnectionId
    data: bytes

@dataclass(frozen=True)
class CreateLease:
    lease_id: LeaseId

@dataclass(frozen=True)
class SendLeaseFds:
    connection_id: ConnectionId
    data: bytes
    lease_ids: tuple[LeaseId, ...]

@dataclass(frozen=True)
class CloseTransport:
    connection_id: ConnectionId

@dataclass(frozen=True)
class CloseLease:
    lease_id: LeaseId
```

The IO shell applies these effects:

- `CreateLease` creates a `socketpair()` and records the daemon end by logical
  lease ID.
- `SendLeaseFds` translates logical lease IDs into client fds and sends them via
  `SCM_RIGHTS`.
- `SendBytes` writes ordinary JSON responses.
- `CloseTransport` and `CloseLease` close real sockets and update the handle
  table.

## Reserve Success Flow

For a complete reserve request, one daemon step should:

1. Parse the explicit reserve request.
2. Call `core.grant(state.ledger, request)`.
3. If grant fails:
   - keep the ledger unchanged
   - emit a failure response
   - close the transport
4. If grant succeeds:
   - store the new ledger
   - create one logical lease record per granted reservation
   - emit `CreateLease` for each lease
   - emit `SendLeaseFds` with the success response and lease IDs
   - emit `CloseTransport`

The test should assert state and effects. It should not assert raw fd values.

## Status Flow

For a status request, one daemon step should:

1. Call `core.status(state.ledger)`.
2. Emit the status JSON response.
3. Emit `CloseTransport`.
4. Leave ledger and lease state unchanged.

## Lease Close Flow

For a lease close event, one daemon step should:

1. Look up the logical lease.
2. Release the associated reservation via `core.release`.
3. Remove the lease record from state.
4. Emit `CloseLease`.

This is the primary lifecycle invariant: the reservation exists while the lease
is open, and disappears when the lease closes.

## Suggested Modules

Keep the public package layout simple, but separate responsibilities:

```text
reserved/
  core.py          # existing pure ledger logic
  protocol.py      # existing wire JSON
  daemon_state.py  # new event/effect/state transition layer
  daemon.py        # IO shell: sockets, select, SCM_RIGHTS, signals, HTTP
```

`daemon.py` should import and apply `daemon_state.step()`.

## Test Strategy

Add `tests/test_daemon_state.py` for state-machine tests.

High-value tests:

- **status is read-only**
  - given ledger with devices/reservations
  - event: complete status request
  - assert ledger unchanged
  - assert `SendBytes` contains status
  - assert `CloseTransport`

- **reserve success is atomic**
  - given empty ledger
  - event: complete explicit reserve request
  - assert capacity reduced
  - assert reservation recorded
  - assert one logical lease per granted reservation
  - assert `CreateLease`, `SendLeaseFds`, `CloseTransport`

- **reserve failure is atomic**
  - request more capacity than available
  - assert ledger unchanged
  - assert no leases created
  - assert failure response and close

- **partial transport data does not execute**
  - event: request bytes without newline
  - assert buffer updated
  - assert ledger unchanged
  - assert no response effects
  - event: remaining bytes including newline
  - assert request handled once

- **multi-reservation release is independent**
  - grant two reservations
  - event: `LeaseClosed(first_lease)`
  - assert only first reservation released
  - event: `LeaseClosed(second_lease)`
  - assert all capacity restored

- **malformed request does not corrupt state**
  - bad JSON or malformed reserve payload
  - assert no ledger mutation
  - assert failure/close behavior

## Keep Integration Tests Narrow

After the state layer is covered, add a small number of IO-shell tests:

- fd classification maps readable transport fd to `TransportReceived`
- fd classification maps readable lease fd to `LeaseClosed`
- `SendLeaseFds` effect sends the expected number of fds over a real Unix socket

Avoid full daemon subprocess tests until these seams are in place.

## Avoid

- Do not assert exact log messages.
- Do not assert raw fd numbers.
- Do not encode tests around private dictionary ordering.
- Do not assert exact HTML layout.
- Do not make tests depend on fixed TCP ports.
- Do not retest detailed allocation behavior already covered by `core.py`.

## Expected Coverage Impact

The current daemon has effectively no coverage. State-machine tests should cover
most daemon semantics without fragile threading or subprocess setup.

Expected result after the refactor:

- `daemon_state.py`: high coverage
- `daemon.py`: moderate coverage for IO effect application and polling
- project total coverage should rise materially, but the main value is covering
  the daemon's reservation and lease lifecycle invariants.

## Current Status

Implemented:

- Added `reserved/daemon_state.py` with explicit daemon state, events, effects,
  and a single-step transition function.
- Refactored `reserved/daemon.py` so the select loop translates real socket
  activity into state events and applies returned effects through an outer
  handle table.
- Kept raw sockets and file descriptors out of daemon state. The state layer
  uses logical connection and lease IDs only.
- Added `tests/test_daemon_state.py` covering status, reserve success, reserve
  failure, malformed input, partial transport reads, transport close, unknown
  handles, and lease release invariants.

Latest measured coverage after this pass:

- `daemon_state.py`: 100%
- project total: 52%
- `daemon.py`: 0%, because the remaining code is the IO shell and serve loop
  and still needs narrow shell-level tests.

Recommended next step:

- Add a small `tests/test_daemon_shell.py` for effect application and fd/event
  translation. Keep those tests narrow; do not introduce a full daemon
  subprocess harness unless the lower layers stop being sufficient.
