"""
daemon_state.py - Logical daemon state transitions.

This module contains the state-machine layer for the reserved daemon. It uses
logical connection and lease identifiers, not raw sockets or file descriptors.
The imperative daemon shell owns the actual handles and applies returned
effects.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import core, protocol
from .core import GrantFailure, Ledger


ConnectionId = str
LeaseId = str
ReservationId = str


@dataclass(frozen=True)
class TransportRecord:
    connection_id: ConnectionId
    buffer: bytes = b""


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


Event = Accepted | TransportReceived | TransportClosed | LeaseClosed


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


Effect = SendBytes | CreateLease | SendLeaseFds | CloseTransport | CloseLease


def initial_state(ledger: Ledger) -> DaemonState:
    return DaemonState(ledger=ledger, transports={}, leases={})


def step(state: DaemonState, event: Event) -> tuple[DaemonState, tuple[Effect, ...]]:
    if isinstance(event, Accepted):
        return _accepted(state, event.connection_id)
    if isinstance(event, TransportReceived):
        return _transport_received(state, event.connection_id, event.data)
    if isinstance(event, TransportClosed):
        return _transport_closed(state, event.connection_id)
    if isinstance(event, LeaseClosed):
        return _lease_closed(state, event.lease_id)
    raise TypeError(f"unknown daemon event: {event!r}")


def _accepted(
    state: DaemonState,
    connection_id: ConnectionId,
) -> tuple[DaemonState, tuple[Effect, ...]]:
    transports = dict(state.transports)
    transports[connection_id] = TransportRecord(connection_id=connection_id)
    return _replace_transports(state, transports), ()


def _transport_closed(
    state: DaemonState,
    connection_id: ConnectionId,
) -> tuple[DaemonState, tuple[Effect, ...]]:
    if connection_id not in state.transports:
        return state, (CloseTransport(connection_id),)
    transports = dict(state.transports)
    transports.pop(connection_id, None)
    return _replace_transports(state, transports), (CloseTransport(connection_id),)


def _transport_received(
    state: DaemonState,
    connection_id: ConnectionId,
    data: bytes,
) -> tuple[DaemonState, tuple[Effect, ...]]:
    transport = state.transports.get(connection_id)
    if transport is None:
        return state, ()

    buffer = transport.buffer + data
    if b"\n" not in buffer:
        transports = dict(state.transports)
        transports[connection_id] = TransportRecord(
            connection_id=connection_id,
            buffer=buffer,
        )
        return _replace_transports(state, transports), ()

    line, _, _rest = buffer.partition(b"\n")
    return _handle_request_line(state, connection_id, line)


def _handle_request_line(
    state: DaemonState,
    connection_id: ConnectionId,
    line: bytes,
) -> tuple[DaemonState, tuple[Effect, ...]]:
    try:
        obj = protocol.parse_request(line)
    except ValueError:
        return _close_transport_state(state, connection_id)

    op = obj.get("op")
    if op == "status":
        return _handle_status(state, connection_id)
    if op == "reserve":
        return _handle_reserve(state, connection_id, obj)
    return _close_transport_state(state, connection_id)


def _handle_status(
    state: DaemonState,
    connection_id: ConnectionId,
) -> tuple[DaemonState, tuple[Effect, ...]]:
    result = core.status(state.ledger)
    next_state = _without_transport(state, connection_id)
    return next_state, (
        SendBytes(connection_id, protocol.status_result_to_wire(result)),
        CloseTransport(connection_id),
    )


def _handle_reserve(
    state: DaemonState,
    connection_id: ConnectionId,
    obj: dict,
) -> tuple[DaemonState, tuple[Effect, ...]]:
    try:
        request = protocol.wire_to_reserve_request(obj)
    except (KeyError, TypeError, ValueError) as exc:
        next_state = _without_transport(state, connection_id)
        return next_state, (
            SendBytes(
                connection_id,
                protocol.grant_failure_to_wire(
                    GrantFailure(error=f"malformed request: {exc}")
                ),
            ),
            CloseTransport(connection_id),
        )

    new_ledger, result = core.grant(state.ledger, request)
    next_state = _without_transport(state, connection_id)

    if not result.ok:
        return next_state, (
            SendBytes(connection_id, protocol.grant_failure_to_wire(result)),
            CloseTransport(connection_id),
        )

    leases = dict(next_state.leases)
    lease_ids: list[LeaseId] = []
    for granted in result.reservations:
        lease_id = granted.reservation_id
        leases[lease_id] = LeaseRecord(
            lease_id=lease_id,
            reservation_id=granted.reservation_id,
        )
        lease_ids.append(lease_id)

    next_state = DaemonState(
        ledger=new_ledger,
        transports=next_state.transports,
        leases=leases,
        running=next_state.running,
    )
    return next_state, (
        *(CreateLease(lease_id) for lease_id in lease_ids),
        SendLeaseFds(
            connection_id,
            protocol.grant_success_to_wire(result),
            tuple(lease_ids),
        ),
        CloseTransport(connection_id),
    )


def _lease_closed(
    state: DaemonState,
    lease_id: LeaseId,
) -> tuple[DaemonState, tuple[Effect, ...]]:
    lease = state.leases.get(lease_id)
    if lease is None:
        return state, (CloseLease(lease_id),)

    leases = dict(state.leases)
    leases.pop(lease_id, None)
    return DaemonState(
        ledger=core.release(state.ledger, lease.reservation_id),
        transports=state.transports,
        leases=leases,
        running=state.running,
    ), (CloseLease(lease_id),)


def _close_transport_state(
    state: DaemonState,
    connection_id: ConnectionId,
) -> tuple[DaemonState, tuple[Effect, ...]]:
    next_state = _without_transport(state, connection_id)
    return next_state, (CloseTransport(connection_id),)


def _without_transport(state: DaemonState, connection_id: ConnectionId) -> DaemonState:
    transports = dict(state.transports)
    transports.pop(connection_id, None)
    return _replace_transports(state, transports)


def _replace_transports(
    state: DaemonState,
    transports: dict[ConnectionId, TransportRecord],
) -> DaemonState:
    return DaemonState(
        ledger=state.ledger,
        transports=transports,
        leases=state.leases,
        running=state.running,
    )
