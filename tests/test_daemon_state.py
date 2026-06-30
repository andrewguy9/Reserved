"""
Invariant-focused tests for the daemon state-machine layer.

These tests exercise one daemon step at a time with logical IDs only. They do
not open sockets or file descriptors.
"""

import pytest

from reserved import protocol
from reserved.core import (
    ReservationRequest,
    ReserveRequest,
    SlotRequest,
    load_ledger,
)
from reserved.daemon_state import (
    Accepted,
    CloseLease,
    CloseTransport,
    CreateLease,
    LeaseClosed,
    SendBytes,
    SendLeaseFds,
    TransportClosed,
    TransportReceived,
    initial_state,
    step,
)


CONFIG = {
    "devices": [
        {
            "name": "cuda:0",
            "type": "cuda",
            "model": "rtx3090",
            "ram_mb": 24576,
            "compute_units": 100,
        },
        {
            "name": "cuda:1",
            "type": "cuda",
            "model": "rtx3090",
            "ram_mb": 24576,
            "compute_units": 100,
        },
    ]
}


def make_state():
    return initial_state(load_ledger(CONFIG))


def accept(state, connection_id="conn-1"):
    next_state, effects = step(state, Accepted(connection_id))
    assert effects == ()
    return next_state


def reserve_wire(
    label: str,
    device_name: str = "cuda:0",
    ram_mb: int = 8192,
    compute_units: int = 0,
) -> bytes:
    return protocol.reserve_request_to_wire(
        ReserveRequest(
            reservations=(
                ReservationRequest(
                    label=label,
                    slots=(
                        SlotRequest(
                            name=device_name,
                            ram_mb=ram_mb,
                            compute_units=compute_units,
                        ),
                    ),
                ),
            )
        )
    )


def multi_reserve_wire() -> bytes:
    return protocol.reserve_request_to_wire(
        ReserveRequest(
            reservations=(
                ReservationRequest(
                    label="train-a",
                    slots=(SlotRequest(name="cuda:0", ram_mb=4096),),
                ),
                ReservationRequest(
                    label="train-b",
                    slots=(SlotRequest(name="cuda:0", ram_mb=2048),),
                ),
            )
        )
    )


def available_ram(ledger, device_name):
    return next(
        da.available_ram_mb
        for da in ledger.availability
        if da.device.name == device_name
    )


def effect_of_type(effects, effect_type):
    matches = [effect for effect in effects if isinstance(effect, effect_type)]
    assert len(matches) == 1
    return matches[0]


def effects_of_type(effects, effect_type):
    return [effect for effect in effects if isinstance(effect, effect_type)]


def test_accept_adds_logical_transport_without_touching_ledger():
    state = make_state()

    next_state = accept(state)

    assert "conn-1" in next_state.transports
    assert next_state.ledger == state.ledger
    assert next_state.leases == {}


def test_partial_transport_read_buffers_until_request_is_complete():
    state = accept(make_state())

    next_state, effects = step(state, TransportReceived("conn-1", b'{"op"'))

    assert effects == ()
    assert next_state.transports["conn-1"].buffer == b'{"op"'
    assert next_state.ledger == state.ledger
    assert next_state.leases == {}


def test_transport_close_removes_transport_without_mutating_state():
    state = accept(make_state())

    next_state, effects = step(state, TransportClosed("conn-1"))

    assert next_state.ledger == state.ledger
    assert next_state.leases == {}
    assert "conn-1" not in next_state.transports
    assert effects == (CloseTransport("conn-1"),)


def test_unknown_transport_events_are_idempotent():
    state = make_state()

    next_state, effects = step(
        state,
        TransportReceived("missing", b'{"op":"status"}\n'),
    )

    assert next_state == state
    assert effects == ()

    closed_state, close_effects = step(state, TransportClosed("missing"))

    assert closed_state == state
    assert close_effects == (CloseTransport("missing"),)


def test_invalid_json_closes_transport_without_mutating_ledger():
    state = accept(make_state())

    next_state, effects = step(state, TransportReceived("conn-1", b"not json\n"))

    assert next_state.ledger == state.ledger
    assert next_state.leases == {}
    assert "conn-1" not in next_state.transports
    assert effects == (CloseTransport("conn-1"),)


def test_status_request_is_read_only_and_closes_transport():
    state = accept(make_state())

    next_state, effects = step(
        state,
        TransportReceived("conn-1", protocol.status_request_to_wire()),
    )

    send = effect_of_type(effects, SendBytes)
    response = protocol.decode(send.data)

    assert next_state.ledger == state.ledger
    assert next_state.leases == {}
    assert "conn-1" not in next_state.transports
    assert response["reservations"] == []
    assert {device["name"] for device in response["devices"]} == {"cuda:0", "cuda:1"}
    assert CloseTransport("conn-1") in effects


def test_successful_reserve_commits_ledger_and_creates_matching_lease_effects():
    state = accept(make_state())
    before_ram = available_ram(state.ledger, "cuda:0")

    next_state, effects = step(
        state,
        TransportReceived("conn-1", reserve_wire("train", ram_mb=8192)),
    )

    send = effect_of_type(effects, SendLeaseFds)
    response = protocol.decode(send.data)
    create_effects = effects_of_type(effects, CreateLease)
    lease_ids = tuple(next_state.leases)

    assert available_ram(next_state.ledger, "cuda:0") == before_ram - 8192
    assert len(next_state.ledger.reservations) == 1
    assert lease_ids == (next_state.ledger.reservations[0].reservation_id,)
    assert [effect.lease_id for effect in create_effects] == list(lease_ids)
    assert send.lease_ids == lease_ids
    assert response["ok"] is True
    assert response["reservations"][0]["reservation_id"] in lease_ids
    assert "conn-1" not in next_state.transports
    assert CloseTransport("conn-1") in effects


def test_failed_reserve_leaves_ledger_and_leases_unchanged():
    state = accept(make_state())

    next_state, effects = step(
        state,
        TransportReceived("conn-1", reserve_wire("too-big", ram_mb=999999)),
    )

    send = effect_of_type(effects, SendBytes)
    response = protocol.decode(send.data)

    assert next_state.ledger == state.ledger
    assert next_state.leases == {}
    assert effects_of_type(effects, CreateLease) == []
    assert response["ok"] is False
    assert "insufficient" in response["error"]
    assert "conn-1" not in next_state.transports
    assert CloseTransport("conn-1") in effects


def test_malformed_reserve_closes_transport_without_corrupting_state():
    state = accept(make_state())
    malformed = b'{"op":"reserve","reservations":[{"slots":[{"name":"cuda:0"}]}]}\n'

    next_state, effects = step(state, TransportReceived("conn-1", malformed))

    send = effect_of_type(effects, SendBytes)
    response = protocol.decode(send.data)

    assert next_state.ledger == state.ledger
    assert next_state.leases == {}
    assert effects_of_type(effects, CreateLease) == []
    assert response["ok"] is False
    assert "malformed request" in response["error"]
    assert "conn-1" not in next_state.transports


def test_lease_close_releases_only_the_matching_reservation():
    initial = make_state()
    state = accept(initial)
    reserved_state, _ = step(
        state,
        TransportReceived("conn-1", multi_reserve_wire()),
    )
    lease_ids = tuple(reserved_state.leases)
    first_lease, second_lease = lease_ids
    first_reservation = next(
        reservation
        for reservation in reserved_state.ledger.reservations
        if reservation.reservation_id == first_lease
    )

    next_state, effects = step(reserved_state, LeaseClosed(first_lease))

    assert effects == (CloseLease(first_lease),)
    assert first_lease not in next_state.leases
    assert second_lease in next_state.leases
    assert {r.reservation_id for r in next_state.ledger.reservations} == {second_lease}
    assert available_ram(next_state.ledger, "cuda:0") == (
        available_ram(reserved_state.ledger, "cuda:0")
        + first_reservation.slots[0].ram_mb
    )

    restored_state, effects = step(next_state, LeaseClosed(second_lease))

    assert effects == (CloseLease(second_lease),)
    assert restored_state.leases == {}
    assert restored_state.ledger == initial.ledger


def test_unknown_lease_close_is_idempotent_but_requests_handle_cleanup():
    state = make_state()

    next_state, effects = step(state, LeaseClosed("missing"))

    assert next_state == state
    assert effects == (CloseLease("missing"),)


def test_unknown_operation_closes_transport_without_mutating_ledger():
    state = accept(make_state())

    next_state, effects = step(
        state,
        TransportReceived("conn-1", b'{"op":"cancel"}\n'),
    )

    assert next_state.ledger == state.ledger
    assert next_state.leases == {}
    assert effects == (CloseTransport("conn-1"),)
    assert "conn-1" not in next_state.transports


def test_unknown_event_type_is_rejected():
    with pytest.raises(TypeError):
        step(make_state(), object())  # type: ignore[arg-type]
