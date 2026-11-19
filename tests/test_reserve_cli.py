"""
Invariant-focused tests for the reserve CLI helpers.
"""

import json

import pytest

from reserved.bin.reserve import build_exec_env, build_reserve_request, main, parse_slot
from reserved.core import ReservationRequest, ReserveRequest, SlotRequest
from reserved.sdk.client import DeviceConstraint, ExplicitSlot, ReservationSpec
from reserved.sdk.solvers import NoPlacement, solve_reservations_consolidated


def test_parse_slot_ram_and_compute_aliases_are_equivalent():
    short = parse_slot("type=cuda,model=a6000,ram=8192,compute=25")
    long = parse_slot("type=cuda,model=a6000,ram_mb=8192,compute_units=25")

    assert isinstance(short, DeviceConstraint)
    assert isinstance(long, DeviceConstraint)
    assert short == long


def test_parse_slot_name_selects_explicit_slot():
    slot = parse_slot("name=cuda:2,type=cuda,ram=4096,compute=50")

    assert isinstance(slot, DeviceConstraint)
    assert slot.type == "cuda"
    assert slot.name == "cuda:2"
    assert slot.ram_mb == 4096
    assert slot.compute_units == 50


def test_parse_slot_rejects_fields_without_key_value_shape():
    with pytest.raises(SystemExit):
        parse_slot("type=cuda,ram")


@pytest.mark.parametrize("argv, message", [
    (["job", "--"], "no command specified"),
    (["job", "--solver=invalid", "--", "true"], "--solver must be"),
])
def test_main_rejects_invalid_invocations(argv, message, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_build_reserve_request_for_explicit_slots_preserves_order_and_capacity():
    slots = [
        ExplicitSlot("cuda:1", ram_mb=4096, compute_units=25),
        ExplicitSlot("cpu", ram_mb=1024),
    ]

    devices = [
        {"name": "cuda:1", "type": "cuda", "available_ram_mb": 4096,
         "available_compute_units": 25},
        {"name": "cpu", "type": "cpu", "available_ram_mb": 1024,
         "available_compute_units": 0},
    ]
    req = build_reserve_request("job", slots, solve_reservations_consolidated, devices)

    assert req.reservations == (
        ReservationRequest(label="job", slots=(
            SlotRequest(name="cuda:1", ram_mb=4096, compute_units=25),
            SlotRequest(name="cpu", ram_mb=1024, compute_units=0),
        )),
    )


def test_build_reserve_request_without_slots_is_valid_marker_reservation():
    req = build_reserve_request("marker", [], solve_reservations_consolidated, [])

    assert req.reservations == (
        ReservationRequest(label="marker", slots=()),
    )


def test_build_reserve_request_delegates_constraints_to_solver():
    calls = []
    def solver(capacity, specifications):
        calls.append((capacity, specifications))
        return [ReservationRequest(specifications[0].label, (
            SlotRequest(capacity[0]["name"], specifications[0].requirements[0].ram_mb),
        ))]
    devices = [{"name": "cuda:0"}]
    slots = [DeviceConstraint("cuda", ram_mb=2048)]

    req = build_reserve_request("job", slots, solver, devices)

    assert calls == [(devices, [ReservationSpec("job", tuple(slots))])]
    assert req.reservations[0].label == "job"
    assert req.reservations[0].slots[0] == SlotRequest(name="cuda:0", ram_mb=2048)


def test_build_reserve_request_requires_status_for_constraints():
    with pytest.raises(ValueError, match="device status"):
        build_reserve_request("job", [DeviceConstraint("cuda")], solve_reservations_consolidated)


@pytest.mark.parametrize("outcome", [
    NoPlacement("no room"), [],
    [ReservationRequest("job", ()), ReservationRequest("extra", ())],
    [ReservationRequest("wrong-label", ())], (),
    [ReservationRequest("job", (DeviceConstraint("cpu"),))],
])
def test_cli_rejects_no_placement_and_invalid_single_reservation_results(outcome):
    with pytest.raises(ValueError):
        build_reserve_request("job", [DeviceConstraint("cpu")], lambda *_: outcome, [])


def test_build_exec_env_sets_reservation_metadata_without_mutating_base_env():
    base = {"KEEP": "yes"}
    granted_slots = [
        {"device_name": "cpu:batch,worker", "cuda_ordinal": -1},
        {"device_name": "cuda:2", "cuda_ordinal": 2},
        {"device_name": "cuda:2", "cuda_ordinal": 2},
        {"device_name": "metal:0", "cuda_ordinal": -1},
        {"device_name": "cuda:3", "cuda_ordinal": 3},
    ]

    env = build_exec_env(base, granted_slots, "abc123", 9)

    assert base == {"KEEP": "yes"}
    assert env["KEEP"] == "yes"
    assert env["RESERVED_ID"] == "abc123"
    assert env["RESERVED_FD"] == "9"
    assert env["RESERVED_DEVICE"] == "cuda:2"
    assert env["RESERVED_DEVICES"] == "cpu:batch,worker,cuda:2,cuda:2,metal:0,cuda:3"
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert json.loads(env["RESERVED_GRANTED_SLOTS"]) == granted_slots


def test_build_exec_env_without_cuda_slots_does_not_add_cuda_visible_devices():
    env = build_exec_env(
        {"RESERVED_FD": "9"},
        [{"device_name": "metal:0", "cuda_ordinal": -1}],
        "rid",
        None,
    )

    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "RESERVED_FD" not in env
    assert env["RESERVED_DEVICE"] == "metal:0"
