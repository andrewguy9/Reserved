"""
Invariant-focused tests for the reserve CLI helpers.
"""

import pytest

from reserved.bin.reserve import build_exec_env, build_reserve_request, parse_slot
from reserved.core import ReservationRequest, ReserveRequest, SlotRequest
from reserved.sdk.client import DeviceConstraint, ExplicitSlot
from reserved.sdk.solvers import ConsolidateSolver


def test_parse_slot_ram_and_compute_aliases_are_equivalent():
    short = parse_slot("type=cuda,model=a6000,ram=8192,compute=25")
    long = parse_slot("type=cuda,model=a6000,ram_mb=8192,compute_units=25")

    assert isinstance(short, DeviceConstraint)
    assert isinstance(long, DeviceConstraint)
    assert short == long


def test_parse_slot_name_selects_explicit_slot():
    slot = parse_slot("name=cuda:2,type=cuda,ram=4096,compute=50")

    assert isinstance(slot, ExplicitSlot)
    assert slot.name == "cuda:2"
    assert slot.ram_mb == 4096
    assert slot.compute_units == 50


def test_parse_slot_rejects_fields_without_key_value_shape():
    with pytest.raises(SystemExit):
        parse_slot("type=cuda,ram")


def test_build_reserve_request_for_explicit_slots_preserves_order_and_capacity():
    slots = [
        ExplicitSlot("cuda:1", ram_mb=4096, compute_units=25),
        ExplicitSlot("cpu", ram_mb=1024),
    ]

    req = build_reserve_request("job", slots, ConsolidateSolver())

    assert req.reservations == (
        ReservationRequest(label="job", slots=(
            SlotRequest(name="cuda:1", ram_mb=4096, compute_units=25),
            SlotRequest(name="cpu", ram_mb=1024, compute_units=0),
        )),
    )


def test_build_reserve_request_without_slots_is_valid_marker_reservation():
    req = build_reserve_request("marker", [], ConsolidateSolver())

    assert req.reservations == (
        ReservationRequest(label="marker", slots=()),
    )


def test_build_reserve_request_delegates_constraints_to_solver():
    class RecordingSolver:
        def __init__(self):
            self.calls = []

        def solve(self, specs, devices):
            self.calls.append((specs, devices))
            return ReserveRequest(reservations=(
                ReservationRequest(label=specs[0][0], slots=(
                    SlotRequest(name=devices[0]["name"], ram_mb=specs[0][1][0].ram_mb),
                )),
            ))

    solver = RecordingSolver()
    devices = [{"name": "cuda:0"}]
    slots = [DeviceConstraint("cuda", ram_mb=2048)]

    req = build_reserve_request("job", slots, solver, devices)

    assert solver.calls == [([("job", slots)], devices)]
    assert req.reservations[0].slots[0] == SlotRequest(name="cuda:0", ram_mb=2048)


def test_build_reserve_request_requires_status_for_constraints():
    with pytest.raises(ValueError, match="device status"):
        build_reserve_request("job", [DeviceConstraint("cuda")], ConsolidateSolver())


def test_build_exec_env_sets_reservation_metadata_without_mutating_base_env():
    base = {"KEEP": "yes"}
    granted_slots = [
        {"device_name": "cpu", "cuda_ordinal": -1},
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
    assert env["RESERVED_DEVICES"] == "cpu,cuda:2,cuda:2,metal:0,cuda:3"
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_build_exec_env_without_cuda_slots_does_not_add_cuda_visible_devices():
    env = build_exec_env(
        {},
        [{"device_name": "metal:0", "cuda_ordinal": -1}],
        "rid",
        None,
    )

    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "RESERVED_FD" not in env
    assert env["RESERVED_DEVICE"] == "metal:0"
