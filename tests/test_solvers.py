"""
Tests for reserved/sdk/solvers.py — constraint solvers.
"""

from copy import deepcopy

import pytest

from reserved.sdk.client import DeviceConstraint, ExplicitSlot, ReservationSpec
from reserved.sdk.solvers import (
    NoPlacement, solve_reservations_consolidated, solve_reservations_spread,
)
from reserved.core import ReservationRequest, ReserveRequest, SlotRequest


DEVICES = [
    {
        "name": "cuda:0",
        "type": "cuda",
        "model": "rtx3090",
        "ram_mb": 24576,
        "available_ram_mb": 24576,
        "compute_units": 100,
        "available_compute_units": 100,
    },
    {
        "name": "cuda:1",
        "type": "cuda",
        "model": "rtx3090",
        "ram_mb": 24576,
        "available_ram_mb": 8192,  # already partially used
        "compute_units": 100,
        "available_compute_units": 60,
    },
    {
        "name": "cuda:2",
        "type": "cuda",
        "model": "a6000",
        "ram_mb": 49152,
        "available_ram_mb": 49152,
        "compute_units": 100,
        "available_compute_units": 100,
    },
    {
        "name": "cpu",
        "type": "cpu",
        "model": None,
        "ram_mb": 65536,
        "available_ram_mb": 65536,
        "compute_units": 0,
        "available_compute_units": 0,
    },
]


POLICIES = [solve_reservations_consolidated, solve_reservations_spread]


@pytest.mark.parametrize("solver, requirement, device_name", [
    (solve_reservations_consolidated, DeviceConstraint("cuda", ram_mb=4096), "cuda:1"),
    (solve_reservations_spread, DeviceConstraint("cuda", ram_mb=4096), "cuda:2"),
    (solve_reservations_consolidated, DeviceConstraint("cuda", ram_mb=10000), "cuda:0"),
    (solve_reservations_consolidated, DeviceConstraint("cuda", compute_units=80), "cuda:0"),
    (solve_reservations_consolidated, DeviceConstraint("cuda", model="a6000"), "cuda:2"),
    (solve_reservations_consolidated, DeviceConstraint("cpu"), "cpu"),
    (solve_reservations_spread, DeviceConstraint(None), "cpu"),
])
def test_policy_selects_expected_device(solver, requirement, device_name):
    placement = solver(DEVICES, [ReservationSpec("worker", (requirement,))])
    assert placement == ReserveRequest((ReservationRequest("worker", (
        SlotRequest(device_name, requirement.ram_mb, requirement.compute_units),
    )),))


@pytest.mark.parametrize("solver", POLICIES)
def test_explicit_assignments_keep_names_and_amounts(solver):
    placement = solver(DEVICES, [ReservationSpec("pinned", (
        ExplicitSlot("cuda:1", ram_mb=2048, compute_units=25),
        ExplicitSlot("cuda:2", ram_mb=4096),
    ))])
    assert placement == ReserveRequest((ReservationRequest("pinned", (
        SlotRequest("cuda:1", 2048, 25), SlotRequest("cuda:2", 4096),
    )),))


@pytest.mark.parametrize("solver", POLICIES)
def test_explicit_passthrough_still_leaves_validation_to_daemon(solver):
    assert solver([], [ReservationSpec("pinned", (ExplicitSlot("cuda:99"),))]) == (
        ReserveRequest((ReservationRequest("pinned", (SlotRequest("cuda:99"),)),))
    )


@pytest.mark.parametrize("solver", POLICIES)
@pytest.mark.parametrize("requirement", [
    DeviceConstraint("cuda", model="nonexistent"),
    DeviceConstraint("cuda", ram_mb=999999),
    DeviceConstraint("cuda", compute_units=101),
])
def test_no_matching_device_is_a_returned_outcome(solver, requirement):
    result = solver(DEVICES, [ReservationSpec("worker", (requirement,))])
    assert isinstance(result, NoPlacement)
    assert "no device" in result.reason
    assert "worker" in result.reason


@pytest.mark.parametrize("solver", POLICIES)
def test_capacity_is_shared_across_reservations_and_no_partial_batch_is_returned(solver):
    capacity = [{"name": "cuda:0", "type": "cuda", "model": None,
                 "available_ram_mb": 8192, "available_compute_units": 100}]
    result = solver(capacity, [
        ReservationSpec("first", (DeviceConstraint("cuda", ram_mb=5000),)),
        ReservationSpec("second", (DeviceConstraint("cuda", ram_mb=5000),)),
    ])
    assert isinstance(result, NoPlacement)
    assert "second" in result.reason


@pytest.mark.parametrize("solver", POLICIES)
def test_multiple_devices_within_one_reservation(solver):
    capacity = [
        {"name": name, "type": "cuda", "model": None,
         "available_ram_mb": 8192, "available_compute_units": 100}
        for name in ("cuda:0", "cuda:1")
    ]
    assert solver(capacity, [ReservationSpec("distributed", (
        DeviceConstraint("cuda", ram_mb=6000),
        DeviceConstraint("cuda", ram_mb=6000),
    ))]) == ReserveRequest((ReservationRequest("distributed", (
        SlotRequest("cuda:0", 6000), SlotRequest("cuda:1", 6000),
    )),))


@pytest.mark.parametrize("solver", POLICIES)
def test_empty_batch_and_marker_reservation_are_distinct_successes(solver):
    assert solver([], []) == ReserveRequest(())
    assert solver([], [ReservationSpec("marker", ())]) == (
        ReserveRequest((ReservationRequest("marker", ()),))
    )


@pytest.mark.parametrize("solver", POLICIES)
def test_results_preserve_labels_order_and_inputs_on_success_and_failure(solver):
    capacity = deepcopy(DEVICES)
    worker = (DeviceConstraint("cpu", ram_mb=1024),)
    specs = [ReservationSpec("first", worker), ReservationSpec("second", worker),
             ReservationSpec("marker", ())]
    original = deepcopy((capacity, specs))
    expected = ReserveRequest((
        ReservationRequest("first", (SlotRequest("cpu", 1024),)),
        ReservationRequest("second", (SlotRequest("cpu", 1024),)),
        ReservationRequest("marker", ()),
    ))
    assert solver(capacity, specs) == expected
    assert solver(capacity, specs) == expected
    assert (capacity, specs) == original

    specs.append(ReservationSpec("too-large", (DeviceConstraint("cuda", ram_mb=1000000),)))
    before_failure = deepcopy((capacity, specs))
    assert isinstance(solver(capacity, specs), NoPlacement)
    assert (capacity, specs) == before_failure


@pytest.mark.parametrize("solver", POLICIES)
def test_explicit_assignments_consume_simulated_capacity(solver):
    result = solver(DEVICES, [
        ReservationSpec("pinned", (ExplicitSlot("cpu", ram_mb=65536),)),
        ReservationSpec("extra", (DeviceConstraint("cpu", ram_mb=1),)),
    ])
    assert isinstance(result, NoPlacement)
    assert "extra" in result.reason
