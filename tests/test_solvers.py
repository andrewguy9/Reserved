"""
Tests for reserved/sdk/solvers.py — constraint solvers.
"""

from copy import deepcopy

import pytest

from reserved.sdk.client import DeviceConstraint, ExplicitSlot
from reserved.sdk.solvers import ConsolidateSolver, SpreadSolver
from reserved.core import SlotRequest


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


class TestConsolidateSolver:
    def _solve(self, requirements):
        return ConsolidateSolver().solve(DEVICES, requirements)

    def test_single_cuda_constraint(self):
        req = self._solve([[DeviceConstraint("cuda")]])
        assert len(req) == 1
        assert req[0][0].name.startswith("cuda:")

    def test_consolidate_picks_most_full_device(self):
        """ConsolidateSolver should prefer cuda:1 (less available) over cuda:0."""
        req = self._solve([[DeviceConstraint("cuda", ram_mb=4096)]])
        # cuda:1 has 8192 available; cuda:0 has 24576; cuda:2 has 49152.
        # Consolidate picks least available RAM that still fits → cuda:1 (8192).
        assert req[0][0].name == "cuda:1"

    def test_type_filter_cpu(self):
        req = self._solve([[DeviceConstraint("cpu")]])
        assert req[0][0].name == "cpu"

    def test_model_filter(self):
        req = self._solve([[DeviceConstraint("cuda", model="a6000")]])
        assert req[0][0].name == "cuda:2"

    def test_ram_filter_excludes_small(self):
        # Request more than cuda:1 has available (8192 MB).
        req = self._solve([[DeviceConstraint("cuda", ram_mb=10000)]])
        # cuda:1 is excluded; should pick cuda:0 (24576) or cuda:2 (49152).
        # Consolidate picks least available that still fits → cuda:0 (24576).
        chosen = req[0][0].name
        assert chosen in ("cuda:0", "cuda:2")
        assert chosen != "cuda:1"

    def test_explicit_slot_passthrough(self):
        req = self._solve([[ExplicitSlot("cuda:2", ram_mb=4096)]])
        assert req[0][0].name == "cuda:2"
        assert req[0][0].ram_mb == 4096

    def test_no_matching_device_raises(self):
        with pytest.raises(ValueError, match="no device"):
            self._solve([[DeviceConstraint("cuda", model="nonexistent")]])

    def test_two_slots_consolidated(self):
        """Two constraints should both be assigned, consuming simulated capacity."""
        req = self._solve([[
            DeviceConstraint("cuda", ram_mb=4096),
            DeviceConstraint("cuda", ram_mb=4096),
        ]])
        slots = req[0]
        assert len(slots) == 2

    def test_multiple_requirement_groups_preserve_order(self):
        req = self._solve([
            [DeviceConstraint("cuda")],
            [DeviceConstraint("cpu")],
        ])
        assert len(req) == 2
        assert req[0][0].name.startswith("cuda:")
        assert req[1][0].name == "cpu"

    def test_ram_subtracted_across_reservations(self):
        """Simulated availability should decrease between reservation solve steps."""
        # Use a small DEVICES list to force re-use detection.
        devices = [
            {"name": "cuda:0", "type": "cuda", "model": None,
             "ram_mb": 8192, "available_ram_mb": 8192,
             "compute_units": 100, "available_compute_units": 100},
        ]
        # Request two reservations each wanting 5000 MB — only one can fit.
        with pytest.raises(ValueError, match="no device"):
            ConsolidateSolver().solve(devices, [
                [DeviceConstraint("cuda", ram_mb=5000)],
                [DeviceConstraint("cuda", ram_mb=5000)],
            ])


class TestSpreadSolver:
    def _solve(self, requirements):
        return SpreadSolver().solve(DEVICES, requirements)

    def test_spread_picks_most_empty_device(self):
        """SpreadSolver should prefer cuda:2 (most available RAM) over cuda:0."""
        req = self._solve([[DeviceConstraint("cuda", ram_mb=4096)]])
        # cuda:2 has 49152 available — most empty.
        assert req[0][0].name == "cuda:2"

    def test_spread_distributes_two_slots(self):
        """SpreadSolver should distribute onto different devices when capacity forces it."""
        # Use a tight device list so the second slot is forced onto a different device.
        devices_tight = [
            {
                "name": "cuda:0", "type": "cuda", "model": None,
                "ram_mb": 8192, "available_ram_mb": 8192,
                "compute_units": 100, "available_compute_units": 100,
            },
            {
                "name": "cuda:1", "type": "cuda", "model": None,
                "ram_mb": 8192, "available_ram_mb": 8192,
                "compute_units": 100, "available_compute_units": 100,
            },
        ]
        # Request two slots, each needing 6000 MB — only one fits per device.
        req = SpreadSolver().solve(devices_tight, [[
            DeviceConstraint("cuda", ram_mb=6000),
            DeviceConstraint("cuda", ram_mb=6000),
        ]])
        slots = req[0]
        assert len(slots) == 2
        # Both devices have equal available RAM before solving; after taking
        # 6000 from cuda:0 (or whichever is picked first), the second must
        # go to the other device because cuda:0 only has 2192 left.
        names = [s.name for s in slots]
        assert names[0] != names[1]

    def test_explicit_slot_passthrough(self):
        req = self._solve([[ExplicitSlot("cuda:1", ram_mb=2048)]])
        assert req[0][0].name == "cuda:1"

    def test_no_matching_device_raises(self):
        with pytest.raises(ValueError, match="no device"):
            self._solve([[DeviceConstraint("cuda", ram_mb=999999)]])

    def test_type_none_matches_any(self):
        """DeviceConstraint with type=None should match any device type."""
        req = self._solve([[DeviceConstraint(None)]])
        assert req[0][0].name is not None


@pytest.mark.parametrize("solver", [ConsolidateSolver(), SpreadSolver()])
def test_selected_slots_are_repeatable_and_preserve_inputs_on_success_and_failure(solver):
    capacity = deepcopy(DEVICES)
    worker = [DeviceConstraint("cpu", ram_mb=1024)]
    requirements = [worker, worker, []]
    original = deepcopy((capacity, requirements))
    slot_groups = solver.solve(capacity, requirements)
    assert slot_groups == ((SlotRequest("cpu", 1024, 0),), (SlotRequest("cpu", 1024, 0),), ())
    assert solver.solve(capacity, requirements) == slot_groups
    assert (capacity, requirements) == original
    requirements.append([DeviceConstraint("cuda", ram_mb=1000000)])
    failed_inputs = deepcopy((capacity, requirements))
    with pytest.raises(ValueError):
        solver.solve(capacity, requirements)
    assert (capacity, requirements) == failed_inputs
