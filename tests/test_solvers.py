"""
Tests for reserved/sdk/solvers.py — constraint solvers.
"""

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
    def _solve(self, specs):
        return ConsolidateSolver().solve(specs, DEVICES)

    def test_single_cuda_constraint(self):
        req = self._solve([("train", [DeviceConstraint("cuda")])])
        assert len(req.reservations) == 1
        assert req.reservations[0].slots[0].name.startswith("cuda:")

    def test_consolidate_picks_most_full_device(self):
        """ConsolidateSolver should prefer cuda:1 (less available) over cuda:0."""
        req = self._solve([("train", [DeviceConstraint("cuda", ram_mb=4096)])])
        # cuda:1 has 8192 available; cuda:0 has 24576; cuda:2 has 49152.
        # Consolidate picks least available RAM that still fits → cuda:1 (8192).
        assert req.reservations[0].slots[0].name == "cuda:1"

    def test_type_filter_cpu(self):
        req = self._solve([("job", [DeviceConstraint("cpu")])])
        assert req.reservations[0].slots[0].name == "cpu"

    def test_model_filter(self):
        req = self._solve([("job", [DeviceConstraint("cuda", model="a6000")])])
        assert req.reservations[0].slots[0].name == "cuda:2"

    def test_ram_filter_excludes_small(self):
        # Request more than cuda:1 has available (8192 MB).
        req = self._solve([("job", [DeviceConstraint("cuda", ram_mb=10000)])])
        # cuda:1 is excluded; should pick cuda:0 (24576) or cuda:2 (49152).
        # Consolidate picks least available that still fits → cuda:0 (24576).
        chosen = req.reservations[0].slots[0].name
        assert chosen in ("cuda:0", "cuda:2")
        assert chosen != "cuda:1"

    def test_explicit_slot_passthrough(self):
        req = self._solve([("job", [ExplicitSlot("cuda:2", ram_mb=4096)])])
        assert req.reservations[0].slots[0].name == "cuda:2"
        assert req.reservations[0].slots[0].ram_mb == 4096

    def test_no_matching_device_raises(self):
        with pytest.raises(ValueError, match="no device"):
            self._solve([("job", [DeviceConstraint("cuda", model="nonexistent")])])

    def test_two_slots_consolidated(self):
        """Two constraints should both be assigned, consuming simulated capacity."""
        req = self._solve([("job", [
            DeviceConstraint("cuda", ram_mb=4096),
            DeviceConstraint("cuda", ram_mb=4096),
        ])])
        slots = req.reservations[0].slots
        assert len(slots) == 2

    def test_multi_reservation_separate_labels(self):
        req = self._solve([
            ("r1", [DeviceConstraint("cuda")]),
            ("r2", [DeviceConstraint("cuda")]),
        ])
        assert len(req.reservations) == 2
        assert req.reservations[0].label == "r1"
        assert req.reservations[1].label == "r2"

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
            ConsolidateSolver().solve([
                ("r1", [DeviceConstraint("cuda", ram_mb=5000)]),
                ("r2", [DeviceConstraint("cuda", ram_mb=5000)]),
            ], devices)


class TestSpreadSolver:
    def _solve(self, specs):
        return SpreadSolver().solve(specs, DEVICES)

    def test_spread_picks_most_empty_device(self):
        """SpreadSolver should prefer cuda:2 (most available RAM) over cuda:0."""
        req = self._solve([("train", [DeviceConstraint("cuda", ram_mb=4096)])])
        # cuda:2 has 49152 available — most empty.
        assert req.reservations[0].slots[0].name == "cuda:2"

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
        req = SpreadSolver().solve([("job", [
            DeviceConstraint("cuda", ram_mb=6000),
            DeviceConstraint("cuda", ram_mb=6000),
        ])], devices_tight)
        slots = req.reservations[0].slots
        assert len(slots) == 2
        # Both devices have equal available RAM before solving; after taking
        # 6000 from cuda:0 (or whichever is picked first), the second must
        # go to the other device because cuda:0 only has 2192 left.
        names = [s.name for s in slots]
        assert names[0] != names[1]

    def test_explicit_slot_passthrough(self):
        req = self._solve([("job", [ExplicitSlot("cuda:1", ram_mb=2048)])])
        assert req.reservations[0].slots[0].name == "cuda:1"

    def test_no_matching_device_raises(self):
        with pytest.raises(ValueError, match="no device"):
            self._solve([("job", [DeviceConstraint("cuda", ram_mb=999999)])])

    def test_type_none_matches_any(self):
        """DeviceConstraint with type=None should match any device type."""
        req = self._solve([("job", [DeviceConstraint(None)])])
        assert req.reservations[0].slots[0].name is not None
