"""
Tests for reserved/sdk/solvers.py — constraint solvers.
"""

import pytest

from reserved.sdk.client import DeviceConstraint
from reserved.sdk.solvers import ConsolidateSolver, SpreadSolver


DEVICES = [
    {
        "name": "cuda:GPU-00000000-0000-0000-0000-000000000000",
        "type": "cuda",
        "model": "rtx3090",
        "ram_mb": 24576,
        "available_ram_mb": 24576,
        "compute_units": 100,
        "available_compute_units": 100,
    },
    {
        "name": "cuda:GPU-00000000-0000-0000-0000-000000000001",
        "type": "cuda",
        "model": "rtx3090",
        "ram_mb": 24576,
        "available_ram_mb": 8192,  # already partially used
        "compute_units": 100,
        "available_compute_units": 60,
    },
    {
        "name": "cuda:GPU-00000000-0000-0000-0000-000000000002",
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
        assert req.reservations[0].allocations[0].name.startswith("cuda:")

    def test_consolidate_picks_most_full_device(self):
        """ConsolidateSolver should prefer cuda:GPU-00000000-0000-0000-0000-000000000001 (less available) over cuda:GPU-00000000-0000-0000-0000-000000000000."""
        req = self._solve([("train", [DeviceConstraint("cuda", ram_mb=4096)])])
        # cuda:GPU-00000000-0000-0000-0000-000000000001 has 8192 available; cuda:GPU-00000000-0000-0000-0000-000000000000 has 24576; cuda:GPU-00000000-0000-0000-0000-000000000002 has 49152.
        # Consolidate picks least available RAM that still fits → cuda:GPU-00000000-0000-0000-0000-000000000001 (8192).
        assert req.reservations[0].allocations[0].name == "cuda:GPU-00000000-0000-0000-0000-000000000001"

    def test_type_filter_cpu(self):
        req = self._solve([("job", [DeviceConstraint("cpu")])])
        assert req.reservations[0].allocations[0].name == "cpu"

    def test_ram_filter_excludes_small(self):
        # Request more than cuda:GPU-00000000-0000-0000-0000-000000000001 has available (8192 MB).
        req = self._solve([("job", [DeviceConstraint("cuda", ram_mb=10000)])])
        # cuda:GPU-00000000-0000-0000-0000-000000000001 is excluded; should pick cuda:GPU-00000000-0000-0000-0000-000000000000 (24576) or cuda:GPU-00000000-0000-0000-0000-000000000002 (49152).
        # Consolidate picks least available that still fits → cuda:GPU-00000000-0000-0000-0000-000000000000 (24576).
        chosen = req.reservations[0].allocations[0].name
        assert chosen in ("cuda:GPU-00000000-0000-0000-0000-000000000000", "cuda:GPU-00000000-0000-0000-0000-000000000002")
        assert chosen != "cuda:GPU-00000000-0000-0000-0000-000000000001"

    def test_name_pins_the_device(self):
        name = "cuda:GPU-00000000-0000-0000-0000-000000000002"
        req = self._solve([("job", [DeviceConstraint(name=name, ram_mb=4096)])])
        assert req.reservations[0].allocations[0].name == name
        assert req.reservations[0].allocations[0].ram_mb == 4096

    def test_name_with_mismatched_type_raises(self):
        name = "cuda:GPU-00000000-0000-0000-0000-000000000002"
        with pytest.raises(ValueError, match="no device"):
            self._solve([("job", [DeviceConstraint("cpu", name=name)])])

    def test_name_with_insufficient_capacity_raises(self):
        name = "cuda:GPU-00000000-0000-0000-0000-000000000001"  # 8192 MB available
        with pytest.raises(ValueError, match="no device"):
            self._solve([("job", [DeviceConstraint(name=name, ram_mb=8193)])])

    def test_repeated_name_claims_accumulate_within_a_request(self):
        name = "cuda:GPU-00000000-0000-0000-0000-000000000001"  # 8192 MB available
        req = self._solve([("job", [
            DeviceConstraint(name=name, ram_mb=4096), DeviceConstraint(name=name, ram_mb=4096),
        ])])
        assert [a.name for a in req.reservations[0].allocations] == [name, name]
        with pytest.raises(ValueError, match="no device"):
            self._solve([("job", [
                DeviceConstraint(name=name, ram_mb=4096), DeviceConstraint(name=name, ram_mb=4097),
            ])])

    def test_no_matching_device_raises(self):
        with pytest.raises(ValueError, match="no device"):
            self._solve([("job", [DeviceConstraint("tpu")])])

    def test_two_allocations_consolidated(self):
        """Two constraints should both be assigned, consuming simulated capacity."""
        req = self._solve([("job", [
            DeviceConstraint("cuda", ram_mb=4096),
            DeviceConstraint("cuda", ram_mb=4096),
        ])])
        allocations = req.reservations[0].allocations
        assert len(allocations) == 2

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
            {"name": "cuda:GPU-00000000-0000-0000-0000-000000000000", "type": "cuda", "model": None,
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
        """SpreadSolver should prefer cuda:GPU-00000000-0000-0000-0000-000000000002 (most available RAM) over cuda:GPU-00000000-0000-0000-0000-000000000000."""
        req = self._solve([("train", [DeviceConstraint("cuda", ram_mb=4096)])])
        # cuda:GPU-00000000-0000-0000-0000-000000000002 has 49152 available — most empty.
        assert req.reservations[0].allocations[0].name == "cuda:GPU-00000000-0000-0000-0000-000000000002"

    def test_spread_distributes_two_allocations(self):
        """SpreadSolver should distribute onto different devices when capacity forces it."""
        # Use a tight device list so the second allocation is forced onto a different device.
        devices_tight = [
            {
                "name": "cuda:GPU-00000000-0000-0000-0000-000000000000", "type": "cuda", "model": None,
                "ram_mb": 8192, "available_ram_mb": 8192,
                "compute_units": 100, "available_compute_units": 100,
            },
            {
                "name": "cuda:GPU-00000000-0000-0000-0000-000000000001", "type": "cuda", "model": None,
                "ram_mb": 8192, "available_ram_mb": 8192,
                "compute_units": 100, "available_compute_units": 100,
            },
        ]
        # Request two allocations, each needing 6000 MB — only one fits per device.
        req = SpreadSolver().solve([("job", [
            DeviceConstraint("cuda", ram_mb=6000),
            DeviceConstraint("cuda", ram_mb=6000),
        ])], devices_tight)
        allocations = req.reservations[0].allocations
        assert len(allocations) == 2
        # Both devices have equal available RAM before solving; after taking
        # 6000 from cuda:GPU-00000000-0000-0000-0000-000000000000 (or whichever is picked first), the second must
        # go to the other device because cuda:GPU-00000000-0000-0000-0000-000000000000 only has 2192 left.
        names = [s.name for s in allocations]
        assert names[0] != names[1]

    def test_name_pins_the_device(self):
        name = "cuda:GPU-00000000-0000-0000-0000-000000000001"
        req = self._solve([("job", [DeviceConstraint(name=name, ram_mb=2048)])])
        assert req.reservations[0].allocations[0].name == name

    def test_no_matching_device_raises(self):
        with pytest.raises(ValueError, match="no device"):
            self._solve([("job", [DeviceConstraint("cuda", ram_mb=999999)])])

    def test_type_none_matches_any(self):
        """DeviceConstraint with type=None should match any device type."""
        req = self._solve([("job", [DeviceConstraint(None)])])
        assert req.reservations[0].allocations[0].name is not None
