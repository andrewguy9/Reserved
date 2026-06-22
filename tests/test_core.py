"""
Tests for reserved/core.py — the functional core.

All tests run as plain function calls: no daemon, no sockets, no hardware.
"""

import time
import pytest

from reserved.core import (
    ActiveReservation,
    Device,
    DeviceAvailability,
    GrantFailure,
    GrantSuccess,
    Ledger,
    ReservationRequest,
    ReserveRequest,
    SlotRequest,
    grant,
    load_ledger,
    release,
    status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_CONFIG = {
    "devices": [
        {"name": "cuda:0", "type": "cuda", "model": "rtx3090", "ram_mb": 24576, "compute_units": 100},
        {"name": "cuda:1", "type": "cuda", "model": "rtx3090", "ram_mb": 24576, "compute_units": 100},
        {"name": "cuda:2", "type": "cuda", "model": "a6000",   "ram_mb": 49152, "compute_units": 100},
        {"name": "cpu",    "type": "cpu",                       "ram_mb": 65536, "compute_units": 0},
    ]
}


def make_ledger(config: dict | None = None) -> Ledger:
    return load_ledger(config or SIMPLE_CONFIG)


def simple_request(
    label: str,
    device_name: str,
    ram_mb: int = 8192,
    compute_units: int = 0,
) -> ReserveRequest:
    return ReserveRequest(reservations=(
        ReservationRequest(
            label=label,
            slots=(SlotRequest(name=device_name, ram_mb=ram_mb, compute_units=compute_units),),
        ),
    ))


# ---------------------------------------------------------------------------
# load_ledger
# ---------------------------------------------------------------------------

class TestLoadLedger:
    def test_loads_all_devices(self):
        ledger = make_ledger()
        assert len(ledger.availability) == 4

    def test_initial_availability_equals_total(self):
        ledger = make_ledger()
        for da in ledger.availability:
            assert da.available_ram_mb == da.device.ram_mb
            assert da.available_compute_units == da.device.compute_units

    def test_no_reservations_initially(self):
        ledger = make_ledger()
        assert ledger.reservations == ()

    def test_device_with_no_model(self):
        cfg = {"devices": [{"name": "cpu", "type": "cpu", "ram_mb": 32768}]}
        ledger = load_ledger(cfg)
        assert ledger.availability[0].device.model is None

    def test_invalid_cuda_name_no_colon(self):
        cfg = {"devices": [{"name": "cuda0", "type": "cuda", "ram_mb": 8192}]}
        with pytest.raises(ValueError, match="cuda:N"):
            load_ledger(cfg)

    def test_invalid_cuda_name_non_integer_ordinal(self):
        cfg = {"devices": [{"name": "cuda:gpu", "type": "cuda", "ram_mb": 8192}]}
        with pytest.raises(ValueError, match="integer"):
            load_ledger(cfg)

    def test_invalid_cuda_name_negative_ordinal(self):
        cfg = {"devices": [{"name": "cuda:-1", "type": "cuda", "ram_mb": 8192}]}
        with pytest.raises(ValueError, match="non-negative"):
            load_ledger(cfg)

    def test_non_cuda_names_not_validated(self):
        cfg = {"devices": [
            {"name": "metal:0", "type": "metal", "ram_mb": 16384},
            {"name": "cpu", "type": "cpu", "ram_mb": 32768},
        ]}
        ledger = load_ledger(cfg)
        assert len(ledger.availability) == 2

    def test_device_with_no_compute_units(self):
        cfg = {"devices": [{"name": "cpu", "type": "cpu", "ram_mb": 32768}]}
        ledger = load_ledger(cfg)
        assert ledger.availability[0].device.compute_units == 0

    def test_empty_config(self):
        ledger = load_ledger({})
        assert ledger.availability == ()


METAL_CONFIG = {
    "devices": [
        {"name": "metal:0", "type": "metal", "model": "m3-max", "ram_mb": 36864, "compute_units": 100},
        {"name": "cpu",     "type": "cpu",                       "ram_mb": 65536, "compute_units": 0},
    ]
}


class TestMetalGrant:
    def test_grant_metal_device(self):
        ledger = load_ledger(METAL_CONFIG)
        req = simple_request("infer", "metal:0", ram_mb=8192)
        new_ledger, result = grant(ledger, req)
        assert result.ok
        assert result.reservations[0].slots[0].device_name == "metal:0"

    def test_metal_cuda_ordinal_is_minus_one(self):
        ledger = load_ledger(METAL_CONFIG)
        req = simple_request("infer", "metal:0", ram_mb=8192)
        _, result = grant(ledger, req)
        assert result.reservations[0].slots[0].cuda_ordinal == -1

    def test_metal_capacity_subtracted(self):
        ledger = load_ledger(METAL_CONFIG)
        req = simple_request("infer", "metal:0", ram_mb=8192)
        new_ledger, result = grant(ledger, req)
        metal_avail = next(
            da for da in new_ledger.availability if da.device.name == "metal:0"
        )
        assert metal_avail.available_ram_mb == 36864 - 8192

    def test_metal_and_cpu_same_request(self):
        ledger = load_ledger(METAL_CONFIG)
        req = ReserveRequest(reservations=(
            ReservationRequest(label="pipeline", slots=(
                SlotRequest(name="metal:0", ram_mb=8192),
                SlotRequest(name="cpu", ram_mb=16384),
            )),
        ))
        new_ledger, result = grant(ledger, req)
        assert result.ok
        assert len(result.reservations[0].slots) == 2

    def test_metal_release(self):
        ledger = load_ledger(METAL_CONFIG)
        req = simple_request("infer", "metal:0", ram_mb=8192)
        new_ledger, result = grant(ledger, req)
        reservation_id = result.reservations[0].reservation_id
        released_ledger = release(new_ledger, reservation_id)
        metal_avail = next(
            da for da in released_ledger.availability if da.device.name == "metal:0"
        )
        assert metal_avail.available_ram_mb == 36864


# ---------------------------------------------------------------------------
# grant — success cases
# ---------------------------------------------------------------------------

class TestGrantSuccess:
    def test_basic_grant(self):
        ledger = make_ledger()
        req = simple_request("train", "cuda:0", ram_mb=8192, compute_units=50)
        new_ledger, result = grant(ledger, req)

        assert isinstance(result, GrantSuccess)
        assert result.ok is True
        assert len(result.reservations) == 1
        granted = result.reservations[0]
        assert granted.label == "train"
        assert len(granted.reservation_id) == 6
        assert len(granted.slots) == 1
        assert granted.slots[0].device_name == "cuda:0"

    def test_ram_subtracted_from_new_ledger(self):
        ledger = make_ledger()
        req = simple_request("train", "cuda:0", ram_mb=8192)
        new_ledger, _ = grant(ledger, req)

        # Find cuda:0 in new ledger.
        avail_map = {da.device.name: da for da in new_ledger.availability}
        assert avail_map["cuda:0"].available_ram_mb == 24576 - 8192

    def test_compute_subtracted_from_new_ledger(self):
        ledger = make_ledger()
        req = simple_request("train", "cuda:0", ram_mb=0, compute_units=40)
        new_ledger, _ = grant(ledger, req)

        avail_map = {da.device.name: da for da in new_ledger.availability}
        assert avail_map["cuda:0"].available_compute_units == 100 - 40

    def test_original_ledger_unchanged(self):
        ledger = make_ledger()
        req = simple_request("train", "cuda:0", ram_mb=8192)
        original_ram = ledger.availability[0].available_ram_mb
        _, _ = grant(ledger, req)
        assert ledger.availability[0].available_ram_mb == original_ram

    def test_cuda_ordinal_in_response(self):
        ledger = make_ledger()
        req = simple_request("infer", "cuda:2", ram_mb=4096)
        _, result = grant(ledger, req)
        assert result.reservations[0].slots[0].cuda_ordinal == 2

    def test_cpu_cuda_ordinal_minus_one(self):
        ledger = make_ledger()
        req = simple_request("cpu-job", "cpu", ram_mb=1024)
        _, result = grant(ledger, req)
        assert result.reservations[0].slots[0].cuda_ordinal == -1

    def test_reservation_id_is_unique(self):
        ledger = make_ledger()
        ids = set()
        for i in range(20):
            req = simple_request(f"job-{i}", "cuda:0", ram_mb=0)
            _, result = grant(ledger, req)
            ids.add(result.reservations[0].reservation_id)
        # Very unlikely to collide for 20 calls.
        assert len(ids) >= 18

    def test_grant_with_zero_resources(self):
        """Granting 0 MB / 0 compute is valid."""
        ledger = make_ledger()
        req = simple_request("marker", "cuda:0", ram_mb=0, compute_units=0)
        new_ledger, result = grant(ledger, req)
        assert result.ok
        avail_map = {da.device.name: da for da in new_ledger.availability}
        assert avail_map["cuda:0"].available_ram_mb == 24576  # unchanged

    def test_multiple_grants_accumulate(self):
        ledger = make_ledger()
        req1 = simple_request("job1", "cuda:0", ram_mb=8192)
        req2 = simple_request("job2", "cuda:0", ram_mb=4096)
        ledger, _ = grant(ledger, req1)
        ledger, _ = grant(ledger, req2)
        avail_map = {da.device.name: da for da in ledger.availability}
        assert avail_map["cuda:0"].available_ram_mb == 24576 - 8192 - 4096

    def test_reservation_added_to_ledger(self):
        ledger = make_ledger()
        req = simple_request("train", "cuda:0", ram_mb=8192)
        new_ledger, _ = grant(ledger, req)
        assert len(new_ledger.reservations) == 1

    def test_created_at_is_recent(self):
        before = time.time()
        ledger = make_ledger()
        req = simple_request("train", "cuda:0", ram_mb=0)
        _, result = grant(ledger, req)
        after = time.time()
        ts = result.reservations[0].created_at
        assert before <= ts <= after


# ---------------------------------------------------------------------------
# grant — failure cases
# ---------------------------------------------------------------------------

class TestGrantFailure:
    def test_insufficient_ram(self):
        ledger = make_ledger()
        req = simple_request("toobig", "cuda:0", ram_mb=99999)
        new_ledger, result = grant(ledger, req)

        assert isinstance(result, GrantFailure)
        assert result.ok is False
        assert "insufficient" in result.error.lower() or "ram" in result.error.lower()

    def test_insufficient_compute(self):
        ledger = make_ledger()
        req = simple_request("toobig", "cuda:0", ram_mb=0, compute_units=999)
        _, result = grant(ledger, req)
        assert result.ok is False

    def test_unknown_device(self):
        ledger = make_ledger()
        req = simple_request("job", "cuda:99", ram_mb=0)
        _, result = grant(ledger, req)
        assert result.ok is False
        assert "cuda:99" in result.error

    def test_original_ledger_unchanged_on_failure(self):
        ledger = make_ledger()
        req = simple_request("toobig", "cuda:0", ram_mb=99999)
        original_avail = ledger.availability
        new_ledger, result = grant(ledger, req)
        # new_ledger should be the same object on failure.
        assert new_ledger is ledger
        assert new_ledger.availability == original_avail

    def test_no_reservation_added_on_failure(self):
        ledger = make_ledger()
        req = simple_request("toobig", "cuda:0", ram_mb=99999)
        new_ledger, _ = grant(ledger, req)
        assert len(new_ledger.reservations) == 0

    def test_exact_capacity_succeeds(self):
        ledger = make_ledger()
        req = simple_request("full", "cuda:0", ram_mb=24576)
        _, result = grant(ledger, req)
        assert result.ok

    def test_one_over_capacity_fails(self):
        ledger = make_ledger()
        req = simple_request("over", "cuda:0", ram_mb=24577)
        _, result = grant(ledger, req)
        assert not result.ok


# ---------------------------------------------------------------------------
# grant — atomic multi-reservation
# ---------------------------------------------------------------------------

class TestGrantAtomic:
    def _multi_request(self) -> ReserveRequest:
        return ReserveRequest(reservations=(
            ReservationRequest(
                label="checkpoint",
                slots=(SlotRequest(name="cuda:2", ram_mb=8192, compute_units=0),),
            ),
            ReservationRequest(
                label="lora",
                slots=(SlotRequest(name="cuda:2", ram_mb=2048, compute_units=0),),
            ),
        ))

    def test_multi_reservation_success(self):
        ledger = make_ledger()
        req = self._multi_request()
        new_ledger, result = grant(ledger, req)

        assert result.ok
        assert len(result.reservations) == 2
        assert result.reservations[0].label == "checkpoint"
        assert result.reservations[1].label == "lora"

    def test_multi_reservation_both_subtracted(self):
        ledger = make_ledger()
        req = self._multi_request()
        new_ledger, _ = grant(ledger, req)
        avail_map = {da.device.name: da for da in new_ledger.availability}
        assert avail_map["cuda:2"].available_ram_mb == 49152 - 8192 - 2048

    def test_multi_reservation_two_reservations_added(self):
        ledger = make_ledger()
        req = self._multi_request()
        new_ledger, _ = grant(ledger, req)
        assert len(new_ledger.reservations) == 2

    def test_multi_reservation_atomic_failure_rolls_back(self):
        """If second reservation fails, first should not be committed."""
        ledger = make_ledger()
        req = ReserveRequest(reservations=(
            ReservationRequest(
                label="ok",
                slots=(SlotRequest(name="cuda:0", ram_mb=8192),),
            ),
            ReservationRequest(
                label="toobig",
                slots=(SlotRequest(name="cuda:0", ram_mb=99999),),
            ),
        ))
        new_ledger, result = grant(ledger, req)

        assert not result.ok
        assert new_ledger is ledger  # original returned unchanged
        assert len(new_ledger.reservations) == 0

    def test_multi_reservation_different_devices(self):
        ledger = make_ledger()
        req = ReserveRequest(reservations=(
            ReservationRequest(
                label="r1",
                slots=(SlotRequest(name="cuda:0", ram_mb=8192),),
            ),
            ReservationRequest(
                label="r2",
                slots=(SlotRequest(name="cuda:1", ram_mb=4096),),
            ),
        ))
        new_ledger, result = grant(ledger, req)
        assert result.ok
        avail_map = {da.device.name: da for da in new_ledger.availability}
        assert avail_map["cuda:0"].available_ram_mb == 24576 - 8192
        assert avail_map["cuda:1"].available_ram_mb == 24576 - 4096

    def test_multi_slot_single_reservation(self):
        """Single reservation with multiple slots on different devices."""
        ledger = make_ledger()
        req = ReserveRequest(reservations=(
            ReservationRequest(
                label="distributed",
                slots=(
                    SlotRequest(name="cuda:0", ram_mb=4096),
                    SlotRequest(name="cuda:1", ram_mb=4096),
                ),
            ),
        ))
        new_ledger, result = grant(ledger, req)
        assert result.ok
        assert len(result.reservations[0].slots) == 2
        # cuda_ordinal should be the physical ordinal from the device name.
        assert result.reservations[0].slots[0].cuda_ordinal == 0
        assert result.reservations[0].slots[1].cuda_ordinal == 1


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------

class TestRelease:
    def test_release_restores_ram(self):
        ledger = make_ledger()
        req = simple_request("train", "cuda:0", ram_mb=8192)
        ledger, result = grant(ledger, req)
        reservation_id = result.reservations[0].reservation_id

        ledger = release(ledger, reservation_id)
        avail_map = {da.device.name: da for da in ledger.availability}
        assert avail_map["cuda:0"].available_ram_mb == 24576

    def test_release_restores_compute(self):
        ledger = make_ledger()
        req = simple_request("train", "cuda:0", ram_mb=0, compute_units=50)
        ledger, result = grant(ledger, req)
        rid = result.reservations[0].reservation_id

        ledger = release(ledger, rid)
        avail_map = {da.device.name: da for da in ledger.availability}
        assert avail_map["cuda:0"].available_compute_units == 100

    def test_release_removes_reservation(self):
        ledger = make_ledger()
        req = simple_request("train", "cuda:0", ram_mb=8192)
        ledger, result = grant(ledger, req)
        rid = result.reservations[0].reservation_id

        ledger = release(ledger, rid)
        assert len(ledger.reservations) == 0

    def test_release_unknown_id_unchanged(self):
        ledger = make_ledger()
        new_ledger = release(ledger, "nonexistent")
        assert new_ledger is ledger

    def test_release_only_target(self):
        ledger = make_ledger()
        req1 = simple_request("job1", "cuda:0", ram_mb=4096)
        req2 = simple_request("job2", "cuda:1", ram_mb=2048)
        ledger, r1 = grant(ledger, req1)
        ledger, r2 = grant(ledger, req2)
        rid1 = r1.reservations[0].reservation_id

        ledger = release(ledger, rid1)
        # job2 should still be there.
        assert len(ledger.reservations) == 1
        avail_map = {da.device.name: da for da in ledger.availability}
        # cuda:0 restored, cuda:1 still claimed.
        assert avail_map["cuda:0"].available_ram_mb == 24576
        assert avail_map["cuda:1"].available_ram_mb == 24576 - 2048

    def test_release_idempotent_after_release(self):
        ledger = make_ledger()
        req = simple_request("train", "cuda:0", ram_mb=8192)
        ledger, result = grant(ledger, req)
        rid = result.reservations[0].reservation_id

        ledger = release(ledger, rid)
        # Second release should be a no-op.
        ledger2 = release(ledger, rid)
        assert ledger2 is ledger


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_all_devices_shown(self):
        ledger = make_ledger()
        result = status(ledger)
        assert len(result.devices) == len(SIMPLE_CONFIG["devices"])

    def test_status_device_fields(self):
        ledger = make_ledger()
        result = status(ledger)
        dev_map = {d.name: d for d in result.devices}
        assert dev_map["cuda:0"].type == "cuda"
        assert dev_map["cuda:0"].model == "rtx3090"
        assert dev_map["cuda:0"].ram_mb == 24576
        assert dev_map["cuda:0"].available_ram_mb == 24576
        assert dev_map["cuda:0"].compute_units == 100
        assert dev_map["cuda:0"].available_compute_units == 100

    def test_status_reflects_grants(self):
        ledger = make_ledger()
        req = simple_request("train", "cuda:0", ram_mb=8192, compute_units=50)
        ledger, _ = grant(ledger, req)
        result = status(ledger)
        dev_map = {d.name: d for d in result.devices}
        assert dev_map["cuda:0"].available_ram_mb == 24576 - 8192
        assert dev_map["cuda:0"].available_compute_units == 100 - 50

    def test_status_lists_reservations(self):
        ledger = make_ledger()
        req = simple_request("train", "cuda:0", ram_mb=8192)
        ledger, result_grant = grant(ledger, req)
        result = status(ledger)

        assert len(result.reservations) == 1
        r = result.reservations[0]
        assert r.id == result_grant.reservations[0].reservation_id
        assert r.label == "train"
        assert len(r.slots) == 1
        assert r.slots[0].device_name == "cuda:0"
        assert r.slots[0].ram_mb == 8192

    def test_status_no_reservations_initially(self):
        ledger = make_ledger()
        result = status(ledger)
        assert result.reservations == ()

    def test_status_after_release_clears_reservation(self):
        ledger = make_ledger()
        req = simple_request("train", "cuda:0", ram_mb=8192)
        ledger, result_grant = grant(ledger, req)
        rid = result_grant.reservations[0].reservation_id
        ledger = release(ledger, rid)

        result = status(ledger)
        assert len(result.reservations) == 0


# ---------------------------------------------------------------------------
# Round-trip: grant → status → release → status
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_full_lifecycle(self):
        ledger = make_ledger()

        # Grant.
        req = simple_request("train", "cuda:2", ram_mb=16384, compute_units=75)
        ledger, grant_result = grant(ledger, req)
        assert grant_result.ok
        rid = grant_result.reservations[0].reservation_id

        # Status shows claimed resources.
        s = status(ledger)
        dev_map = {d.name: d for d in s.devices}
        assert dev_map["cuda:2"].available_ram_mb == 49152 - 16384
        assert dev_map["cuda:2"].available_compute_units == 100 - 75
        assert len(s.reservations) == 1

        # Release.
        ledger = release(ledger, rid)

        # Status shows all resources restored.
        s2 = status(ledger)
        dev_map2 = {d.name: d for d in s2.devices}
        assert dev_map2["cuda:2"].available_ram_mb == 49152
        assert dev_map2["cuda:2"].available_compute_units == 100
        assert len(s2.reservations) == 0
