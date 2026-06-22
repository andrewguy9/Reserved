"""
Tests for reserved/protocol.py — wire serialization.
"""

import json
import pytest

from reserved.core import (
    GrantFailure,
    GrantSuccess,
    GrantedReservation,
    GrantedSlot,
    ReservationRequest,
    ReserveRequest,
    SlotRequest,
    StatusResult,
    DeviceStatus,
    ReservationStatus,
    ActiveReservation,
)
from reserved.protocol import (
    decode,
    encode,
    grant_failure_to_wire,
    grant_success_to_wire,
    parse_request,
    parse_response,
    reserve_request_to_wire,
    status_request_to_wire,
    status_result_to_wire,
    wire_to_reserve_request,
)


class TestEncodeDecodeRoundtrip:
    def test_encode_terminates_with_newline(self):
        data = encode({"op": "status"})
        assert data.endswith(b"\n")

    def test_encode_single_line(self):
        data = encode({"a": 1, "b": 2})
        assert data.count(b"\n") == 1

    def test_decode_bytes(self):
        obj = decode(b'{"op": "status"}\n')
        assert obj == {"op": "status"}

    def test_decode_str(self):
        obj = decode('{"ok": true}')
        assert obj == {"ok": True}

    def test_roundtrip(self):
        original = {"op": "reserve", "reservations": [{"label": "x", "slots": []}]}
        assert decode(encode(original)) == original


class TestReserveRequestWire:
    def _make_request(self) -> ReserveRequest:
        return ReserveRequest(reservations=(
            ReservationRequest(
                label="train",
                slots=(
                    SlotRequest(name="cuda:0", ram_mb=8192, compute_units=50),
                ),
            ),
        ))

    def test_serialize_sets_op(self):
        req = self._make_request()
        data = reserve_request_to_wire(req)
        obj = json.loads(data)
        assert obj["op"] == "reserve"

    def test_serialize_label(self):
        req = self._make_request()
        data = reserve_request_to_wire(req)
        obj = json.loads(data)
        assert obj["reservations"][0]["label"] == "train"

    def test_serialize_slot_fields(self):
        req = self._make_request()
        data = reserve_request_to_wire(req)
        obj = json.loads(data)
        slot = obj["reservations"][0]["slots"][0]
        assert slot["name"] == "cuda:0"
        assert slot["ram_mb"] == 8192
        assert slot["compute_units"] == 50

    def test_deserialize_reserve_request(self):
        wire = b'{"op":"reserve","reservations":[{"label":"lora","slots":[{"name":"cuda:1","ram_mb":2048,"compute_units":0}]}]}\n'
        obj = parse_request(wire)
        req = wire_to_reserve_request(obj)
        assert req.reservations[0].label == "lora"
        assert req.reservations[0].slots[0].name == "cuda:1"
        assert req.reservations[0].slots[0].ram_mb == 2048

    def test_serialize_deserialize_roundtrip(self):
        req = self._make_request()
        wire = reserve_request_to_wire(req)
        obj = parse_request(wire)
        req2 = wire_to_reserve_request(obj)
        assert req2.reservations[0].label == req.reservations[0].label
        assert req2.reservations[0].slots[0].name == req.reservations[0].slots[0].name
        assert req2.reservations[0].slots[0].ram_mb == req.reservations[0].slots[0].ram_mb

    def test_multi_reservation_roundtrip(self):
        req = ReserveRequest(reservations=(
            ReservationRequest(label="a", slots=(SlotRequest(name="cuda:0", ram_mb=4096),)),
            ReservationRequest(label="b", slots=(SlotRequest(name="cuda:1", ram_mb=2048),)),
        ))
        wire = reserve_request_to_wire(req)
        obj = parse_request(wire)
        req2 = wire_to_reserve_request(obj)
        assert len(req2.reservations) == 2
        assert req2.reservations[1].label == "b"


class TestStatusRequestWire:
    def test_status_request_op(self):
        data = status_request_to_wire()
        obj = json.loads(data)
        assert obj["op"] == "status"


class TestGrantSuccessWire:
    def _make_success(self) -> GrantSuccess:
        return GrantSuccess(reservations=(
            GrantedReservation(
                reservation_id="a3f9c2",
                label="train",
                slots=(GrantedSlot(device_name="cuda:0", cuda_ordinal=0),),
                created_at=1748000000.0,
            ),
        ))

    def test_ok_true(self):
        data = grant_success_to_wire(self._make_success())
        obj = json.loads(data)
        assert obj["ok"] is True

    def test_reservation_id_present(self):
        data = grant_success_to_wire(self._make_success())
        obj = json.loads(data)
        assert obj["reservations"][0]["reservation_id"] == "a3f9c2"

    def test_slot_device_name(self):
        data = grant_success_to_wire(self._make_success())
        obj = json.loads(data)
        assert obj["reservations"][0]["slots"][0]["device_name"] == "cuda:0"

    def test_slot_cuda_ordinal(self):
        data = grant_success_to_wire(self._make_success())
        obj = json.loads(data)
        assert obj["reservations"][0]["slots"][0]["cuda_ordinal"] == 0


class TestGrantFailureWire:
    def test_ok_false(self):
        result = GrantFailure(error="insufficient resources")
        data = grant_failure_to_wire(result)
        obj = json.loads(data)
        assert obj["ok"] is False
        assert "insufficient" in obj["error"]


class TestStatusResultWire:
    def _make_status(self) -> StatusResult:
        return StatusResult(
            devices=(
                DeviceStatus(
                    name="cuda:0",
                    type="cuda",
                    model="rtx3090",
                    ram_mb=24576,
                    available_ram_mb=16384,
                    compute_units=100,
                    available_compute_units=50,
                ),
            ),
            reservations=(
                ReservationStatus(
                    id="a3f9c2",
                    label="train",
                    created_at=1748000000.0,
                    slots=(
                        ActiveReservation.ClaimedSlot(
                            device_name="cuda:0",
                            ram_mb=8192,
                            compute_units=50,
                        ),
                    ),
                ),
            ),
        )

    def test_device_fields(self):
        data = status_result_to_wire(self._make_status())
        obj = json.loads(data)
        d = obj["devices"][0]
        assert d["name"] == "cuda:0"
        assert d["available_ram_mb"] == 16384
        assert d["available_compute_units"] == 50

    def test_reservation_fields(self):
        data = status_result_to_wire(self._make_status())
        obj = json.loads(data)
        r = obj["reservations"][0]
        assert r["id"] == "a3f9c2"
        assert r["label"] == "train"
        assert r["slots"][0]["device_name"] == "cuda:0"
        assert r["slots"][0]["ram_mb"] == 8192


class TestParseErrors:
    def test_parse_bad_json_raises(self):
        with pytest.raises(ValueError):
            parse_request(b"not json")

    def test_parse_non_object_raises(self):
        with pytest.raises(ValueError):
            parse_request(b'"just a string"\n')
