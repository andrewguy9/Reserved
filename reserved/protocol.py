"""
protocol.py — Wire serialization for the reserved daemon protocol.

Handles only explicit requests and responses. Constraint types (DeviceConstraint,
ExplicitSlot from the SDK) never appear here.

Wire format: single-line JSON terminated by '\\n' over a Unix domain socket.
"""

from __future__ import annotations

import json
from typing import Any

from .core import (
    ActiveReservation,
    DeviceStatus,
    GrantFailure,
    GrantSuccess,
    GrantedReservation,
    ReservationRequest,
    ReservationStatus,
    ReserveRequest,
    SlotRequest,
    StatusResult,
)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def encode(obj: Any) -> bytes:
    """Serialize a dict to a single-line JSON bytes object terminated by '\\n'."""
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode()


def decode(line: bytes | str) -> Any:
    """Deserialize one JSON line (may or may not include trailing newline)."""
    if isinstance(line, bytes):
        line = line.decode()
    return json.loads(line.strip())


# ---------------------------------------------------------------------------
# Request deserialization (daemon-side: parse incoming wire messages)
# ---------------------------------------------------------------------------

def parse_request(line: bytes | str) -> dict:
    """Parse a raw wire line into a dict. Raises ValueError on bad JSON."""
    try:
        obj = decode(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("request must be a JSON object")
    return obj


def wire_to_reserve_request(obj: dict) -> ReserveRequest:
    """Convert a parsed wire dict to a ReserveRequest."""
    raw_reservations = obj.get("reservations", [])
    reservations = []
    for r in raw_reservations:
        slots = []
        for s in r.get("slots", []):
            slots.append(SlotRequest(
                name=s["name"],
                ram_mb=int(s.get("ram_mb", 0)),
                compute_units=int(s.get("compute_units", 0)),
            ))
        reservations.append(ReservationRequest(
            label=r["label"],
            slots=tuple(slots),
        ))
    return ReserveRequest(reservations=tuple(reservations))


# ---------------------------------------------------------------------------
# Request serialization (SDK-side: build wire messages to send)
# ---------------------------------------------------------------------------

def reserve_request_to_wire(request: ReserveRequest) -> bytes:
    """Serialize a ReserveRequest to wire bytes."""
    obj = {
        "op": "reserve",
        "reservations": [
            {
                "label": r.label,
                "slots": [
                    {
                        "name": s.name,
                        "ram_mb": s.ram_mb,
                        "compute_units": s.compute_units,
                    }
                    for s in r.slots
                ],
            }
            for r in request.reservations
        ],
    }
    return encode(obj)


def status_request_to_wire() -> bytes:
    """Serialize a status request to wire bytes."""
    return encode({"op": "status"})


# ---------------------------------------------------------------------------
# Response serialization (daemon-side: build wire responses)
# ---------------------------------------------------------------------------

def grant_success_to_wire(result: GrantSuccess) -> bytes:
    """Serialize a successful grant result to wire bytes."""
    obj = {
        "ok": True,
        "reservations": [
            {
                "reservation_id": gr.reservation_id,
                "label": gr.label,
                "slots": [
                    {
                        "device_name": gs.device_name,
                        "cuda_ordinal": gs.cuda_ordinal,
                    }
                    for gs in gr.slots
                ],
            }
            for gr in result.reservations
        ],
    }
    return encode(obj)


def grant_failure_to_wire(result: GrantFailure) -> bytes:
    """Serialize a failed grant result to wire bytes."""
    return encode({"ok": False, "error": result.error})


def status_result_to_wire(result: StatusResult) -> bytes:
    """Serialize a StatusResult to wire bytes."""
    obj = {
        "devices": [
            {
                "name": d.name,
                "type": d.type,
                "model": d.model,
                "ram_mb": d.ram_mb,
                "available_ram_mb": d.available_ram_mb,
                "compute_units": d.compute_units,
                "available_compute_units": d.available_compute_units,
            }
            for d in result.devices
        ],
        "reservations": [
            {
                "id": r.id,
                "label": r.label,
                "created_at": r.created_at,
                "slots": [
                    {
                        "device_name": s.device_name,
                        "ram_mb": s.ram_mb,
                        "compute_units": s.compute_units,
                    }
                    for s in r.slots
                ],
            }
            for r in result.reservations
        ],
    }
    return encode(obj)


# ---------------------------------------------------------------------------
# Response deserialization (SDK-side: parse wire responses)
# ---------------------------------------------------------------------------

def parse_response(line: bytes | str) -> dict:
    """Parse a raw wire response line into a dict."""
    try:
        obj = decode(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON response: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("response must be a JSON object")
    return obj
