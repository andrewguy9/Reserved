"""
core.py — Pure functional core for the reserved daemon.

All business logic lives here as pure functions operating on immutable frozen
dataclasses. No IO, no sockets, no global state. Every function takes a Ledger
and returns a new Ledger plus a result object.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field, replace


# ---------------------------------------------------------------------------
# Ledger types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Device:
    name: str
    type: str
    model: str | None
    ram_mb: int
    compute_units: int


@dataclass(frozen=True)
class DeviceAvailability:
    device: Device
    available_ram_mb: int
    available_compute_units: int


@dataclass(frozen=True)
class SlotRequest:
    """An explicit slot in a reserve request — names a specific device."""
    name: str          # physical device name, e.g. "cuda:0"
    ram_mb: int = 0
    compute_units: int = 0


@dataclass(frozen=True)
class ReservationRequest:
    """One reservation in a reserve request."""
    label: str
    slots: tuple[SlotRequest, ...]


@dataclass(frozen=True)
class ReserveRequest:
    """A full reserve request — one or more reservations, granted atomically."""
    reservations: tuple[ReservationRequest, ...]


@dataclass(frozen=True)
class GrantedSlot:
    """One granted slot in a successful reservation."""
    device_name: str
    # Physical CUDA ordinal extracted from the device name (e.g. cuda:2 → 2).
    # Set in CUDA_VISIBLE_DEVICES so the child process sees only granted devices.
    # The child addresses them as cuda:0, cuda:1, ... (logical indices within
    # whatever CUDA_VISIBLE_DEVICES exposes). -1 for non-CUDA devices.
    cuda_ordinal: int


@dataclass(frozen=True)
class GrantedReservation:
    reservation_id: str
    label: str
    slots: tuple[GrantedSlot, ...]
    created_at: float


@dataclass(frozen=True)
class ActiveReservation:
    """Tracks a live reservation in the ledger — includes the claimed resources."""

    @dataclass(frozen=True)
    class ClaimedSlot:
        device_name: str
        ram_mb: int
        compute_units: int

    reservation_id: str
    label: str
    slots: tuple[ActiveReservation.ClaimedSlot, ...]
    created_at: float


@dataclass(frozen=True)
class Ledger:
    availability: tuple[DeviceAvailability, ...]
    reservations: tuple[ActiveReservation, ...]


# ---------------------------------------------------------------------------
# Grant / release results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GrantSuccess:
    ok: bool = True
    reservations: tuple[GrantedReservation, ...] = ()


@dataclass(frozen=True)
class GrantFailure:
    ok: bool = False
    error: str = "insufficient resources"


GrantResult = GrantSuccess | GrantFailure


@dataclass(frozen=True)
class DeviceStatus:
    name: str
    type: str
    model: str | None
    ram_mb: int
    available_ram_mb: int
    compute_units: int
    available_compute_units: int


@dataclass(frozen=True)
class ReservationStatus:
    id: str
    label: str
    created_at: float
    slots: tuple[ActiveReservation.ClaimedSlot, ...]


@dataclass(frozen=True)
class StatusResult:
    devices: tuple[DeviceStatus, ...]
    reservations: tuple[ReservationStatus, ...]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def load_ledger(config: dict) -> Ledger:
    """Build an initial Ledger from a config dict (parsed from reserved.json)."""
    devices = []
    for d in config.get("devices", []):
        dev = Device(
            name=d["name"],
            type=d["type"],
            model=d.get("model"),
            ram_mb=d.get("ram_mb", 0),
            compute_units=d.get("compute_units", 0),
        )
        devices.append(DeviceAvailability(
            device=dev,
            available_ram_mb=dev.ram_mb,
            available_compute_units=dev.compute_units,
        ))
    return Ledger(
        availability=tuple(devices),
        reservations=(),
    )


def _availability_index(ledger: Ledger) -> dict[str, int]:
    """Return a name→index map for the availability tuple."""
    return {da.device.name: i for i, da in enumerate(ledger.availability)}


def grant(ledger: Ledger, request: ReserveRequest) -> tuple[Ledger, GrantResult]:
    """
    Attempt to grant all reservations in *request* atomically.

    Returns (new_ledger, GrantSuccess) if all slots can be satisfied, or
    (original_ledger, GrantFailure) if any slot cannot be satisfied.
    """
    idx = _availability_index(ledger)
    avail = list(ledger.availability)  # mutable working copy

    # Collect all slots across all reservations for atomic checking.
    # We'll simulate the subtraction in avail[] and roll back on failure.
    saved = {}  # name → original DeviceAvailability

    def _fail(reason: str) -> tuple[Ledger, GrantResult]:
        return ledger, GrantFailure(error=reason)

    # Pass 1: check and tentatively subtract.
    for res_req in request.reservations:
        for slot in res_req.slots:
            if slot.name not in idx:
                return _fail(f"unknown device: {slot.name!r}")
            i = idx[slot.name]
            da = avail[i]
            if slot.name not in saved:
                saved[slot.name] = ledger.availability[i]
            if da.available_ram_mb < slot.ram_mb:
                return _fail(
                    f"insufficient RAM on {slot.name!r}: "
                    f"need {slot.ram_mb} MB, have {da.available_ram_mb} MB"
                )
            if da.available_compute_units < slot.compute_units:
                return _fail(
                    f"insufficient compute_units on {slot.name!r}: "
                    f"need {slot.compute_units}, have {da.available_compute_units}"
                )
            avail[i] = DeviceAvailability(
                device=da.device,
                available_ram_mb=da.available_ram_mb - slot.ram_mb,
                available_compute_units=da.available_compute_units - slot.compute_units,
            )

    # Pass 2: build granted reservations.
    now = time.time()
    granted_reservations: list[GrantedReservation] = []
    active_reservations: list[ActiveReservation] = list(ledger.reservations)

    for res_req in request.reservations:
        reservation_id = secrets.token_hex(3)  # 6 hex chars

        granted_slots: list[GrantedSlot] = []
        claimed_slots: list[ActiveReservation.ClaimedSlot] = []

        for slot in res_req.slots:
            i = idx[slot.name]
            dev = avail[i].device
            if dev.type == "cuda":
                # Extract physical ordinal from name e.g. "cuda:2" → 2
                try:
                    cuda_ordinal = int(slot.name.split(":")[1])
                except (IndexError, ValueError):
                    cuda_ordinal = -1
            else:
                cuda_ordinal = -1
            granted_slots.append(GrantedSlot(
                device_name=slot.name,
                cuda_ordinal=cuda_ordinal,
            ))
            claimed_slots.append(ActiveReservation.ClaimedSlot(
                device_name=slot.name,
                ram_mb=slot.ram_mb,
                compute_units=slot.compute_units,
            ))

        granted_reservations.append(GrantedReservation(
            reservation_id=reservation_id,
            label=res_req.label,
            slots=tuple(granted_slots),
            created_at=now,
        ))
        active_reservations.append(ActiveReservation(
            reservation_id=reservation_id,
            label=res_req.label,
            slots=tuple(claimed_slots),
            created_at=now,
        ))

    new_ledger = Ledger(
        availability=tuple(avail),
        reservations=tuple(active_reservations),
    )
    return new_ledger, GrantSuccess(reservations=tuple(granted_reservations))


def release(ledger: Ledger, reservation_id: str) -> Ledger:
    """
    Release a reservation by ID, returning capacity to the ledger.

    If the reservation_id is not found the ledger is returned unchanged.
    """
    # Find the reservation.
    target: ActiveReservation | None = None
    remaining: list[ActiveReservation] = []
    for ar in ledger.reservations:
        if ar.reservation_id == reservation_id:
            target = ar
        else:
            remaining.append(ar)

    if target is None:
        return ledger

    idx = _availability_index(ledger)
    avail = list(ledger.availability)

    for claimed in target.slots:
        if claimed.device_name not in idx:
            continue
        i = idx[claimed.device_name]
        da = avail[i]
        avail[i] = DeviceAvailability(
            device=da.device,
            available_ram_mb=da.available_ram_mb + claimed.ram_mb,
            available_compute_units=da.available_compute_units + claimed.compute_units,
        )

    return Ledger(
        availability=tuple(avail),
        reservations=tuple(remaining),
    )


def status(ledger: Ledger) -> StatusResult:
    """Return a snapshot of device availability and active reservations."""
    devices = tuple(
        DeviceStatus(
            name=da.device.name,
            type=da.device.type,
            model=da.device.model,
            ram_mb=da.device.ram_mb,
            available_ram_mb=da.available_ram_mb,
            compute_units=da.device.compute_units,
            available_compute_units=da.available_compute_units,
        )
        for da in ledger.availability
    )
    reservations = tuple(
        ReservationStatus(
            id=ar.reservation_id,
            label=ar.label,
            created_at=ar.created_at,
            slots=ar.slots,
        )
        for ar in ledger.reservations
    )
    return StatusResult(devices=devices, reservations=reservations)
