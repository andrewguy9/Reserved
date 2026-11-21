"""
core.py — Pure functional core for the reserved daemon.

All business logic lives here as pure functions operating on immutable frozen
dataclasses. No IO, no sockets, no global state. Every function takes a Ledger
and returns a new Ledger plus a result object.
"""

from __future__ import annotations

import re
import secrets
import time
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Ledger types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Device:
    name: str  # Physical device name.
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
class DeviceAllocation:
    """An explicit device allocation: this device, these resources.

    Clients send it in a reserve request, and a successful grant returns the
    same record for each allocation it holds.
    """
    name: str          # physical device name, e.g. "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782" or "cpu"
    ram_mb: int = 0
    compute_units: int = 0


@dataclass(frozen=True)
class ReservationRequest:
    """One reservation in a reserve request."""
    label: str
    allocations: tuple[DeviceAllocation, ...]


@dataclass(frozen=True)
class ReserveRequest:
    """A full reserve request — one or more reservations, granted atomically."""
    reservations: tuple[ReservationRequest, ...]


@dataclass(frozen=True)
class GrantedReservation:
    reservation_id: str
    label: str
    allocations: tuple[DeviceAllocation, ...]
    created_at: float


@dataclass(frozen=True)
class ActiveReservation:
    """Tracks a live reservation in the ledger: the allocations it holds."""
    reservation_id: str
    label: str
    allocations: tuple[DeviceAllocation, ...]
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
    name: str  # Physical device name.
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
    allocations: tuple[DeviceAllocation, ...]


@dataclass(frozen=True)
class StatusResult:
    devices: tuple[DeviceStatus, ...]
    reservations: tuple[ReservationStatus, ...]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

_CUDA_UUID = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)


def cuda_uuid(name: str) -> str:
    """Return the GPU UUID encoded in a ``cuda:GPU-<uuid>`` device name.

    The UUID is the device's only identity. It is what ``nvidia-smi -L``
    prints, what ``CUDA_VISIBLE_DEVICES`` accepts, and what torch reports for
    each visible device, so no enumeration order is involved anywhere. The hex
    digits must be lowercase, exactly as the driver prints them. Raises
    ValueError for any other name, including non-CUDA names.
    """
    kind, sep, uuid_text = name.partition(":")
    if kind != "cuda" or not sep or not _CUDA_UUID.match(uuid_text):
        raise ValueError(
            f"invalid CUDA device name {name!r}: must be 'cuda:GPU-<uuid>' "
            "with the lowercase UUID printed by 'nvidia-smi -L'"
        )
    return uuid_text


def _validate_device_name(name: str, device_type: str) -> None:
    """Raise ValueError if a device name is invalid for its type."""
    if device_type == "cuda":
        cuda_uuid(name)


def load_ledger(config: dict) -> Ledger:
    """Build an initial Ledger from a config dict (parsed from reserved.json)."""
    devices = []
    for d in config.get("devices", []):
        name = d["name"]
        device_type = d["type"]
        _validate_device_name(name, device_type)
        dev = Device(
            name=name,
            type=device_type,
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

    Returns (new_ledger, GrantSuccess) if all allocations can be satisfied, or
    (original_ledger, GrantFailure) if any allocation cannot be satisfied.
    """
    idx = _availability_index(ledger)
    avail = list(ledger.availability)  # mutable working copy

    # Collect all allocations across all reservations for atomic checking.
    # We'll simulate the subtraction in avail[] and roll back on failure.
    saved = {}  # name → original DeviceAvailability

    def _fail(reason: str) -> tuple[Ledger, GrantResult]:
        return ledger, GrantFailure(error=reason)

    # Pass 1: check and tentatively subtract.
    for res_req in request.reservations:
        for allocation in res_req.allocations:
            if allocation.name not in idx:
                return _fail(f"unknown device: {allocation.name!r}")
            i = idx[allocation.name]
            da = avail[i]
            if allocation.name not in saved:
                saved[allocation.name] = ledger.availability[i]
            if da.available_ram_mb < allocation.ram_mb:
                return _fail(
                    f"insufficient RAM on {allocation.name!r}: "
                    f"need {allocation.ram_mb} MB, have {da.available_ram_mb} MB"
                )
            if da.available_compute_units < allocation.compute_units:
                return _fail(
                    f"insufficient compute_units on {allocation.name!r}: "
                    f"need {allocation.compute_units}, have {da.available_compute_units}"
                )
            avail[i] = DeviceAvailability(
                device=da.device,
                available_ram_mb=da.available_ram_mb - allocation.ram_mb,
                available_compute_units=da.available_compute_units - allocation.compute_units,
            )

    # Pass 2: build granted reservations.
    now = time.time()
    granted_reservations: list[GrantedReservation] = []
    active_reservations: list[ActiveReservation] = list(ledger.reservations)

    for res_req in request.reservations:
        reservation_id = secrets.token_hex(3)  # 6 hex chars

        # The grant and the ledger both hold the allocations exactly as requested.
        granted_reservations.append(GrantedReservation(
            reservation_id=reservation_id,
            label=res_req.label,
            allocations=res_req.allocations,
            created_at=now,
        ))
        active_reservations.append(ActiveReservation(
            reservation_id=reservation_id,
            label=res_req.label,
            allocations=res_req.allocations,
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

    for claimed in target.allocations:
        if claimed.name not in idx:
            continue
        i = idx[claimed.name]
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
            allocations=ar.allocations,
        )
        for ar in ledger.reservations
    )
    return StatusResult(devices=devices, reservations=reservations)
