"""
solvers.py — Constraint solvers for the reserved SDK.

Solvers live entirely in the SDK — they never cross the wire. A solver reads
the current device status, resolves DeviceConstraint objects to explicit slot
assignments, and returns a ReserveRequest ready to send to the daemon.

Two built-in solvers:
  - ConsolidateSolver: pack slots onto fewest physical devices (default).
  - SpreadSolver: distribute slots across as many physical devices as possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .client import DeviceConstraint, ReservationSpec

from ..core import (
    ReservationRequest,
    ReserveRequest,
    SlotRequest,
)


# ---------------------------------------------------------------------------
# Type alias for a parsed status response (dict from JSON)
# ---------------------------------------------------------------------------

DeviceStatusDict = dict  # keys: name, type, model, ram_mb, available_ram_mb, etc.


def _matches(device: DeviceStatusDict, constraint: "DeviceConstraint") -> bool:
    """Return True if *device* satisfies *constraint*."""
    from .client import DeviceConstraint  # local import to avoid circular
    if constraint.type is not None and device["type"] != constraint.type:
        return False
    if constraint.model is not None and device.get("model") != constraint.model:
        return False
    if device["available_ram_mb"] < constraint.ram_mb:
        return False
    if device["available_compute_units"] < constraint.compute_units:
        return False
    return True


class ConsolidateSolver:
    """
    Pack slots onto as few physical devices as possible.

    For each constraint, pick the eligible device that has the least remaining
    available RAM (i.e. most full) while still satisfying the constraint.
    This leaves the maximum contiguous free capacity on other devices.
    """

    def solve(
        self,
        reservations: list["ReservationSpec"],
        devices: list[DeviceStatusDict],
    ) -> ReserveRequest:
        """
        Resolve constraints to an explicit ReserveRequest.

        *reservations* is a list of (label, slots) where each slot is either a
        DeviceConstraint or an ExplicitSlot.  *devices* is the list of device
        dicts from a status response.
        """
        from .client import DeviceConstraint, ExplicitSlot

        # Work with a mutable copy of available capacities for simulation.
        avail: dict[str, dict] = {d["name"]: dict(d) for d in devices}

        res_requests: list[ReservationRequest] = []

        for label, slots in reservations:
            slot_requests: list[SlotRequest] = []
            for slot in slots:
                if isinstance(slot, ExplicitSlot):
                    slot_requests.append(SlotRequest(
                        name=slot.name,
                        ram_mb=slot.ram_mb,
                        compute_units=slot.compute_units,
                    ))
                    # Update simulated availability.
                    if slot.name in avail:
                        avail[slot.name]["available_ram_mb"] -= slot.ram_mb
                        avail[slot.name]["available_compute_units"] -= slot.compute_units
                else:
                    # DeviceConstraint — find best device.
                    candidates = [
                        d for d in avail.values()
                        if _matches(d, slot)
                    ]
                    if not candidates:
                        raise ValueError(
                            f"no device satisfies constraint {slot!r} "
                            f"for reservation {label!r}"
                        )
                    # Consolidate: prefer device with least available RAM
                    # (most full) that still fits.
                    best = min(candidates, key=lambda d: d["available_ram_mb"])
                    slot_requests.append(SlotRequest(
                        name=best["name"],
                        ram_mb=slot.ram_mb,
                        compute_units=slot.compute_units,
                    ))
                    avail[best["name"]]["available_ram_mb"] -= slot.ram_mb
                    avail[best["name"]]["available_compute_units"] -= slot.compute_units

            res_requests.append(ReservationRequest(
                label=label,
                slots=tuple(slot_requests),
            ))

        return ReserveRequest(reservations=tuple(res_requests))


class SpreadSolver:
    """
    Distribute slots across as many physical devices as possible.

    For each constraint, pick the eligible device that has the most remaining
    available RAM (i.e. most empty). This maximises parallel throughput.
    """

    def solve(
        self,
        reservations: list["ReservationSpec"],
        devices: list[DeviceStatusDict],
    ) -> ReserveRequest:
        from .client import DeviceConstraint, ExplicitSlot

        avail: dict[str, dict] = {d["name"]: dict(d) for d in devices}

        res_requests: list[ReservationRequest] = []

        for label, slots in reservations:
            slot_requests: list[SlotRequest] = []
            for slot in slots:
                if isinstance(slot, ExplicitSlot):
                    slot_requests.append(SlotRequest(
                        name=slot.name,
                        ram_mb=slot.ram_mb,
                        compute_units=slot.compute_units,
                    ))
                    if slot.name in avail:
                        avail[slot.name]["available_ram_mb"] -= slot.ram_mb
                        avail[slot.name]["available_compute_units"] -= slot.compute_units
                else:
                    candidates = [
                        d for d in avail.values()
                        if _matches(d, slot)
                    ]
                    if not candidates:
                        raise ValueError(
                            f"no device satisfies constraint {slot!r} "
                            f"for reservation {label!r}"
                        )
                    # Spread: prefer device with most available RAM (most empty).
                    best = max(candidates, key=lambda d: d["available_ram_mb"])
                    slot_requests.append(SlotRequest(
                        name=best["name"],
                        ram_mb=slot.ram_mb,
                        compute_units=slot.compute_units,
                    ))
                    avail[best["name"]]["available_ram_mb"] -= slot.ram_mb
                    avail[best["name"]]["available_compute_units"] -= slot.compute_units

            res_requests.append(ReservationRequest(
                label=label,
                slots=tuple(slot_requests),
            ))

        return ReserveRequest(reservations=tuple(res_requests))
