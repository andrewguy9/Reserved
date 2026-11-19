"""
solvers.py — Constraint solvers for the reserved SDK.

Solvers are pure computations over supplied capacity and grouped requirements.
They return explicit slot assignments in input order. Labels, daemon requests,
and lease acquisition belong to the SDK/CLI shell.

Two built-in solvers:
  - ConsolidateSolver: pack slots onto fewest physical devices (default).
  - SpreadSolver: distribute slots across as many physical devices as possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .client import DeviceConstraint, ExplicitSlot

from ..core import SlotRequest


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
        capacity: list[DeviceStatusDict],
        requirements: Sequence[Sequence[DeviceConstraint | ExplicitSlot]],
    ) -> tuple[tuple[SlotRequest, ...], ...]:
        """
        Return one tuple of assigned slots per requirement group, without IO
        or changing either input. Raise ValueError when placement fails.
        """
        from .client import DeviceConstraint, ExplicitSlot

        # Work with a mutable copy of available capacities for simulation.
        avail: dict[str, dict] = {d["name"]: dict(d) for d in capacity}

        slot_groups: list[tuple[SlotRequest, ...]] = []

        for slots in requirements:
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
                            f"no device satisfies constraint {slot!r}"
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

            slot_groups.append(tuple(slot_requests))

        return tuple(slot_groups)


class SpreadSolver:
    """
    Distribute slots across as many physical devices as possible.

    For each constraint, pick the eligible device that has the most remaining
    available RAM (i.e. most empty). This maximises parallel throughput.
    """

    def solve(
        self,
        capacity: list[DeviceStatusDict],
        requirements: Sequence[Sequence[DeviceConstraint | ExplicitSlot]],
    ) -> tuple[tuple[SlotRequest, ...], ...]:
        from .client import DeviceConstraint, ExplicitSlot

        avail: dict[str, dict] = {d["name"]: dict(d) for d in capacity}

        slot_groups: list[tuple[SlotRequest, ...]] = []

        for slots in requirements:
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
                            f"no device satisfies constraint {slot!r}"
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

            slot_groups.append(tuple(slot_requests))

        return tuple(slot_groups)
