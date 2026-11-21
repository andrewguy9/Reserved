"""
solvers.py — Constraint solvers for the reserved SDK.

Solvers live entirely in the SDK — they never cross the wire. A solver reads
the current device status, resolves each DeviceConstraint to an explicit
DeviceAllocation, and returns a ReserveRequest ready to send to the daemon.
A DeviceAllocation given directly passes through unchanged.

Two built-in solvers:
  - ConsolidateSolver: pack allocations onto fewest physical devices (default).
  - SpreadSolver: distribute allocations across as many physical devices as possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import DeviceConstraint, ReservationSpec

from ..core import (
    DeviceAllocation,
    ReservationRequest,
    ReserveRequest,
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
    Pack allocations onto as few physical devices as possible.

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

        *reservations* is a list of (label, specs) where each spec is either a
        DeviceConstraint or a DeviceAllocation.  *devices* is the list of device
        dicts from a status response.
        """

        # Work with a mutable copy of available capacities for simulation.
        avail: dict[str, dict] = {d["name"]: dict(d) for d in devices}

        res_requests: list[ReservationRequest] = []

        for label, specs in reservations:
            allocations: list[DeviceAllocation] = []
            for spec in specs:
                if isinstance(spec, DeviceAllocation):
                    allocations.append(spec)
                    # Update simulated availability.
                    if spec.name in avail:
                        avail[spec.name]["available_ram_mb"] -= spec.ram_mb
                        avail[spec.name]["available_compute_units"] -= spec.compute_units
                else:
                    # DeviceConstraint — find best device.
                    candidates = [
                        d for d in avail.values()
                        if _matches(d, spec)
                    ]
                    if not candidates:
                        raise ValueError(
                            f"no device satisfies constraint {spec!r} "
                            f"for reservation {label!r}"
                        )
                    # Consolidate: prefer device with least available RAM
                    # (most full) that still fits.
                    best = min(candidates, key=lambda d: d["available_ram_mb"])
                    allocations.append(DeviceAllocation(
                        name=best["name"],
                        ram_mb=spec.ram_mb,
                        compute_units=spec.compute_units,
                    ))
                    avail[best["name"]]["available_ram_mb"] -= spec.ram_mb
                    avail[best["name"]]["available_compute_units"] -= spec.compute_units

            res_requests.append(ReservationRequest(
                label=label,
                allocations=tuple(allocations),
            ))

        return ReserveRequest(reservations=tuple(res_requests))


class SpreadSolver:
    """
    Distribute allocations across as many physical devices as possible.

    For each constraint, pick the eligible device that has the most remaining
    available RAM (i.e. most empty). This maximises parallel throughput.
    """

    def solve(
        self,
        reservations: list["ReservationSpec"],
        devices: list[DeviceStatusDict],
    ) -> ReserveRequest:
        avail: dict[str, dict] = {d["name"]: dict(d) for d in devices}

        res_requests: list[ReservationRequest] = []

        for label, specs in reservations:
            allocations: list[DeviceAllocation] = []
            for spec in specs:
                if isinstance(spec, DeviceAllocation):
                    allocations.append(spec)
                    if spec.name in avail:
                        avail[spec.name]["available_ram_mb"] -= spec.ram_mb
                        avail[spec.name]["available_compute_units"] -= spec.compute_units
                else:
                    candidates = [
                        d for d in avail.values()
                        if _matches(d, spec)
                    ]
                    if not candidates:
                        raise ValueError(
                            f"no device satisfies constraint {spec!r} "
                            f"for reservation {label!r}"
                        )
                    # Spread: prefer device with most available RAM (most empty).
                    best = max(candidates, key=lambda d: d["available_ram_mb"])
                    allocations.append(DeviceAllocation(
                        name=best["name"],
                        ram_mb=spec.ram_mb,
                        compute_units=spec.compute_units,
                    ))
                    avail[best["name"]]["available_ram_mb"] -= spec.ram_mb
                    avail[best["name"]]["available_compute_units"] -= spec.compute_units

            res_requests.append(ReservationRequest(
                label=label,
                allocations=tuple(allocations),
            ))

        return ReserveRequest(reservations=tuple(res_requests))
