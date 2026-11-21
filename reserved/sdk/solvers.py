"""
solvers.py — Constraint solvers for the reserved SDK.

Solvers live entirely in the SDK — they never cross the wire. A solver reads
the current device status, resolves each DeviceConstraint to an explicit
DeviceAllocation, and returns a ReserveRequest ready to send to the daemon.
A constraint that names a device resolves to that device, if it fits.

Two built-in solvers:
  - ConsolidateSolver: pack allocations onto fewest physical devices (default).
  - SpreadSolver: distribute allocations across as many physical devices as possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

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
    """Return True if *device* satisfies every field of *constraint*."""
    if constraint.name is not None and device["name"] != constraint.name:
        return False
    if constraint.type is not None and device["type"] != constraint.type:
        return False
    if device["available_ram_mb"] < constraint.ram_mb:
        return False
    if device["available_compute_units"] < constraint.compute_units:
        return False
    return True


def _place(
    reservations: list["ReservationSpec"],
    devices: list[DeviceStatusDict],
    choose: Callable[[list[DeviceStatusDict]], DeviceStatusDict],
) -> ReserveRequest:
    """Resolve every constraint in order, picking among eligible devices with *choose*.

    Capacity is simulated as constraints are placed, so later constraints see
    what earlier ones in the same request have already claimed. Raises
    ValueError when no device satisfies a constraint.
    """
    avail: dict[str, dict] = {d["name"]: dict(d) for d in devices}

    res_requests: list[ReservationRequest] = []
    for label, constraints in reservations:
        allocations: list[DeviceAllocation] = []
        for constraint in constraints:
            candidates = [d for d in avail.values() if _matches(d, constraint)]
            if not candidates:
                raise ValueError(
                    f"no device satisfies constraint {constraint!r} "
                    f"for reservation {label!r}"
                )
            best = choose(candidates)
            allocations.append(DeviceAllocation(
                name=best["name"],
                ram_mb=constraint.ram_mb,
                compute_units=constraint.compute_units,
            ))
            avail[best["name"]]["available_ram_mb"] -= constraint.ram_mb
            avail[best["name"]]["available_compute_units"] -= constraint.compute_units
        res_requests.append(ReservationRequest(label=label, allocations=tuple(allocations)))

    return ReserveRequest(reservations=tuple(res_requests))


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

        *reservations* is a list of (label, constraints) where each constraint
        is a DeviceConstraint.  *devices* is the list of device dicts from a
        status response.
        """
        return _place(reservations, devices, lambda c: min(c, key=lambda d: d["available_ram_mb"]))


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
        return _place(reservations, devices, lambda c: max(c, key=lambda d: d["available_ram_mb"]))
