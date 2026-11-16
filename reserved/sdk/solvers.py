"""Resolve SDK constraints to explicit requests against a device snapshot."""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING, Callable, Iterable

from ..core import ReservationRequest, ReserveRequest, SlotRequest

if TYPE_CHECKING:
    from .client import DeviceConstraint, ExplicitSlot, ReservationSpec


DeviceStatusDict = dict
_free_ram = itemgetter("available_ram_mb")


def _matches(device: DeviceStatusDict, constraint: DeviceConstraint,
             ram_mb: int, compute_units: int) -> bool:
    return (
        (constraint.type is None or device["type"] == constraint.type)
        and (constraint.model is None or device.get("model") == constraint.model)
        and ram_mb >= constraint.ram_mb
        and compute_units >= constraint.compute_units
    )


class _DevicePool:
    """A private copy of free capacity shared across placements in one solve."""

    def __init__(self, devices: Iterable[DeviceStatusDict], select: Callable) -> None:
        self._available = {device["name"]: dict(device) for device in devices}
        self._select = select

    def place(self, constraint: DeviceConstraint | ExplicitSlot) -> SlotRequest | None:
        from .client import ExplicitSlot

        if isinstance(constraint, ExplicitSlot):
            # Pass explicit names through; the daemon validates the full request.
            name = constraint.name
            device = self._available.get(name)
        else:
            candidates = (
                device for device in self._available.values()
                if _matches(device, constraint, device["available_ram_mb"],
                            device["available_compute_units"])
            )
            device = self._select(candidates, key=_free_ram, default=None)
            if device is None:
                return None
            name = device["name"]
        if device is not None:
            device["available_ram_mb"] -= constraint.ram_mb
            device["available_compute_units"] -= constraint.compute_units
        return SlotRequest(name, constraint.ram_mb, constraint.compute_units)


def _solve(reservations: list[ReservationSpec], pool: _DevicePool) -> ReserveRequest:
    requests = []
    for label, slots in reservations:
        placed = []
        for slot in slots:
            assignment = pool.place(slot)
            if assignment is None:
                raise ValueError(f"no device satisfies constraint {slot!r} for reservation {label!r}")
            placed.append(assignment)
        requests.append(ReservationRequest(label, tuple(placed)))
    return ReserveRequest(tuple(requests))


class ConsolidateSolver:
    """Prefer the fitting device with the least available RAM."""

    def solve(
        self, reservations: list[ReservationSpec], devices: list[DeviceStatusDict],
    ) -> ReserveRequest:
        return _solve(reservations, _DevicePool(devices, min))


class SpreadSolver:
    """Prefer the fitting device with the most available RAM."""

    def _pool(self, devices: Iterable[DeviceStatusDict]) -> _DevicePool:
        return _DevicePool(devices, max)

    def solve(
        self, reservations: list[ReservationSpec], devices: list[DeviceStatusDict],
    ) -> ReserveRequest:
        return _solve(reservations, self._pool(devices))
