"""Pure, greedy placement functions. Only their explicit results reach the daemon.

Each policy attempts every reservation in declaration order and returns one
concrete batch, or NoPlacement. It does not enumerate alternatives or return a
partial batch. All inputs are rich device constraints, including optional pins.
The daemon is authoritative about whether selected assignments can be acquired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

from ..core import ReservationRequest, DeviceAllocation

if TYPE_CHECKING:
    from .client import DeviceConstraint, ReservationSpec


DeviceStatusDict = dict


@dataclass(frozen=True)
class NoPlacement:
    """This policy found no complete placement; not a proof of infeasibility."""

    reason: str


ReservationSolver = Callable[
    [Sequence[DeviceStatusDict], Sequence["ReservationSpec"]],
    list[ReservationRequest] | NoPlacement,
]


def _matches(device: DeviceStatusDict, requirement: DeviceConstraint) -> bool:
    return (
        (requirement.name is None or device["name"] == requirement.name)
        and (requirement.type is None or device["type"] == requirement.type)
        and (requirement.model is None or device.get("model") == requirement.model)
        and device["available_ram_mb"] >= requirement.ram_mb
        and device["available_compute_units"] >= requirement.compute_units
    )


def _place_reservations(
    capacity: Sequence[DeviceStatusDict],
    reservation_specs: Sequence[ReservationSpec],
    rank_device: Callable[[DeviceStatusDict], int],
) -> list[ReservationRequest] | NoPlacement:
    available = {device["name"]: dict(device) for device in capacity}
    reservation_requests = []
    for reservation_spec in reservation_specs:
        assignments = []
        for requirement in reservation_spec.requirements:
            candidates = [
                device for device in available.values()
                if _matches(device, requirement)
            ]
            if not candidates:
                return NoPlacement(
                    f"no device satisfies constraint {requirement!r} "
                    f"for reservation {reservation_spec.label!r}"
                )
            selected_device = min(candidates, key=rank_device)
            device_name = selected_device["name"]

            assignments.append(DeviceAllocation(
                device_name, requirement.ram_mb, requirement.compute_units,
            ))
            selected_device["available_ram_mb"] -= requirement.ram_mb
            selected_device["available_compute_units"] -= requirement.compute_units

        reservation_requests.append(ReservationRequest(
            reservation_spec.label, tuple(assignments),
        ))
    return reservation_requests


def solve_reservations_consolidated(
    capacity: Sequence[DeviceStatusDict],
    reservation_specs: Sequence[ReservationSpec],
) -> list[ReservationRequest] | NoPlacement:
    """Greedily prefer eligible devices with less available RAM, without IO.

    Labels and reservation order are preserved; neither input is changed.
    """
    return _place_reservations(
        capacity, reservation_specs, lambda device: device["available_ram_mb"],
    )


def solve_reservations_spread(
    capacity: Sequence[DeviceStatusDict],
    reservation_specs: Sequence[ReservationSpec],
) -> list[ReservationRequest] | NoPlacement:
    """Greedily prefer eligible devices with more available RAM, without IO."""
    return _place_reservations(
        capacity, reservation_specs, lambda device: -device["available_ram_mb"],
    )
