"""
reserved SDK — Python client library for the reserved daemon.
"""

from .client import (
    DeviceConstraint,
    ExplicitSlot,
    GrantedDevice,
    Request,
    Reservation,
    ReservationSpec,
    acquire_reservations,
    adopt_inherited_lease,
)
from .errors import (
    DaemonNotRunningError,
    InheritedLeaseAlreadyAdoptedError,
    ReservationError,
    ReservationExpiredError,
)
from reserved.core import ReservationRequest, ReserveRequest, DeviceAllocation
from .solvers import (
    NoPlacement,
    ReservationSolver,
    solve_reservations_consolidated,
    solve_reservations_spread,
)

__all__ = [
    "Reservation",
    "acquire_reservations",
    "adopt_inherited_lease",
    "Request",
    "DeviceConstraint",
    "ExplicitSlot",
    "GrantedDevice",
    "ReservationSpec",
    "ReservationRequest",
    "ReserveRequest",
    "DeviceAllocation",
    "NoPlacement",
    "ReservationSolver",
    "solve_reservations_consolidated",
    "solve_reservations_spread",
    "ReservationError",
    "DaemonNotRunningError",
    "ReservationExpiredError",
    "InheritedLeaseAlreadyAdoptedError",
]
