"""
reserved SDK — Python client library for the reserved daemon.
"""

from .client import (
    DeviceConstraint,
    ExplicitSlot,
    GrantedDevice,
    Request,
    Reservation,
    adopt_inherited_lease,
)
from .errors import (
    DaemonNotRunningError,
    InheritedLeaseAlreadyAdoptedError,
    ReservationError,
    ReservationExpiredError,
)
from .solvers import ConsolidateSolver, SpreadSolver

__all__ = [
    "Reservation",
    "adopt_inherited_lease",
    "Request",
    "DeviceConstraint",
    "ExplicitSlot",
    "GrantedDevice",
    "ConsolidateSolver",
    "SpreadSolver",
    "ReservationError",
    "DaemonNotRunningError",
    "ReservationExpiredError",
    "InheritedLeaseAlreadyAdoptedError",
]
