"""
reserved SDK — Python client library for the reserved daemon.
"""

from reserved.core import DeviceAllocation

from .client import (
    DeviceConstraint,
    Request,
    Reservation,
    adopt_inherited_lease,
    torch_device,
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
    "DeviceAllocation",
    "torch_device",
    "ConsolidateSolver",
    "SpreadSolver",
    "ReservationError",
    "DaemonNotRunningError",
    "ReservationExpiredError",
    "InheritedLeaseAlreadyAdoptedError",
]
