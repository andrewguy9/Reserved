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
from .planner import (
    Backlog, OwnedWorker, WorkerPlan, WorkerShortfall, WorkerStart,
    plan_workers, recommend_workers,
)

__all__ = [
    "Reservation",
    "adopt_inherited_lease",
    "Request",
    "DeviceConstraint",
    "ExplicitSlot",
    "GrantedDevice",
    "ConsolidateSolver",
    "SpreadSolver",
    "Backlog",
    "OwnedWorker",
    "WorkerPlan",
    "WorkerShortfall",
    "WorkerStart",
    "plan_workers",
    "recommend_workers",
    "ReservationError",
    "DaemonNotRunningError",
    "ReservationExpiredError",
    "InheritedLeaseAlreadyAdoptedError",
]
