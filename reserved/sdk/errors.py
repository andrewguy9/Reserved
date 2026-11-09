"""
errors.py — Exception types for the reserved SDK.
"""


class ReservationError(Exception):
    """Base error for reservation acquisition and lease lifecycle operations."""


class DaemonNotRunningError(ReservationError):
    """Raised when the daemon socket does not exist or connection is refused."""


class ReservationExpiredError(ReservationError):
    """
    Raised by Reservation.check() when the lease fd has been closed by the
    daemon (e.g. daemon restarted). Treat as a signal to abort and exit.
    """


class InheritedLeaseAlreadyAdoptedError(ReservationError):
    """Raised when the process has already claimed the launcher handoff."""
