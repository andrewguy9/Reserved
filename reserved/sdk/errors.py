"""
errors.py — Exception types for the reserved SDK.
"""


class ReservationError(Exception):
    """Raised by acquire() when the daemon rejects a reservation request."""


class DaemonNotRunningError(ReservationError):
    """Raised when the daemon socket does not exist or connection is refused."""


class ReservationExpiredError(ReservationError):
    """
    Raised by Reservation.check() when the lease fd has been closed by the
    daemon (e.g. daemon restarted). Treat as a signal to abort and exit.
    """
