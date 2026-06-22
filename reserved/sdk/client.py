"""
client.py — Python SDK client for the reserved daemon.

Public API:
  Reservation(label, *slots, solver=ConsolidateSolver(), socket_path=None)
  Request(*reservations, socket_path=None)
  DeviceConstraint(type, *, model=None, ram_mb=0, compute_units=0)
  ExplicitSlot(name, *, ram_mb=0, compute_units=0)
  GrantedDevice(device_name, device_index)
"""

from __future__ import annotations

import array
import os
import select
import socket
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..core import (
    ReservationRequest,
    ReserveRequest,
    SlotRequest,
)
from .. import protocol
from .errors import DaemonNotRunningError, ReservationError, ReservationExpiredError
from .solvers import ConsolidateSolver

# Type alias used by solvers.
ReservationSpec = tuple[str, list]  # (label, list[DeviceConstraint | ExplicitSlot])

_DEFAULT_SOCKET = "/var/run/reserved.sock"


def _get_socket_path(override: str | None = None) -> str:
    if override:
        return override
    return os.environ.get("RESERVED_SOCKET", _DEFAULT_SOCKET)


# ---------------------------------------------------------------------------
# Slot descriptor types (in-process only, never serialized)
# ---------------------------------------------------------------------------

@dataclass
class DeviceConstraint:
    """A constraint-based slot — the solver picks the device."""
    type: str | None = None
    model: str | None = None
    ram_mb: int = 0
    compute_units: int = 0

    def __init__(
        self,
        type: str | None = None,
        *,
        model: str | None = None,
        ram_mb: int = 0,
        compute_units: int = 0,
    ) -> None:
        self.type = type
        self.model = model
        self.ram_mb = ram_mb
        self.compute_units = compute_units


@dataclass
class ExplicitSlot:
    """An explicit slot — directly names a physical device."""
    name: str
    ram_mb: int = 0
    compute_units: int = 0

    def __init__(self, name: str, *, ram_mb: int = 0, compute_units: int = 0) -> None:
        self.name = name
        self.ram_mb = ram_mb
        self.compute_units = compute_units


# ---------------------------------------------------------------------------
# GrantedDevice — returned to callers after acquire()
# ---------------------------------------------------------------------------

@dataclass
class GrantedDevice:
    """A physical device slot granted by the daemon."""
    device_name: str
    device_index: int  # 0-based CUDA index within reservation; -1 for non-CUDA

    def torch_device(self) -> Any:
        """
        Return an appropriate torch.device for this granted device.

        Does not import torch at module level — only when called.
        """
        import torch  # type: ignore[import]
        dev_type = self.device_name.split(":")[0]
        if dev_type == "cuda":
            return torch.device(f"cuda:{self.device_index}")
        elif dev_type == "metal":
            return torch.device("mps")
        else:
            return torch.device("cpu")


# ---------------------------------------------------------------------------
# SCM_RIGHTS receive helper
# ---------------------------------------------------------------------------

def _recv_fds(sock: socket.socket, bufsize: int, maxfds: int) -> tuple[bytes, list[int]]:
    """Receive data plus up to *maxfds* file descriptors via SCM_RIGHTS."""
    fd_array = array.array("i")
    cmsg_space = socket.CMSG_SPACE(maxfds * fd_array.itemsize)
    msg, ancdata, _flags, _addr = sock.recvmsg(bufsize, cmsg_space)
    fds: list[int] = []
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fd_array.frombytes(
                cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fd_array.itemsize)]
            )
            fds.extend(fd_array)
    return msg, fds


# ---------------------------------------------------------------------------
# Low-level transport
# ---------------------------------------------------------------------------

def _connect(socket_path: str) -> socket.socket:
    """Open a transport connection to the daemon."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(socket_path)
    except FileNotFoundError as exc:
        raise DaemonNotRunningError(
            f"daemon socket not found: {socket_path!r}"
        ) from exc
    except ConnectionRefusedError as exc:
        raise DaemonNotRunningError(
            f"daemon not running (connection refused): {socket_path!r}"
        ) from exc
    return sock


def _query_status(socket_path: str) -> dict:
    """Send a status request and return the parsed response dict."""
    sock = _connect(socket_path)
    try:
        sock.sendall(protocol.status_request_to_wire())
        # Read until newline.
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        line, _, _ = buf.partition(b"\n")
        return protocol.parse_response(line)
    finally:
        sock.close()


def _send_reserve(
    socket_path: str,
    request: ReserveRequest,
    num_reservations: int,
) -> tuple[dict, list[int]]:
    """
    Send a reserve request and receive the response plus lease fds.

    Returns (response_dict, list_of_fds).
    """
    sock = _connect(socket_path)
    try:
        wire = protocol.reserve_request_to_wire(request)
        sock.sendall(wire)

        # Receive response + SCM_RIGHTS fds.
        # We may need to call recvmsg multiple times if the response is large.
        buf = b""
        received_fds: list[int] = []

        # First recvmsg — should arrive with the fds attached.
        chunk, fds = _recv_fds(sock, 65536, num_reservations)
        buf += chunk
        received_fds.extend(fds)

        # Keep reading until we have a complete line.
        while b"\n" not in buf:
            more = sock.recv(4096)
            if not more:
                break
            buf += more

        line, _, _ = buf.partition(b"\n")
        resp = protocol.parse_response(line)
        return resp, received_fds
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Reservation
# ---------------------------------------------------------------------------

class Reservation:
    """
    A named reservation of one or more device slots.

    Typically used as a context manager:

        with Reservation("train", DeviceConstraint("cuda", ram_mb=16384)) as r:
            model.to(r.devices[0].torch_device())
            run_training()

    Can also be acquired manually:

        r = Reservation("train", DeviceConstraint("cuda"))
        r = r.acquire_single()   # returns self after acquiring
        ...
        r.release()
    """

    def __init__(
        self,
        label: str,
        *slots: DeviceConstraint | ExplicitSlot,
        solver: ConsolidateSolver | None = None,
        socket_path: str | None = None,
    ) -> None:
        self.label = label
        self._slots = list(slots)
        self._solver = solver if solver is not None else ConsolidateSolver()
        self._socket_path = socket_path

        # Set after acquire.
        self.reservation_id: str | None = None
        self.devices: list[GrantedDevice] = []
        self._lease_sock: socket.socket | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Reservation":
        self._acquire_single()
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Acquire (called by Request.acquire() or __enter__)
    # ------------------------------------------------------------------

    def _acquire_single(self) -> None:
        """Acquire this reservation by itself (not part of a multi-reservation Request)."""
        from ..core import ReservationRequest, ReserveRequest, SlotRequest
        socket_path = _get_socket_path(self._socket_path)

        if self._all_explicit():
            # Skip solver step.
            slots = [
                SlotRequest(name=s.name, ram_mb=s.ram_mb, compute_units=s.compute_units)
                for s in self._slots
            ]
            req = ReserveRequest(reservations=(
                ReservationRequest(label=self.label, slots=tuple(slots)),
            ))
        else:
            status = _query_status(socket_path)
            req = self._solver.solve(
                [(self.label, self._slots)],
                status["devices"],
            )

        resp, fds = _send_reserve(socket_path, req, num_reservations=1)
        if not resp.get("ok"):
            # Close any stray fds.
            for fd in fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise ReservationError(resp.get("error", "reservation rejected"))

        granted = resp["reservations"][0]
        self.reservation_id = granted["reservation_id"]
        self.devices = [
            GrantedDevice(device_name=s["device_name"], device_index=s["device_index"])
            for s in granted["slots"]
        ]
        if fds:
            self._lease_sock = socket.socket(fileno=fds[0])
        else:
            self._lease_sock = None

    def _all_explicit(self) -> bool:
        return all(isinstance(s, ExplicitSlot) for s in self._slots)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release this reservation by closing the lease fd."""
        if self._lease_sock is not None:
            try:
                self._lease_sock.close()
            except OSError:
                pass
            self._lease_sock = None

    def check(self) -> None:
        """
        Non-blocking check that the reservation lease is still alive.

        Raises ReservationExpiredError if the daemon has closed its end of the
        socketpair (e.g. daemon restarted). Returns immediately if healthy.
        """
        if self._lease_sock is None:
            raise ReservationExpiredError("reservation not acquired or already released")
        fd = self._lease_sock.fileno()
        if fd < 0:
            raise ReservationExpiredError("lease fd is closed")
        readable, _, _ = select.select([fd], [], [], 0.0)
        if readable:
            # Something is readable — either data (unexpected) or EOF.
            try:
                data = self._lease_sock.recv(1, socket.MSG_DONTWAIT)
            except BlockingIOError:
                return  # not actually readable yet
            if data == b"":
                raise ReservationExpiredError(
                    f"reservation {self.reservation_id!r} has expired (daemon restarted)"
                )

    def torch_devices(self) -> list[Any]:
        """Convenience: return a list of torch.device for all granted slots."""
        return [d.torch_device() for d in self.devices]

    def __repr__(self) -> str:
        return (
            f"Reservation(label={self.label!r}, "
            f"id={self.reservation_id!r}, "
            f"devices={self.devices!r})"
        )


# ---------------------------------------------------------------------------
# Request — multi-reservation atomic acquire
# ---------------------------------------------------------------------------

class Request:
    """
    Atomically acquire multiple reservations in a single wire request.

    All reservations are granted or none are. Each is independently releasable.

    Usage:
        checkpoint, lora = Request(
            Reservation("checkpoint", DeviceConstraint("cuda", ram_mb=8192)),
            Reservation("lora",       DeviceConstraint("cuda", ram_mb=2048)),
        ).acquire()
    """

    def __init__(
        self,
        *reservations: Reservation,
        socket_path: str | None = None,
    ) -> None:
        self._reservations = list(reservations)
        self._socket_path = socket_path

    def acquire(self) -> tuple[Reservation, ...]:
        """
        Send a single atomic reserve request for all reservations.

        Returns a tuple of Reservation objects (in declaration order), each
        populated with granted device info and holding its own lease fd.

        Raises ReservationError if the daemon rejects.
        Raises DaemonNotRunningError if the daemon is not running.
        """
        if not self._reservations:
            return ()

        socket_path = _get_socket_path(self._socket_path)

        # Determine if all slots are explicit (across all reservations).
        all_explicit = all(
            all(isinstance(s, ExplicitSlot) for s in r._slots)
            for r in self._reservations
        )

        if all_explicit:
            req = self._build_explicit_request()
        else:
            status = _query_status(socket_path)
            req = self._solve(status["devices"])

        num = len(self._reservations)
        resp, fds = _send_reserve(socket_path, req, num_reservations=num)

        if not resp.get("ok"):
            for fd in fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise ReservationError(resp.get("error", "reservation rejected"))

        granted_list = resp["reservations"]
        result: list[Reservation] = []
        for res, granted, fd in zip(self._reservations, granted_list, fds):
            res.reservation_id = granted["reservation_id"]
            res.devices = [
                GrantedDevice(device_name=s["device_name"], device_index=s["device_index"])
                for s in granted["slots"]
            ]
            res._lease_sock = socket.socket(fileno=fd)
            result.append(res)

        # If fewer fds than reservations (shouldn't happen but be safe):
        for res in self._reservations[len(fds):]:
            res._lease_sock = None

        return tuple(result)

    def _build_explicit_request(self) -> ReserveRequest:
        from ..core import ReservationRequest, ReserveRequest, SlotRequest
        res_reqs = []
        for r in self._reservations:
            slots = [
                SlotRequest(name=s.name, ram_mb=s.ram_mb, compute_units=s.compute_units)
                for s in r._slots
            ]
            res_reqs.append(ReservationRequest(label=r.label, slots=tuple(slots)))
        return ReserveRequest(reservations=tuple(res_reqs))

    def _solve(self, devices: list[dict]) -> ReserveRequest:
        # All reservations share the same solver (from the first one, or default).
        solver = self._reservations[0]._solver if self._reservations else ConsolidateSolver()
        specs = [(r.label, r._slots) for r in self._reservations]
        return solver.solve(specs, devices)
