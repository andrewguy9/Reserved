"""
client.py — Python SDK client for the reserved daemon.

Public API:
  Reservation(label, *slots, socket_path=None)
  adopt_inherited_lease()
  Request(*reservations, solver=None, socket_path=None)
  acquire_reservations(reservation_requests, *, socket_path=None)
  ReservationSpec(label, requirements)
  DeviceConstraint(type, *, name=None, model=None, ram_mb=0, compute_units=0)
  ExplicitSlot(name, *, ram_mb=0, compute_units=0)
  GrantedDevice(device_name, cuda_ordinal)
"""

from __future__ import annotations

import array
import json
import os
import select
import socket
import threading
from dataclasses import dataclass, field
from typing import Any, Sequence

from reserved import protocol
from reserved.core import (
    ReservationRequest,
    ReserveRequest,
    SlotRequest,
)
from reserved.paths import client_socket_override as _get_socket_override, client_socket_paths
from reserved.sdk.errors import (
    DaemonNotRunningError,
    InheritedLeaseAlreadyAdoptedError,
    ReservationError,
    ReservationExpiredError,
)
from reserved.sdk.solvers import (
    NoPlacement, ReservationSolver, solve_reservations_consolidated,
)

_inherited_lease_lock = threading.Lock()
_inherited_lease_adopted = False


def _reset_inherited_lease_lock() -> None:
    # A child cannot inherit a lock held by a thread that did not survive fork.
    global _inherited_lease_lock
    _inherited_lease_lock = threading.Lock()


os.register_at_fork(after_in_child=_reset_inherited_lease_lock)


# ---------------------------------------------------------------------------
# Slot descriptor types (in-process only, never serialized)
# ---------------------------------------------------------------------------

@dataclass
class DeviceConstraint:
    """Device requirements, optionally pinned to one physical device by name."""
    type: str | None = None
    name: str | None = None
    model: str | None = None
    ram_mb: int = 0
    compute_units: int = 0

    def __init__(
        self,
        type: str | None = None,
        *,
        name: str | None = None,
        model: str | None = None,
        ram_mb: int = 0,
        compute_units: int = 0,
    ) -> None:
        self.type = type
        self.name = name
        self.model = model
        self.ram_mb = ram_mb
        self.compute_units = compute_units


def ExplicitSlot(name: str, *, ram_mb: int = 0, compute_units: int = 0) -> DeviceConstraint:
    """Compatibility constructor for DeviceConstraint(name=...)."""
    return DeviceConstraint(name=name, ram_mb=ram_mb, compute_units=compute_units)


@dataclass(frozen=True)
class ReservationSpec:
    """SDK placement input for one reservation; its label is carried unchanged."""

    label: str
    requirements: tuple[DeviceConstraint, ...]


# ---------------------------------------------------------------------------
# GrantedDevice — returned to callers after acquire()
# ---------------------------------------------------------------------------

@dataclass
class GrantedDevice:
    """A physical device slot granted by the daemon."""
    device_name: str  # Physical device name.
    cuda_ordinal: int  # Physical CUDA ordinal, or -1 for non-CUDA devices.

    def torch_device(self) -> Any:
        """Return the process-local PyTorch device for this physical grant.

        CUDA uses the grant's logical index within CUDA_VISIBLE_DEVICES. Set
        the mask before CUDA initializes. Hidden devices, ambiguous masks, and
        unsupported device kinds raise ReservationError.
        """
        device_type = self.device_name.partition(":")[0]
        if self.cuda_ordinal >= 0:
            logical_index = _logical_cuda_index(
                self.device_name, self.cuda_ordinal, os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
            torch_name = f"cuda:{logical_index}"
        elif device_type == "metal":
            torch_name = "mps"
        elif device_type == "cpu":
            torch_name = "cpu"
        else:
            raise ReservationError(f"unsupported PyTorch device {self.device_name!r}")

        import torch  # type: ignore[import]
        return torch.device(torch_name)


def _logical_cuda_index(physical_name: str, physical_ordinal: int, mask: str | None) -> int:
    if mask is None:
        return physical_ordinal

    visible_ordinals: list[int] = []
    if mask not in ("", "-1"):
        tokens = mask.split(",")
        try:
            if not all(token.isascii() and token.isdecimal() for token in tokens):
                raise ValueError
            visible_ordinals = [int(token) for token in tokens]
            if max(visible_ordinals) > 2**31 - 1 or len(set(visible_ordinals)) != len(tokens):
                raise ValueError
        except ValueError as exc:
            raise ReservationError(
                f"cannot map physical device {physical_name!r}: invalid "
                f"CUDA_VISIBLE_DEVICES={mask!r}; expected distinct numeric CUDA ordinals"
            ) from exc

    try:
        return visible_ordinals.index(physical_ordinal)
    except ValueError as exc:
        raise ReservationError(
            f"physical device {physical_name!r} is not visible in CUDA_VISIBLE_DEVICES={mask!r}"
        ) from exc


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

def _connect(socket_path: str | None) -> socket.socket:
    """Connect to an explicit target or the first available default socket."""
    errors: list[str] = []
    last_error: OSError | None = None
    for path in client_socket_paths(socket_path):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(path)
        except OSError as exc:
            sock.close()
            if isinstance(exc, FileNotFoundError):
                errors.append(f"daemon socket not found: {path!r}")
            elif isinstance(exc, ConnectionRefusedError):
                errors.append(f"daemon not running (connection refused): {path!r}")
            else:
                raise
            last_error = exc
        else:
            return sock
    raise DaemonNotRunningError("; ".join(errors)) from last_error


def _query_status(socket_path: str | None) -> dict:
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
    socket_path: str | None,
    request: ReserveRequest,
    num_reservations: int,
) -> tuple[dict, list[int]]:
    """
    Send a reserve request and receive the response plus lease fds.

    Returns (response_dict, list_of_fds).
    """
    sock = _connect(socket_path)
    received_fds: list[int] = []
    try:
        wire = protocol.reserve_request_to_wire(request)
        sock.sendall(wire)

        # Receive response + SCM_RIGHTS fds.
        # We may need to call recvmsg multiple times if the response is large.
        buf = b""

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
    except BaseException:
        for fd in received_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        raise
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
        *slots: DeviceConstraint,
        socket_path: str | None = None,
    ) -> None:
        self.label = label
        self._slots = list(slots)
        self._socket_path = socket_path

        # Physical grant metadata set after acquisition or adoption.
        self.reservation_id: str | None = None
        self.devices: list[GrantedDevice] = []
        self._lease_sock: socket.socket | None = None
        self._inherited = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Reservation":
        if self._inherited:
            if self._lease_sock is None:
                raise ReservationExpiredError("inherited reservation has already been released")
        else:
            self._acquire_single()
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Acquire (called by Request.acquire() or __enter__)
    # ------------------------------------------------------------------

    def _acquire_single(self) -> None:
        """Acquire this reservation by itself (not part of a multi-reservation Request)."""
        Request(self, socket_path=self._socket_path).acquire()

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
        """Return process-local PyTorch devices in physical grant order."""
        return [d.torch_device() for d in self.devices]

    def __repr__(self) -> str:
        return (
            f"Reservation(label={self.label!r}, "
            f"id={self.reservation_id!r}, "
            f"devices={self.devices!r})"
        )


def _wrap_inherited_lease(fd_text: str) -> socket.socket:
    if not fd_text.isascii() or not fd_text.isdecimal():
        raise ReservationError("RESERVED_FD must be a non-negative decimal file descriptor")

    try:
        lease_sock = socket.socket(fileno=int(fd_text))
    except (OSError, OverflowError, ValueError) as exc:
        raise ReservationError("RESERVED_FD does not name an open socket") from exc

    accepted = False
    try:
        if (lease_sock.family, lease_sock.type) != (socket.AF_UNIX, socket.SOCK_STREAM):
            raise ReservationError("RESERVED_FD must name a Unix stream socket")
        try:
            lease_sock.getpeername()
        except OSError:
            # macOS can lose the peer address after EOF. Let check() report expiry.
            if lease_sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) != b"":
                raise
        lease_sock.set_inheritable(False)
        accepted = True
        return lease_sock
    except OSError as exc:
        raise ReservationError("cannot adopt RESERVED_FD as a lease socket") from exc
    finally:
        if not accepted:
            # A rejected descriptor still belongs to its original owner.
            lease_sock.detach()


def _parse_inherited_grants(grants_json: str | None) -> list[GrantedDevice]:
    if grants_json is None:
        raise ReservationError("RESERVED_GRANTED_SLOTS is missing; launch the worker with the updated reserve CLI")

    try:
        slots = json.loads(grants_json)
        if not isinstance(slots, list):
            raise ValueError

        devices: list[GrantedDevice] = []
        for slot in slots:
            if not isinstance(slot, dict):
                raise ValueError
            physical_name = slot.get("device_name")
            physical_ordinal = slot.get("cuda_ordinal")
            if not isinstance(physical_name, str) or type(physical_ordinal) is not int:
                raise ValueError
            if physical_ordinal < -1:
                raise ValueError
            if physical_ordinal >= 0:
                kind, _, ordinal_text = physical_name.partition(":")
                if kind != "cuda" or int(ordinal_text) != physical_ordinal:
                    raise ValueError
            devices.append(GrantedDevice(physical_name, physical_ordinal))
        return devices
    except (RecursionError, ValueError) as exc:
        raise ReservationError("RESERVED_GRANTED_SLOTS contains invalid physical grant metadata") from exc


def adopt_inherited_lease() -> Reservation:
    """Take ownership of the lease passed by ``reserve`` in ``RESERVED_FD``.

    Use the returned reservation as a context manager or release it explicitly.
    This claims the existing lease once, clears ``RESERVED_FD``, and marks the
    descriptor close-on-exec. The reservation carries the physical grants
    from ``RESERVED_GRANTED_SLOTS`` and the launcher ID when available.

    Raises ReservationError for a missing or invalid descriptor or grant
    metadata, and InheritedLeaseAlreadyAdoptedError if the handoff was already
    consumed.
    """
    global _inherited_lease_adopted
    with _inherited_lease_lock:
        if _inherited_lease_adopted:
            raise InheritedLeaseAlreadyAdoptedError("inherited lease has already been adopted")

        fd_text = os.environ.get("RESERVED_FD")
        if fd_text is None:
            raise ReservationError("RESERVED_FD is missing; launch the worker with reserve")

        reservation = Reservation("inherited")
        reservation.reservation_id = os.environ.get("RESERVED_ID")
        reservation.devices = _parse_inherited_grants(os.environ.get("RESERVED_GRANTED_SLOTS"))
        reservation._inherited = True
        reservation._lease_sock = _wrap_inherited_lease(fd_text)
        _inherited_lease_adopted = True
        os.environ.pop("RESERVED_FD", None)
        return reservation


# ---------------------------------------------------------------------------
# Request — multi-reservation atomic acquire
# ---------------------------------------------------------------------------

class Request:
    """Solve and atomically acquire a batch, preserving its SDK handles.

    Supply a solver function once on Request, including for a single reservation.
    The default is consolidated; member reservations contain no solver policy.
    """

    def __init__(
        self,
        *reservations: Reservation,
        solver: ReservationSolver | None = None,
        socket_path: str | None = None,
    ) -> None:
        self._reservations = list(reservations)
        self._solver = solver if solver is not None else solve_reservations_consolidated
        self._socket_path = socket_path

    def acquire(self) -> tuple[Reservation, ...]:
        """Return the acquired handles in order, or raise ReservationError.

        Placement uses a status snapshot; the daemon rechecks current capacity.
        Each acquired reservation has its own independently releasable lease.
        """
        if not self._reservations:
            return ()
        if any(r._inherited for r in self._reservations):
            raise ReservationError("an inherited reservation cannot be acquired again")
        socket_path = _get_socket_override(self._socket_path)
        capacity = _query_status(socket_path)["devices"]
        reservation_specs = [
            ReservationSpec(r.label, tuple(r._slots)) for r in self._reservations
        ]
        placement = self._solver(capacity, reservation_specs)
        if isinstance(placement, NoPlacement):
            raise ReservationError(placement.reason)
        return _acquire_into(placement, self._reservations, socket_path)


def acquire_reservations(
    reservation_requests: Sequence[ReservationRequest],
    *,
    socket_path: str | None = None,
) -> tuple[Reservation, ...]:
    """Acquire a concrete batch without querying status or running a solver.

    The daemon grants all reservations or rejects the batch, including stale
    placements. Results retain request order and have independent leases.
    Call release() in a finally block after use.
    """
    if (
        not isinstance(reservation_requests, Sequence)
        or isinstance(reservation_requests, (str, bytes))
        or any(not isinstance(request, ReservationRequest) for request in reservation_requests)
    ):
        raise TypeError("acquisition requires concrete ReservationRequest records")
    reservations = []
    for reservation_request in reservation_requests:
        assignments = []
        for assignment in reservation_request.slots:
            if not isinstance(assignment, SlotRequest):
                raise TypeError("acquisition requires explicit SlotRequest assignments")
            assignments.append(DeviceConstraint(
                name=assignment.name, ram_mb=assignment.ram_mb,
                compute_units=assignment.compute_units,
            ))
        reservations.append(Reservation(reservation_request.label, *assignments))
    return _acquire_into(reservation_requests, reservations, _get_socket_override(socket_path))


def _acquire_into(
    reservation_requests: Sequence[ReservationRequest],
    reservations: Sequence[Reservation],
    socket_path: str | None,
) -> tuple[Reservation, ...]:
    """Acquire explicit requests into SDK handles; own received fds until attached."""
    if (
        not isinstance(reservation_requests, Sequence)
        or isinstance(reservation_requests, (str, bytes))
        or any(not isinstance(request, ReservationRequest) for request in reservation_requests)
    ):
        raise ReservationError("solver must return reservation requests or NoPlacement")
    if len(reservation_requests) != len(reservations):
        raise ReservationError("solver must return exactly one reservation per specification")
    if any(req.label != res.label for req, res in zip(reservation_requests, reservations)):
        raise ReservationError("solver must preserve reservation labels and order")
    if any(not isinstance(assignment, SlotRequest)
           for req in reservation_requests for assignment in req.slots):
        raise ReservationError("solver must return explicit SlotRequest assignments")
    if not reservations:
        return ()

    request = ReserveRequest(tuple(reservation_requests))
    response, pending_fds = _send_reserve(socket_path, request, len(reservations))
    acquired = []
    try:
        if not response.get("ok"):
            raise ReservationError(response.get("error", "reservation rejected"))
        grants = response["reservations"]
        if len(grants) != len(reservations) or len(pending_fds) != len(reservations):
            raise ReservationError("daemon returned an incomplete reservation batch")
        for reservation, grant in zip(reservations, grants):
            reservation_id = grant["reservation_id"]
            devices = [GrantedDevice(s["device_name"], s["cuda_ordinal"])
                       for s in grant["slots"]]
            lease = socket.socket(fileno=pending_fds[0])
            pending_fds.pop(0)  # Ownership moves from the raw fd list to this handle.
            reservation.reservation_id = reservation_id
            reservation.devices = devices
            reservation._lease_sock = lease
            acquired.append(reservation)
        return tuple(acquired)
    except BaseException:
        for reservation in acquired:
            reservation.release()
        raise
    finally:
        for fd in pending_fds:
            try:
                os.close(fd)
            except OSError:
                pass
