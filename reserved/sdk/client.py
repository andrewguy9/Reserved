"""
client.py — Python SDK client for the reserved daemon.

Public API:
  Reservation(label, *constraints, solver=ConsolidateSolver(), socket_path=None)
  adopt_inherited_lease()
  Request(*reservations, socket_path=None)
  DeviceConstraint(type, *, name=None, ram_mb=0, compute_units=0)
  GrantedDevice(device_name)
"""

from __future__ import annotations

import array
import json
import os
import select
import socket
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from reserved import protocol
from reserved.core import ReserveRequest, cuda_uuid
from reserved.paths import client_socket_override as _get_socket_override, client_socket_paths
from reserved.sdk.errors import (
    DaemonNotRunningError,
    InheritedLeaseAlreadyAdoptedError,
    ReservationError,
    ReservationExpiredError,
)
from reserved.sdk.solvers import ConsolidateSolver

# Type alias used by solvers.
ReservationSpec = tuple[str, list]  # (label, list[DeviceConstraint])

_inherited_lease_lock = threading.Lock()
_inherited_lease_adopted = False


def _reset_inherited_lease_lock() -> None:
    # A child cannot inherit a lock held by a thread that did not survive fork.
    global _inherited_lease_lock
    _inherited_lease_lock = threading.Lock()


os.register_at_fork(after_in_child=_reset_inherited_lease_lock)


# ---------------------------------------------------------------------------
# Device constraints (in-process only, never serialized)
# ---------------------------------------------------------------------------

@dataclass
class DeviceConstraint:
    """What a reservation needs from one device; the solver picks the device.

    Every field given narrows the choice. ``type`` must match the device's
    configured type, ``name`` pins one physical device, and ``ram_mb`` and
    ``compute_units`` must fit in its available capacity. The solver turns
    each constraint into a DeviceAllocation naming the chosen device.
    """
    type: str | None = None
    name: str | None = None
    ram_mb: int = 0
    compute_units: int = 0

    def __init__(
        self,
        type: str | None = None,
        *,
        name: str | None = None,
        ram_mb: int = 0,
        compute_units: int = 0,
    ) -> None:
        self.type = type
        self.name = name
        self.ram_mb = ram_mb
        self.compute_units = compute_units


# ---------------------------------------------------------------------------
# GrantedDevice — returned to callers after acquire()
# ---------------------------------------------------------------------------

@dataclass
class GrantedDevice:
    """A physical device allocation granted by the daemon."""
    device_name: str  # Physical device name, e.g. "cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782".

    def torch_device(self) -> Any:
        """Return the process-local PyTorch device for this physical grant.

        PyTorch is expected to be installed in the calling process; this
        package does not declare or install it, and no other SDK operation
        imports it. A CUDA grant resolves to the device torch reports with
        the same GPU UUID, so the result is correct under any
        CUDA_VISIBLE_DEVICES, set or unset, and the SDK never reads or writes
        the mask. This initializes CUDA in the calling process, so call it in
        the process that will use the GPU. A grant torch cannot see, a torch
        build that does not report device UUIDs, and unsupported device kinds
        raise ReservationError.
        """
        device_type = self.device_name.partition(":")[0]
        import torch  # type: ignore[import]
        if device_type == "cuda":
            torch_name = f"cuda:{_torch_cuda_index(self.device_name, torch)}"
        elif device_type == "metal":
            torch_name = "mps"
        elif device_type == "cpu":
            torch_name = "cpu"
        else:
            raise ReservationError(f"unsupported PyTorch device {self.device_name!r}")
        return torch.device(torch_name)


def _torch_cuda_index(physical_name: str, torch_module: Any) -> int:
    """Return the index torch assigns to the CUDA device named by *physical_name*.

    torch enumerates the devices visible to this process; each reports its GPU
    UUID as 16 raw bytes. The grant's UUID is matched against them, so the
    index is right whatever CUDA_VISIBLE_DEVICES the process was started with.
    """
    try:
        wanted = cuda_uuid(physical_name)
    except ValueError as exc:
        raise ReservationError(str(exc)) from exc

    visible: list[str] = []
    for index in range(torch_module.cuda.device_count()):
        properties = torch_module.cuda.get_device_properties(index)
        raw = getattr(properties, "uuid", None)
        if raw is None:
            raise ReservationError(
                f"cannot map physical device {physical_name!r}: this torch build does "
                "not report CUDA device UUIDs"
            )
        reported = "GPU-" + str(uuid.UUID(bytes=bytes(raw.bytes)))
        if reported == wanted:
            return index
        visible.append(reported)
    raise ReservationError(
        f"physical device {physical_name!r} is not visible to torch; visible CUDA "
        f"devices: {visible} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
    )


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
    A named reservation of one or more device allocations.

    Each positional argument is a DeviceConstraint; the solver resolves every
    constraint to a device against the daemon's current status. Pin a device
    with ``DeviceConstraint(name=...)``.

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
        *constraints: DeviceConstraint,
        solver: ConsolidateSolver | None = None,
        socket_path: str | None = None,
    ) -> None:
        for constraint in constraints:
            if not isinstance(constraint, DeviceConstraint):
                raise TypeError(
                    f"Reservation takes DeviceConstraint arguments, got {constraint!r}; "
                    "pin a device with DeviceConstraint(name=...)"
                )
        self.label = label
        self._constraints = list(constraints)
        self._solver = solver if solver is not None else ConsolidateSolver()
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
        socket_path = _get_socket_override(self._socket_path)

        status = _query_status(socket_path)
        try:
            req = self._solver.solve([(self.label, self._constraints)], status["devices"])
        except ValueError as exc:
            raise ReservationError(str(exc)) from exc

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
            GrantedDevice(device_name=a["name"])
            for a in granted["allocations"]
        ]
        if fds:
            self._lease_sock = socket.socket(fileno=fds[0])
        else:
            self._lease_sock = None

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
        raise ReservationError("RESERVED_GRANTED_ALLOCATIONS is missing; launch the worker with the updated reserve CLI")

    try:
        grants = json.loads(grants_json)
        if not isinstance(grants, list):
            raise ValueError

        devices: list[GrantedDevice] = []
        for grant in grants:
            if not isinstance(grant, dict):
                raise ValueError
            physical_name = grant.get("name")
            if not isinstance(physical_name, str):
                raise ValueError
            if physical_name.partition(":")[0] == "cuda":
                cuda_uuid(physical_name)  # reject a CUDA name the driver cannot address
            devices.append(GrantedDevice(physical_name))
        return devices
    except (RecursionError, ValueError) as exc:
        raise ReservationError("RESERVED_GRANTED_ALLOCATIONS contains invalid grant metadata") from exc


def adopt_inherited_lease() -> Reservation:
    """Take ownership of the lease passed by ``reserve`` in ``RESERVED_FD``.

    Use the returned reservation as a context manager or release it explicitly.
    This claims the existing lease once, clears ``RESERVED_FD``, and marks the
    descriptor close-on-exec. The reservation carries the physical grants
    from ``RESERVED_GRANTED_ALLOCATIONS`` and the launcher ID when available.

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
        reservation.devices = _parse_inherited_grants(os.environ.get("RESERVED_GRANTED_ALLOCATIONS"))
        reservation._inherited = True
        reservation._lease_sock = _wrap_inherited_lease(fd_text)
        _inherited_lease_adopted = True
        os.environ.pop("RESERVED_FD", None)
        return reservation


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
        if any(r._inherited for r in self._reservations):
            raise ReservationError("an inherited reservation cannot be acquired again")

        socket_path = _get_socket_override(self._socket_path)

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
                GrantedDevice(device_name=a["name"])
                for a in granted["allocations"]
            ]
            res._lease_sock = socket.socket(fileno=fd)
            result.append(res)

        # If fewer fds than reservations (shouldn't happen but be safe):
        for res in self._reservations[len(fds):]:
            res._lease_sock = None

        return tuple(result)

    def _solve(self, devices: list[dict]) -> ReserveRequest:
        # All reservations share the same solver (from the first one, or default).
        solver = self._reservations[0]._solver if self._reservations else ConsolidateSolver()
        specs = [(r.label, r._constraints) for r in self._reservations]
        try:
            return solver.solve(specs, devices)
        except ValueError as exc:
            raise ReservationError(str(exc)) from exc
