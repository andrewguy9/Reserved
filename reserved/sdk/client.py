"""
client.py — Python SDK client for the reserved daemon.

Public API:
  Reservation(label, *constraints, solver=ConsolidateSolver(), socket_path=None)
  adopt_inherited_lease()
  Request(*reservations, socket_path=None)
  Reservation.popen(args, **kwargs)   # child holding a copy of the lease
  Reservation.child_env(base_env)     # launcher variables for such a child
  Reservation.lease_fd                # descriptor for a custom launcher
  DeviceConstraint(type, *, name=None, ram_mb=0, compute_units=0)
  torch_device(allocation)   # PyTorch device for one granted DeviceAllocation
  torch_cuda_uuid(properties)  # GPU-<uuid> of a torch CUDA device
"""

from __future__ import annotations

import array
import json
import os
import select
import socket
import subprocess
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reserved import protocol
from reserved.core import DeviceAllocation, ReserveRequest, cuda_uuid
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
# PyTorch device resolution for granted allocations
# ---------------------------------------------------------------------------

def torch_device(allocation: DeviceAllocation) -> Any:
    """Return the process-local PyTorch device for one granted allocation.

    PyTorch is expected to be installed in the calling process; this package
    does not declare or install it, and no other SDK operation imports it. A
    CUDA grant resolves to the device torch reports with the same GPU UUID,
    so the result is correct under any CUDA_VISIBLE_DEVICES, set or unset,
    and the SDK never reads or writes the mask. This initializes CUDA in the
    calling process, so call it in the process that will use the GPU. A
    grant torch cannot see, a torch build that does not report device UUIDs,
    and unsupported device kinds raise ReservationError.
    """
    device_type = allocation.name.partition(":")[0]
    import torch  # type: ignore[import]
    if device_type == "cuda":
        torch_name = f"cuda:{_torch_cuda_index(allocation.name, torch)}"
    elif device_type == "metal":
        torch_name = "mps"
    elif device_type == "cpu":
        torch_name = "cpu"
    else:
        raise ReservationError(f"unsupported PyTorch device {allocation.name!r}")
    return torch.device(torch_name)


def torch_cuda_uuid(properties: Any) -> str:
    """Return the ``GPU-<uuid>`` identity of a torch CUDA device from its properties.

    *properties* is what ``torch.cuda.get_device_properties(i)`` returns. The
    UUID is the one ``nvidia-smi -L`` prints and a ``cuda:`` device name
    carries. Raises ReservationError if this torch build does not report it.
    """
    raw = getattr(properties, "uuid", None)
    if raw is None:
        raise ReservationError("this torch build does not report CUDA device UUIDs")
    return "GPU-" + str(uuid.UUID(bytes=bytes(raw.bytes)))


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
        try:
            reported = torch_cuda_uuid(torch_module.cuda.get_device_properties(index))
        except ReservationError as exc:
            raise ReservationError(f"cannot map physical device {physical_name!r}: {exc}") from exc
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
            model.to(r.torch_devices()[0])
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

        # Granted allocations, set after acquisition or adoption.
        self.reservation_id: str | None = None
        self.devices: list[DeviceAllocation] = []
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
        self.devices = [_granted_allocation(a) for a in granted["allocations"]]
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
        """Return process-local PyTorch devices, one per granted allocation, in grant order."""
        return [torch_device(a) for a in self.devices]

    # ------------------------------------------------------------------
    # Handing the lease to a child process
    # ------------------------------------------------------------------

    @property
    def lease_fd(self) -> int:
        """The lease descriptor this process holds for the reservation.

        A launcher that cannot use ``popen()`` passes it in ``pass_fds`` and
        sets ``child_env()`` so the child can adopt the lease. Raises
        ReservationExpiredError before acquisition and after release.
        """
        if self._lease_sock is None:
            raise ReservationExpiredError("reservation not acquired or already released")
        return self._lease_sock.fileno()

    def child_env(self, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
        """Environment for a child that receives a copy of this lease.

        Returns a copy of ``base_env`` (``os.environ`` by default) with the
        variables the ``reserve`` launcher sets: ``RESERVED_FD``,
        ``RESERVED_ID``, ``RESERVED_GRANTED_ALLOCATIONS``, ``RESERVED_DEVICE``,
        ``RESERVED_DEVICES``, and ``CUDA_VISIBLE_DEVICES`` for CUDA grants.
        The child holds the lease only if it also inherits ``lease_fd``.
        """
        fd = self.lease_fd
        if self.reservation_id is None:
            raise ReservationError("reservation has no ID to pass to a child")
        source = os.environ if base_env is None else base_env
        return _lease_environment(source, self.devices, self.reservation_id, fd)

    def popen(
        self,
        args: Any,
        *,
        env: Mapping[str, str] | None = None,
        pass_fds: Any = (),
        **kwargs: Any,
    ) -> subprocess.Popen:
        """Start a child process that holds a copy of this lease.

        Calls ``subprocess.Popen(args, **kwargs)`` with the lease descriptor
        added to ``pass_fds`` and ``child_env(env)`` as the environment, so the
        child can call ``adopt_inherited_lease()``. Other descriptors are not
        inherited unless listed in ``pass_fds``.

        This process keeps its own copy. The daemon reclaims capacity only
        after both copies close, so ``release()`` here when the worker is done,
        or keep the lease to relaunch the worker under the same grant.
        """
        fd = self.lease_fd
        return subprocess.Popen(
            args, env=self.child_env(env), pass_fds=(*pass_fds, fd), **kwargs,
        )

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


def _lease_environment(
    base_env: Mapping[str, str],
    allocations: list[DeviceAllocation],
    reservation_id: str,
    lease_fd: int | None,
) -> dict[str, str]:
    """Describe a granted reservation to a child process through its environment.

    Shared by ``Reservation.child_env()`` and the ``reserve`` launcher so both
    produce the variables ``adopt_inherited_lease()`` reads. ``base_env`` is
    copied, not changed. Without a lease descriptor, any stale ``RESERVED_FD``
    is dropped.
    """
    physical_names = [a.name for a in allocations]
    cuda_uuids: list[str] = []
    for name in physical_names:
        if name.partition(":")[0] == "cuda":
            token = cuda_uuid(name)
            if token not in cuda_uuids:
                cuda_uuids.append(token)

    first_accel = next(
        (name for name in physical_names if not name.startswith("cpu")),
        physical_names[0] if physical_names else "",
    )

    env = dict(base_env)
    if cuda_uuids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_uuids)
    env["RESERVED_DEVICE"] = first_accel
    env["RESERVED_DEVICES"] = ",".join(physical_names)
    # Preserve allocation boundaries even when a configured name contains a comma.
    env["RESERVED_GRANTED_ALLOCATIONS"] = json.dumps(
        [protocol._allocation_to_wire(a) for a in allocations], separators=(",", ":"),
    )
    env["RESERVED_ID"] = reservation_id
    if lease_fd is not None:
        env["RESERVED_FD"] = str(lease_fd)
    else:
        env.pop("RESERVED_FD", None)
    return env


def _granted_allocation(record: Any) -> DeviceAllocation:
    """Parse one granted allocation record ({name, ram_mb, compute_units}).

    Raises ValueError for a missing or mistyped field, or a CUDA name the
    driver could not address.
    """
    if not isinstance(record, dict):
        raise ValueError("allocation record is not an object")
    name = record.get("name")
    ram_mb = record.get("ram_mb")
    compute_units = record.get("compute_units")
    if not isinstance(name, str) or type(ram_mb) is not int or type(compute_units) is not int:
        raise ValueError(f"malformed allocation record {record!r}")
    if name.partition(":")[0] == "cuda":
        cuda_uuid(name)
    return DeviceAllocation(name=name, ram_mb=ram_mb, compute_units=compute_units)


def _parse_inherited_grants(grants_json: str | None) -> list[DeviceAllocation]:
    if grants_json is None:
        raise ReservationError("RESERVED_GRANTED_ALLOCATIONS is missing; launch the worker with the updated reserve CLI")

    try:
        grants = json.loads(grants_json)
        if not isinstance(grants, list):
            raise ValueError("grant metadata is not a list")
        return [_granted_allocation(grant) for grant in grants]
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
            res.devices = [_granted_allocation(a) for a in granted["allocations"]]
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
