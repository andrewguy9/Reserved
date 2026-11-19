"""
Invariant-focused tests for SDK client helpers and lease semantics.
"""

import array
import socket
import sys
import types

import pytest

from reserved.core import ReservationRequest, ReserveRequest, DeviceAllocation
from reserved.sdk.client import (
    DeviceConstraint, GrantedDevice, Request, Reservation, _connect, acquire_reservations,
)
from reserved.sdk.solvers import NoPlacement
from reserved.sdk.errors import DaemonNotRunningError, ReservationError, ReservationExpiredError
import reserved.sdk.client as client_module


@pytest.mark.parametrize("outcome, message", [
    (NoPlacement("no room"), "no room"),
    ([], "exactly one reservation"),
    ([ReservationRequest("worker", ()), ReservationRequest("extra", ())],
     "exactly one reservation"),
    ([ReservationRequest("wrong", ())], "labels and order"),
    ([ReservationRequest("worker", (DeviceConstraint("cpu"),))],
     "explicit DeviceAllocation"),
    (object(), "reservation requests or NoPlacement"),
])
@pytest.mark.parametrize("standalone", [False, True])
def test_invalid_solver_outcomes_fail_before_acquisition(monkeypatch, outcome, message, standalone):
    monkeypatch.setattr(client_module, "_query_status", lambda _: {"devices": []})
    monkeypatch.setattr(client_module, "_send_reserve", lambda *_: pytest.fail("acquired"))
    solver = lambda *_: outcome
    worker = Reservation("worker", DeviceConstraint("cpu"))
    if standalone:
        monkeypatch.setattr(client_module, "solve_reservations_consolidated", solver)
    with pytest.raises(ReservationError, match=message):
        if standalone:
            with worker:
                pytest.fail("acquired")
        else:
            Request(worker, solver=solver).acquire()


def test_empty_acquisition_needs_no_daemon(monkeypatch):
    monkeypatch.setattr(client_module, "_connect", lambda _: pytest.fail("connected"))
    assert acquire_reservations([]) == ()
    assert Request().acquire() == ()


@pytest.mark.parametrize("concrete_input", [
    NoPlacement("no room"),
    "",
    [ReservationRequest("worker", (DeviceConstraint("cpu"),))],
])
def test_acquisition_requires_concrete_input_before_connecting(monkeypatch, concrete_input):
    monkeypatch.setattr(client_module, "_connect", lambda _: pytest.fail("connected"))
    with pytest.raises(TypeError):
        acquire_reservations(concrete_input)


@pytest.mark.parametrize("failure", ["rejection", "missing-fd", "malformed-second-grant"])
def test_failed_acquisition_closes_attached_and_unattached_lease_descriptors(monkeypatch, failure):
    leases = [socket.socketpair(), socket.socketpair()]
    descriptors = [lease.detach() for lease, _ in leases]
    grants = [{"reservation_id": name, "slots": []} for name in ("first", "second")]
    response = {"ok": True, "reservations": grants}
    if failure == "rejection":
        response = {"ok": False, "error": "rejected"}
    elif failure == "missing-fd":
        socket.close(descriptors.pop())
    else:
        grants[1] = {"slots": []}
    monkeypatch.setattr(client_module, "_send_reserve", lambda *_: (response, descriptors))
    try:
        with pytest.raises((ReservationError, KeyError)):
            acquire_reservations([
                ReservationRequest("first", ()), ReservationRequest("second", ()),
            ])
        for _, peer in leases:
            peer.settimeout(1)
            assert peer.recv(1) == b""
    finally:
        for fd in descriptors:
            try:
                socket.close(fd)
            except OSError:
                pass
        for _, peer in leases:
            peer.close()


def test_connect_translates_missing_socket_to_domain_error(monkeypatch):
    class MissingSocket:
        def connect(self, socket_path):
            raise FileNotFoundError(socket_path)

        def close(self):
            pass

    monkeypatch.setattr(client_module.socket, "socket", lambda *args: MissingSocket())

    with pytest.raises(DaemonNotRunningError, match="not found"):
        _connect("/missing.sock")


def test_malformed_wire_response_closes_received_lease_descriptor(monkeypatch):
    transport, server = socket.socketpair()
    lease, peer = socket.socketpair()
    monkeypatch.setattr(client_module, "_connect", lambda _: transport)
    try:
        server.sendmsg([b"not-json\n"], [
            (socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [lease.fileno()])),
        ])
        lease.close()  # Only the descriptor queued in the message holds this end open.
        with pytest.raises(ValueError, match="invalid JSON response"):
            client_module._send_reserve(None, ReserveRequest((ReservationRequest("job", ()),)), 1)
        peer.settimeout(1)
        assert peer.recv(1) == b""
    finally:
        transport.close()
        server.close()
        lease.close()
        peer.close()


def test_cpu_and_metal_pytorch_devices_ignore_cuda_visibility(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(device=str))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    assert GrantedDevice("metal:0", -1).torch_device() == "mps"
    assert GrantedDevice("cpu", -1).torch_device() == "cpu"


@pytest.mark.parametrize(("cuda_visibility", "logical_device"), [
    pytest.param(None, "cuda:2", id="unmasked"),
    pytest.param("2", "cuda:0", id="only-physical-2-visible"),
    pytest.param("5,2", "cuda:1", id="physical-5-visible-first"),
])
def test_same_physical_cuda_grant_maps_to_the_process_local_device(
    monkeypatch, cuda_visibility, logical_device,
):
    if cuda_visibility is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_visibility)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(device=str))

    physical_grant = GrantedDevice(device_name="cuda:2", cuda_ordinal=2)

    assert physical_grant.torch_device() == logical_device
    assert physical_grant.device_name == "cuda:2"
    assert physical_grant.cuda_ordinal == 2


@pytest.mark.parametrize(("mask", "reason"), [
    ("", "not visible"),
    ("-1", "not visible"),
    ("0,5", "not visible"),
    ("GPU-abc", "invalid"),
    ("MIG-abc", "invalid"),
    ("2,", "invalid"),
    ("2,-1,5", "invalid"),
    ("2,02", "invalid"),
    ("2147483648,2", "invalid"),
    ("٢,2", "invalid"),
])
def test_granted_device_rejects_hidden_or_ambiguous_cuda_mapping(monkeypatch, mask, reason):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(device=str))

    with pytest.raises(ReservationError, match=reason) as error:
        GrantedDevice("cuda:2", 2).torch_device()

    assert "'cuda:2'" in str(error.value)
    assert f"CUDA_VISIBLE_DEVICES={mask!r}" in str(error.value)


def test_granted_device_rejects_unsupported_torch_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(device=str))

    with pytest.raises(ReservationError, match="fpga:0"):
        GrantedDevice("fpga:0", -1).torch_device()


def test_reservation_release_closes_lease_socket_and_is_idempotent():
    client_sock, daemon_sock = socket.socketpair()
    try:
        r = Reservation("job")
        r._lease_sock = client_sock

        r.release()
        r.release()

        assert r._lease_sock is None
        assert client_sock.fileno() == -1
    finally:
        daemon_sock.close()


def test_check_raises_when_reservation_has_not_been_acquired():
    with pytest.raises(ReservationExpiredError, match="not acquired"):
        Reservation("job").check()


def test_check_raises_when_daemon_side_of_lease_is_closed():
    client_sock, daemon_sock = socket.socketpair()
    r = Reservation("job")
    r.reservation_id = "abc123"
    r._lease_sock = client_sock

    daemon_sock.close()
    try:
        with pytest.raises(ReservationExpiredError, match="abc123"):
            r.check()
    finally:
        r.release()
