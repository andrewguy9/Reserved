"""
Invariant-focused tests for SDK client helpers and lease semantics.
"""

import socket
import sys
import types

import pytest

from reserved.sdk.client import GrantedDevice, Reservation, _connect
from reserved.sdk.errors import DaemonNotRunningError, ReservationError, ReservationExpiredError
import reserved.sdk.client as client_module
from tests.cuda_support import cuda_name, fake_torch

CUDA0, CUDA1, CUDA2, CUDA5 = (cuda_name(n) for n in (0, 1, 2, 5))


def test_connect_translates_missing_socket_to_domain_error(monkeypatch):
    class MissingSocket:
        def connect(self, socket_path):
            raise FileNotFoundError(socket_path)

        def close(self):
            pass

    monkeypatch.setattr(client_module.socket, "socket", lambda *args: MissingSocket())

    with pytest.raises(DaemonNotRunningError, match="not found"):
        _connect("/missing.sock")


def test_cpu_and_metal_pytorch_devices_do_not_touch_torch_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(device=str))

    assert GrantedDevice("metal:0").torch_device() == "mps"
    assert GrantedDevice("cpu").torch_device() == "cpu"


@pytest.mark.parametrize(("visible", "logical_device"), [
    pytest.param((CUDA0, CUDA1, CUDA2), "cuda:2", id="all-visible"),
    pytest.param((CUDA2,), "cuda:0", id="only-granted-visible"),
    pytest.param((CUDA5, CUDA2), "cuda:1", id="reordered-mask"),
])
def test_cuda_grant_maps_to_the_torch_device_with_its_uuid(monkeypatch, visible, logical_device):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(*visible))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "not-consulted-by-the-sdk")

    grant = GrantedDevice(device_name=CUDA2)

    assert grant.torch_device() == logical_device
    assert grant.device_name == CUDA2


def test_cuda_grant_hidden_from_torch_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(CUDA0, CUDA1))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    with pytest.raises(ReservationError, match="not visible") as error:
        GrantedDevice(CUDA2).torch_device()

    assert repr(CUDA2) in str(error.value)
    assert "CUDA_VISIBLE_DEVICES='0,1'" in str(error.value)


def test_torch_without_device_uuids_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(CUDA2, with_uuid=False))

    with pytest.raises(ReservationError, match="does not report CUDA device UUIDs"):
        GrantedDevice(CUDA2).torch_device()


def test_granted_device_rejects_malformed_cuda_name(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(CUDA2))

    with pytest.raises(ReservationError, match="invalid CUDA device name"):
        GrantedDevice("cuda:2").torch_device()


def test_granted_device_rejects_unsupported_torch_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(device=str))

    with pytest.raises(ReservationError, match="fpga:0"):
        GrantedDevice("fpga:0").torch_device()


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
