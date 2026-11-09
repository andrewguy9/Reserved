"""
Invariant-focused tests for SDK client helpers and lease semantics.
"""

import socket
import sys
import types

import pytest

from reserved.sdk.client import GrantedDevice, Reservation, _connect, _get_socket_path
from reserved.sdk.errors import DaemonNotRunningError, ReservationError, ReservationExpiredError
import reserved.sdk.client as client_module


def test_get_socket_path_precedence(monkeypatch):
    monkeypatch.setenv("RESERVED_SOCKET", "/env/reserved.sock")

    assert _get_socket_path("/override.sock") == "/override.sock"
    assert _get_socket_path(None) == "/env/reserved.sock"

    monkeypatch.delenv("RESERVED_SOCKET")
    assert _get_socket_path(None) == "/var/run/reserved.sock"


def test_connect_translates_missing_socket_to_domain_error(monkeypatch):
    class MissingSocket:
        def connect(self, socket_path):
            raise FileNotFoundError(socket_path)

    monkeypatch.setattr(client_module.socket, "socket", lambda *args: MissingSocket())

    with pytest.raises(DaemonNotRunningError, match="not found"):
        _connect("/missing.sock")


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
