"""
Invariant-focused tests for SDK client helpers and lease semantics.
"""

import socket
import sys
import types

import pytest

from reserved.sdk.client import GrantedDevice, Reservation, _connect, _get_socket_path
from reserved.sdk.errors import DaemonNotRunningError, ReservationExpiredError
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


def test_granted_device_torch_device_maps_reserved_device_types(monkeypatch):
    calls = []
    fake_torch = types.SimpleNamespace(device=lambda value: calls.append(value) or value)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert GrantedDevice("cuda:2", 2).torch_device() == "cuda:2"
    assert GrantedDevice("metal:0", -1).torch_device() == "mps"
    assert GrantedDevice("cpu", -1).torch_device() == "cpu"
    assert calls == ["cuda:2", "mps", "cpu"]


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
