"""
Invariant-focused tests for SDK client helpers and lease semantics.
"""

import json
import os
import socket
import sys
import types

import pytest

from reserved.core import DeviceAllocation
from reserved.sdk.client import Reservation, _connect, _python_environment, torch_device
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

    assert torch_device(DeviceAllocation("metal:0")) == "mps"
    assert torch_device(DeviceAllocation("cpu")) == "cpu"


@pytest.mark.parametrize(("visible", "logical_device"), [
    pytest.param((CUDA0, CUDA1, CUDA2), "cuda:2", id="all-visible"),
    pytest.param((CUDA2,), "cuda:0", id="only-granted-visible"),
    pytest.param((CUDA5, CUDA2), "cuda:1", id="reordered-mask"),
])
def test_cuda_grant_maps_to_the_torch_device_with_its_uuid(monkeypatch, visible, logical_device):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(*visible))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "not-consulted-by-the-sdk")

    grant = DeviceAllocation(CUDA2, ram_mb=1024)

    assert torch_device(grant) == logical_device


def test_cuda_grant_hidden_from_torch_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(CUDA0, CUDA1))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    with pytest.raises(ReservationError, match="not visible") as error:
        torch_device(DeviceAllocation(CUDA2))

    assert repr(CUDA2) in str(error.value)
    assert "CUDA_VISIBLE_DEVICES='0,1'" in str(error.value)


def test_torch_without_device_uuids_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(CUDA2, with_uuid=False))

    with pytest.raises(ReservationError, match="does not report CUDA device UUIDs"):
        torch_device(DeviceAllocation(CUDA2))


def test_torch_device_rejects_malformed_cuda_name(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(CUDA2))

    with pytest.raises(ReservationError, match="invalid CUDA device name"):
        torch_device(DeviceAllocation("cuda:2"))


def test_torch_device_rejects_unsupported_torch_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(device=str))

    with pytest.raises(ReservationError, match="fpga:0"):
        torch_device(DeviceAllocation("fpga:0"))


def test_reservation_rejects_arguments_that_are_not_constraints():
    with pytest.raises(TypeError, match="DeviceConstraint"):
        Reservation("job", DeviceAllocation("cpu", ram_mb=1024))


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


def _held_reservation(reservation_id="abc123", devices=()):
    client_sock, daemon_sock = socket.socketpair()
    r = Reservation("job")
    r.reservation_id = reservation_id
    r.devices = list(devices)
    r._lease_sock = client_sock
    return r, daemon_sock


def test_child_env_describes_the_lease_without_mutating_the_base_env():
    r, daemon_sock = _held_reservation(devices=[
        DeviceAllocation("cpu:batch,worker", ram_mb=1024),
        DeviceAllocation(CUDA2, ram_mb=4096),
        DeviceAllocation(CUDA2, ram_mb=2048),
        DeviceAllocation("metal:0", ram_mb=512),
        DeviceAllocation(CUDA5, ram_mb=4096, compute_units=7),
    ])
    base = {"KEEP": "yes", "RESERVED_FD": "stale"}
    with daemon_sock:
        try:
            env = r.child_env(base)
        finally:
            fd = r.lease_fd
            r.release()

    assert base == {"KEEP": "yes", "RESERVED_FD": "stale"}
    assert env["KEEP"] == "yes"
    assert env["RESERVED_FD"] == str(fd)
    assert env["RESERVED_ID"] == "abc123"
    assert env["RESERVED_DEVICE"] == CUDA2
    assert env["RESERVED_DEVICES"] == f"cpu:batch,worker,{CUDA2},{CUDA2},metal:0,{CUDA5}"
    assert env["CUDA_VISIBLE_DEVICES"] == f"{CUDA2.partition(':')[2]},{CUDA5.partition(':')[2]}"
    assert json.loads(env["RESERVED_GRANTED_ALLOCATIONS"]) == [
        {"name": "cpu:batch,worker", "ram_mb": 1024, "compute_units": 0},
        {"name": CUDA2, "ram_mb": 4096, "compute_units": 0},
        {"name": CUDA2, "ram_mb": 2048, "compute_units": 0},
        {"name": "metal:0", "ram_mb": 512, "compute_units": 0},
        {"name": CUDA5, "ram_mb": 4096, "compute_units": 7},
    ]


def test_child_env_defaults_to_the_process_environment(monkeypatch):
    monkeypatch.setenv("FROM_PARENT", "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    r, daemon_sock = _held_reservation(devices=[DeviceAllocation("cpu", ram_mb=256)])
    with daemon_sock:
        try:
            env = r.child_env()
        finally:
            r.release()

    assert env["FROM_PARENT"] == "1"
    assert env["RESERVED_DEVICE"] == "cpu"
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "RESERVED_FD" not in os.environ


def test_lease_fd_and_child_env_require_a_held_lease():
    r = Reservation("job")
    with pytest.raises(ReservationExpiredError, match="not acquired"):
        r.lease_fd
    with pytest.raises(ReservationExpiredError, match="not acquired"):
        r.child_env({})
    with pytest.raises(ReservationExpiredError, match="not acquired"):
        r.popen([sys.executable, "-c", "pass"])

    r, daemon_sock = _held_reservation()
    with daemon_sock:
        assert r.lease_fd == r._lease_sock.fileno()
        r.release()
        with pytest.raises(ReservationExpiredError, match="released"):
            r.child_env({})


def test_child_env_requires_a_reservation_id():
    r, daemon_sock = _held_reservation(reservation_id=None)
    with daemon_sock:
        try:
            with pytest.raises(ReservationError, match="no ID"):
                r.child_env({})
        finally:
            r.release()


def test_python_environment_activates_the_worker_environment_without_mutating_base():
    base = {"PATH": "/orchestrator/bin:/usr/bin", "VIRTUAL_ENV": "/orchestrator",
            "PYTHONPATH": "/orchestrator/src", "PYTHONHOME": "/elsewhere", "KEEP": "yes"}

    env = _python_environment(base, "/workers/tts-env")

    assert base["PYTHONPATH"] == "/orchestrator/src"
    assert env == {"PATH": "/workers/tts-env/bin:/orchestrator/bin:/usr/bin",
                   "VIRTUAL_ENV": "/workers/tts-env", "KEEP": "yes"}


def test_python_environment_without_a_base_path_uses_only_the_worker_bin():
    assert _python_environment({}, "/workers/tts-env") == {
        "PATH": "/workers/tts-env/bin", "VIRTUAL_ENV": "/workers/tts-env",
    }


def test_popen_python_requires_a_held_lease_before_touching_the_filesystem(tmp_path):
    with pytest.raises(ReservationExpiredError, match="not acquired"):
        Reservation("job").popen_python(tmp_path, "model_worker")
