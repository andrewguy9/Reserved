"""Exercise the SDK handoff with Unix sockets and the real reserve launcher."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from reserved import sdk
from reserved.sdk import client


@pytest.fixture
def handoff(monkeypatch):
    monkeypatch.setattr(client, "_inherited_lease_adopted", False)
    monkeypatch.delenv("RESERVED_FD", raising=False)
    monkeypatch.setenv("RESERVED_GRANTED_SLOTS", "[]")
    monkeypatch.setenv("RESERVED_ID", "inherited-id")

    def publish(fd):
        monkeypatch.setenv("RESERVED_FD", str(fd))

    return publish


def test_missing_and_malformed_descriptors(handoff):
    with pytest.raises(sdk.ReservationError, match="RESERVED_FD is missing"):
        sdk.adopt_inherited_lease()

    for malformed in ("", "not-a-descriptor", "-1", "1_2", "3" * 30):
        handoff(malformed)
        with pytest.raises(sdk.ReservationError, match="RESERVED_FD"):
            sdk.adopt_inherited_lease()


def test_rejects_closed_or_unrelated_descriptors(handoff):
    fd = os.open(os.devnull, os.O_RDONLY)
    os.close(fd)
    handoff(fd)
    with pytest.raises(sdk.ReservationError, match="open socket"):
        sdk.adopt_inherited_lease()

    with open(os.devnull, "rb") as unrelated:
        handoff(unrelated.fileno())
        with pytest.raises(sdk.ReservationError, match="open socket"):
            sdk.adopt_inherited_lease()
        assert unrelated.read() == b""

    for socket_type in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
        with socket.socket(socket.AF_UNIX, socket_type) as unrelated:
            handoff(unrelated.fileno())
            with pytest.raises(sdk.ReservationError, match="RESERVED_FD"):
                sdk.adopt_inherited_lease()
            assert unrelated.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == socket_type


@pytest.mark.parametrize("disconnect_before_adoption", [False, True])
def test_adopted_lease_reports_daemon_disconnect(handoff, disconnect_before_adoption):
    worker, daemon = socket.socketpair()
    with worker, daemon:
        fd = worker.detach()
        handoff(fd)
        if disconnect_before_adoption:
            daemon.close()
        try:
            reservation = sdk.adopt_inherited_lease()
        except BaseException:
            os.close(fd)
            raise
        with reservation:
            assert reservation.devices == []  # The handoff explicitly grants zero slots.
            if not disconnect_before_adoption:
                reservation.check()
                daemon.close()
            with pytest.raises(sdk.ReservationExpiredError, match="inherited-id"):
                reservation.check()


def test_missing_grant_metadata_leaves_inherited_descriptor_unclaimed(handoff, monkeypatch):
    worker, daemon = socket.socketpair()
    with worker, daemon:
        fd = worker.fileno()
        worker.set_inheritable(True)
        handoff(fd)
        monkeypatch.delenv("RESERVED_GRANTED_SLOTS")

        with pytest.raises(sdk.ReservationError, match="RESERVED_GRANTED_SLOTS is missing"):
            adopted = sdk.adopt_inherited_lease()
            worker.detach()
            adopted.release()

        assert os.environ["RESERVED_FD"] == str(fd)
        assert worker.get_inheritable()


def test_invalid_grant_metadata_leaves_inherited_descriptor_unclaimed(handoff, monkeypatch):
    worker, daemon = socket.socketpair()
    with worker, daemon:
        fd = worker.fileno()
        handoff(fd)
        for malformed in (
            "", "null", "[{}]",
            '[{"device_name":"cuda:2","cuda_ordinal":true}]',
            '[{"device_name":"cuda:2","cuda_ordinal":5}]',
        ):
            monkeypatch.setenv("RESERVED_GRANTED_SLOTS", malformed)
            with pytest.raises(sdk.ReservationError, match="RESERVED_GRANTED_SLOTS"):
                sdk.adopt_inherited_lease()
            assert os.environ["RESERVED_FD"] == str(fd)
            os.fstat(fd)


def test_concurrent_adoption_has_one_owner(handoff):
    worker, daemon = socket.socketpair()
    with worker, daemon:
        handoff(worker.detach())
        ready = Barrier(2, timeout=5)

        def adopt():
            ready.wait()
            return sdk.adopt_inherited_lease()

        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = [pool.submit(adopt) for _ in range(2)]
            reservations = []
            errors = []
            for attempt in attempts:
                try:
                    reservations.append(attempt.result(timeout=5))
                except sdk.InheritedLeaseAlreadyAdoptedError as exc:
                    errors.append(exc)

        try:
            assert len(reservations) == len(errors) == 1
            reservations[0].check()
        finally:
            for reservation in reservations:
                reservation.release()
        daemon.settimeout(1)
        assert daemon.recv(1) == b""


@pytest.fixture
def daemon_env():
    # Keep the Unix socket path short enough for macOS and Linux.
    with tempfile.TemporaryDirectory(prefix="reserved-test-", dir="/tmp") as directory:
        socket_path = str(Path(directory) / "daemon.sock")
        config = Path(directory) / "devices.json"
        config.write_text(json.dumps({"devices": [
            {"name": "cpu", "type": "cpu", "ram_mb": 1024},
            {"name": "cuda:2", "type": "cuda", "ram_mb": 1024},
            {"name": "cuda:5", "type": "cuda", "ram_mb": 1024},
        ]}))
        env = {**os.environ, "RESERVED_SOCKET": socket_path}
        env.pop("LISTEN_FDS", None)
        with subprocess.Popen(
            [sys.executable, "-m", "reserved.daemon", "--config", str(config)],
            env=env, stderr=subprocess.PIPE, text=True,
        ) as daemon:
            try:
                deadline = time.monotonic() + 5
                while True:
                    try:
                        client._query_status(socket_path)
                        break
                    except sdk.DaemonNotRunningError:
                        if daemon.poll() is not None:
                            pytest.fail(daemon.communicate()[1])
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.01)
                yield env
            finally:
                daemon.terminate()
                daemon.communicate(timeout=5)


@pytest.mark.parametrize("mode", ["normal", "exception"])
def test_adopts_and_releases_real_launcher_lease(daemon_env, mode):
    result = subprocess.run(
        [sys.executable, "-m", "reserved.bin.reserve", "worker", "name=cpu,ram=1024",
         "--", sys.executable, "-m", "tests.test_inherited_lease", mode],
        env=daemon_env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_sdk_keeps_physical_grant_order_with_reordered_cuda_visibility(daemon_env, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,2")
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(device=str))
    with sdk.Reservation(
        "direct", sdk.ExplicitSlot("cuda:2"), sdk.ExplicitSlot("cuda:5"),
        sdk.ExplicitSlot("cuda:2"), socket_path=daemon_env["RESERVED_SOCKET"],
    ) as lease:
        assert lease.devices == [
            sdk.GrantedDevice("cuda:2", 2), sdk.GrantedDevice("cuda:5", 5),
            sdk.GrantedDevice("cuda:2", 2),
        ]
        assert lease.torch_devices() == ["cuda:1", "cuda:0", "cuda:1"]


def test_launcher_hands_physical_cuda_2_to_worker_as_logical_cuda_0(daemon_env):
    result = subprocess.run(
        [sys.executable, "-m", "reserved.bin.reserve", "worker", "name=cuda:2",
         "--", sys.executable, "-m", "tests.test_inherited_lease", "single-cuda"],
        env=daemon_env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


def _run_single_cuda_worker():
    sys.modules["torch"] = types.SimpleNamespace(device=str)

    assert os.environ["RESERVED_DEVICE"] == "cuda:2"
    assert os.environ["RESERVED_DEVICES"] == "cuda:2"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    with sdk.adopt_inherited_lease() as lease:
        assert lease.devices == [sdk.GrantedDevice(device_name="cuda:2", cuda_ordinal=2)]
        assert lease.torch_devices() == ["cuda:0"]


def test_launcher_keeps_repeated_slots_in_physical_grant_order(daemon_env):
    result = subprocess.run(
        [sys.executable, "-m", "reserved.bin.reserve", "worker", "name=cuda:5",
         "name=cuda:2", "name=cuda:5", "--", sys.executable, "-m",
         "tests.test_inherited_lease", "repeated-cuda"],
        env=daemon_env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


def _run_repeated_cuda_worker():
    sys.modules["torch"] = types.SimpleNamespace(device=str)
    assert os.environ["RESERVED_DEVICE"] == "cuda:5"
    assert os.environ["RESERVED_DEVICES"] == "cuda:5,cuda:2,cuda:5"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5,2"
    with sdk.adopt_inherited_lease() as lease:
        assert lease.devices == [
            sdk.GrantedDevice("cuda:5", 5), sdk.GrantedDevice("cuda:2", 2),
            sdk.GrantedDevice("cuda:5", 5),
        ]
        assert lease.torch_devices() == ["cuda:0", "cuda:1", "cuda:0"]


def _run_launched_worker(mode):
    fd = int(os.environ["RESERVED_FD"])
    reservation_id = os.environ["RESERVED_ID"]
    socket_path = os.environ["RESERVED_SOCKET"]
    assert os.get_inheritable(fd)

    class WorkerError(Exception):
        pass

    reservation = sdk.adopt_inherited_lease()
    assert isinstance(reservation, sdk.Reservation)
    assert reservation.reservation_id == reservation_id
    assert reservation.devices == [sdk.GrantedDevice("cpu", -1)]
    assert "RESERVED_FD" not in os.environ
    assert not os.get_inheritable(fd)

    with tempfile.TemporaryFile() as unrelated:
        unrelated.write(b"still open")
        unrelated.seek(0)
        try:
            with reservation as entered:
                assert entered is reservation
                entered.check()
                with pytest.raises(sdk.InheritedLeaseAlreadyAdoptedError):
                    sdk.adopt_inherited_lease()
                with pytest.raises(sdk.ReservationError, match="cannot be acquired again"):
                    sdk.Request(entered).acquire()
                active = client._query_status(socket_path)["reservations"]
                assert [entry["id"] for entry in active] == [reservation_id]
                if mode == "exception":
                    raise WorkerError
        except WorkerError:
            assert mode == "exception"
        else:
            assert mode == "normal"

        # Install an unrelated file at the original fd after the context closes it.
        with pytest.raises(OSError):
            os.fstat(fd)
        os.dup2(unrelated.fileno(), fd, inheritable=True)
        try:
            os.environ["RESERVED_FD"] = str(fd)
            with pytest.raises(sdk.InheritedLeaseAlreadyAdoptedError):
                sdk.adopt_inherited_lease()
            reservation.release()
            assert os.get_inheritable(fd)
            assert os.read(fd, 10) == b"still open"
        finally:
            os.close(fd)

    with pytest.raises(sdk.ReservationExpiredError, match="released"):
        with reservation:
            pytest.fail("released lease was entered again")

    deadline = time.monotonic() + 5
    while client._query_status(socket_path)["reservations"]:
        assert time.monotonic() < deadline, "daemon did not reclaim the worker's lease"
        time.sleep(0.01)
    with sdk.Reservation("ordinary", sdk.ExplicitSlot("cpu", ram_mb=1024)) as ordinary:
        ordinary.check()


if __name__ == "__main__":
    if sys.argv[1] == "single-cuda":
        _run_single_cuda_worker()
    elif sys.argv[1] == "repeated-cuda":
        _run_repeated_cuda_worker()
    else:
        _run_launched_worker(sys.argv[1])
