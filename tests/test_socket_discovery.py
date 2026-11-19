"""Client fallback uses real Unix sockets without touching installed daemons."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import pytest

from reserved import paths, sdk
from reserved.bin import reserve_status
from reserved.sdk import client


@pytest.fixture
def discovery(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="rd-discovery-", dir="/tmp") as directory:
        root = Path(directory)
        runtime = root / "runtime"
        runtime.mkdir()
        system = str(root / "system.sock")
        monkeypatch.delenv("RESERVED_SOCKET", raising=False)
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
        monkeypatch.setattr(paths, "_platform_socket_path", lambda: system)
        config = root / "devices.json"
        config.write_text(json.dumps({"devices": [
            {"name": "cpu", "type": "cpu", "ram_mb": 1024},
        ]}))
        daemon_env = dict(os.environ)
        for key in ("LISTEN_PID", "LISTEN_FDS", "LISTEN_FDNAMES"):
            daemon_env.pop(key, None)
        with subprocess.Popen(
            [sys.executable, "-m", "reserved.daemon", "--socket", system,
             "--config", str(config)],
            env=daemon_env, stderr=subprocess.PIPE, text=True,
        ) as daemon:
            try:
                deadline = time.monotonic() + 5
                while True:
                    try:
                        client._query_status(system)
                        break
                    except sdk.DaemonNotRunningError:
                        if daemon.poll() is not None:
                            pytest.fail(daemon.communicate()[1])
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.01)
                yield str(runtime / "reserved.sock"), system
            finally:
                daemon.terminate()
                daemon.communicate(timeout=5)


@pytest.mark.parametrize("stale", [False, True], ids=["missing", "refused"])
def test_status_falls_back_to_system_socket(discovery, capsys, stale):
    runtime, _ = discovery
    if stale:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.bind(runtime)  # Leave a socket file with no listener.

    reserve_status.main(["--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["devices"][0]["name"] == "cpu"


def test_live_runtime_socket_takes_precedence(discovery):
    runtime, _ = discovery
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(runtime)
        listener.listen(1)
        listener.settimeout(1)
        with client._connect(None) as connected:
            assert connected.getpeername() == runtime
            accepted, _ = listener.accept()
            accepted.close()


@pytest.mark.parametrize("source", ["flag", "environment"])
def test_explicit_missing_socket_never_falls_back(discovery, monkeypatch, capsys, source):
    runtime, _ = discovery
    if source == "environment":
        monkeypatch.setenv("RESERVED_SOCKET", runtime)
    args = ["--socket", runtime] if source == "flag" else []
    with pytest.raises(SystemExit) as error:
        reserve_status.main(args)
    assert error.value.code == 2
    assert runtime in capsys.readouterr().err


@pytest.mark.parametrize("atomic", [False, True])
def test_sdk_acquires_and_releases_through_default_fallback(discovery, atomic):
    _, system = discovery
    reservation = sdk.Reservation("fallback", sdk.DeviceConstraint("cpu", ram_mb=256))
    try:
        if atomic:
            assert sdk.Request(reservation).acquire() == (reservation,)
        else:
            reservation.__enter__()
        reservation.check()
        assert reservation.devices[0].device_name == "cpu"
        status = client._query_status(system)
        assert status["devices"][0]["available_ram_mb"] == 768
    finally:
        reservation.release()
    deadline = time.monotonic() + 5
    while client._query_status(system)["reservations"]:
        assert time.monotonic() < deadline, "reservation was not reclaimed"
        time.sleep(0.01)
    assert client._query_status(system)["devices"][0]["available_ram_mb"] == 1024


def test_acquire_reservations_labels_and_releases_independently(discovery):
    _, system = discovery
    slot_groups = sdk.ConsolidateSolver().solve(client._query_status(system)["devices"], [
        [sdk.DeviceConstraint("cpu", ram_mb=768)],
        [sdk.DeviceConstraint("cpu", ram_mb=256)],
    ])
    first, second = sdk.acquire_reservations(slot_groups, labels=["first", "second"])
    try:
        status = client._query_status(system)
        assert [r["label"] for r in status["reservations"]] == ["first", "second"]
        assert status["devices"][0]["available_ram_mb"] == 0
        first.release()
        deadline = time.monotonic() + 5
        while len(client._query_status(system)["reservations"]) != 1:
            assert time.monotonic() < deadline, "first lease was not reclaimed"
            time.sleep(0.01)
        second.check()
        assert client._query_status(system)["devices"][0]["available_ram_mb"] == 768
        with pytest.raises(sdk.ReservationError):
            sdk.acquire_reservations(slot_groups, labels=["first", "second"], socket_path=system)
        assert len(client._query_status(system)["reservations"]) == 1
        assert client._query_status(system)["devices"][0]["available_ram_mb"] == 768
    finally:
        first.release()
        second.release()


def test_reserve_cli_uses_default_fallback(discovery):
    _, system = discovery
    # Redirect only the platform default in the child, keeping real transport,
    # descriptor passing, and CLI exec behavior.
    launcher = (
        "import sys; from reserved import paths; "
        "system = sys.argv.pop(1); paths._platform_socket_path = lambda: system; "
        "from reserved.bin.reserve import main; main()"
    )
    result = subprocess.run(
        [sys.executable, "-c", launcher, system, "cli-fallback", "type=cpu,ram=256",
         "--", sys.executable, "-c", "print('worker ran')"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "worker ran"


def test_all_missing_defaults_report_every_attempt(discovery, monkeypatch):
    runtime, system = discovery
    missing = system + ".missing"
    monkeypatch.setattr(paths, "_platform_socket_path", lambda: missing)
    with pytest.raises(sdk.DaemonNotRunningError) as error:
        client._query_status(None)
    assert runtime in str(error.value)
    assert missing in str(error.value)


def test_permission_error_closes_socket_without_fallback(monkeypatch):
    created = []

    class DeniedSocket:
        closed = False

        def connect(self, path):
            raise PermissionError("denied")

        def close(self):
            self.closed = True

    def create(*args):
        sock = DeniedSocket()
        created.append(sock)
        return sock

    monkeypatch.setattr(client, "client_socket_paths", lambda _: ("/first", "/second"))
    monkeypatch.setattr(client.socket, "socket", create)
    with pytest.raises(PermissionError, match="denied"):
        client._connect(None)
    assert len(created) == 1
    assert created[0].closed


@pytest.mark.parametrize("failure", [FileNotFoundError, ConnectionRefusedError])
def test_failed_default_is_closed_before_trying_system_socket(monkeypatch, failure):
    created = []
    attempted = []

    class TransportSocket:
        closed = False

        def connect(self, path):
            attempted.append(path)
            if path == "/runtime/reserved.sock":
                raise failure(path)
            assert created[0].closed

        def close(self):
            self.closed = True

    def create(*args):
        sock = TransportSocket()
        created.append(sock)
        return sock

    monkeypatch.delenv("RESERVED_SOCKET", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/runtime")
    monkeypatch.setattr(paths, "_platform_socket_path", lambda: "/system.sock")
    monkeypatch.setattr(client.socket, "socket", create)
    connected = client._connect(None)
    try:
        assert attempted == ["/runtime/reserved.sock", "/system.sock"]
        assert connected is created[1]
        assert not connected.closed
    finally:
        connected.close()


@pytest.mark.parametrize("source", ["argument", "environment"])
@pytest.mark.parametrize("failure", [FileNotFoundError, ConnectionRefusedError])
def test_failed_explicit_target_is_closed_and_never_falls_back(monkeypatch, source, failure):
    attempted = []

    class FailedSocket:
        closed = False

        def connect(self, path):
            attempted.append(path)
            raise failure(path)

        def close(self):
            self.closed = True

    sock = FailedSocket()
    monkeypatch.delenv("RESERVED_SOCKET", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/runtime")
    if source == "environment":
        monkeypatch.setenv("RESERVED_SOCKET", "/explicit.sock")
    monkeypatch.setattr(client.socket, "socket", lambda *args: sock)
    with pytest.raises(sdk.DaemonNotRunningError, match="/explicit.sock") as error:
        client._connect("/explicit.sock" if source == "argument" else None)
    assert isinstance(error.value.__cause__, failure)
    assert attempted == ["/explicit.sock"]
    assert sock.closed
