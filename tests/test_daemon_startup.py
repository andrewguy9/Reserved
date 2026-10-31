"""Configuration and socket-activation startup contracts."""

import errno
import os
from pathlib import Path
import socket
import tempfile

import pytest

from reserved import daemon


@pytest.mark.parametrize("source", ["flag", "environment"])
def test_explicit_missing_config_does_not_fall_back(monkeypatch, tmp_path, source):
    monkeypatch.setenv("HOME", str(tmp_path))
    fallback = tmp_path / ".config" / "reserved" / "reserved.json"
    fallback.parent.mkdir(parents=True)
    fallback.write_text('{"devices": []}')
    missing = str(tmp_path / "missing.json")
    monkeypatch.delenv("RESERVED_CONFIG", raising=False)
    if source == "environment":
        monkeypatch.setenv("RESERVED_CONFIG", missing)
    with pytest.raises(SystemExit) as exc:
        daemon._resolve_config(missing if source == "flag" else None)
    assert exc.value.code == 1


def test_config_flag_overrides_environment(monkeypatch, tmp_path):
    config = tmp_path / "reserved.json"
    config.write_text('{"devices": []}')
    monkeypatch.setenv("RESERVED_CONFIG", "/missing.json")
    assert daemon._resolve_config(str(config)) == str(config)


@pytest.mark.parametrize("pid,count", [(None, None), ("other", "1"), (str(os.getpid()), "0")])
def test_activation_ignores_absent_or_unaddressed_descriptors(monkeypatch, pid, count):
    for name, value in [("LISTEN_PID", pid), ("LISTEN_FDS", count)]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert daemon._acquire_socket_systemd() is None


@pytest.mark.parametrize("count", ["bad", "-1", "2"])
def test_activation_rejects_invalid_descriptor_count(monkeypatch, count):
    monkeypatch.setenv("LISTEN_PID", str(os.getpid()))
    monkeypatch.setenv("LISTEN_FDS", count)
    with pytest.raises(ValueError):
        daemon._acquire_socket_systemd()


def test_activation_adopts_listener_and_preserves_it_for_manager(monkeypatch):
    # Use the actual inherited-socket logic without replacing pytest's fd 3.
    # macOS's Unix socket limit is shorter than pytest's default temp paths.
    with tempfile.TemporaryDirectory(prefix="rd-", dir="/tmp") as temp:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(Path(temp) / "activation.sock"))
            listener.listen(1)
            inherited = os.dup(listener.fileno())
            monkeypatch.setattr(daemon, "SD_LISTEN_FDS_START", inherited)
            monkeypatch.setenv("LISTEN_PID", str(os.getpid()))
            monkeypatch.setenv("LISTEN_FDS", "1")
            monkeypatch.setenv("LISTEN_FDNAMES", "reserved")
            adopted = daemon._acquire_socket_systemd()
            assert adopted.getsockname() == listener.getsockname()
            assert not adopted.get_inheritable()
            assert "LISTEN_FDS" not in os.environ
            assert "LISTEN_PID" not in os.environ
            assert "LISTEN_FDNAMES" not in os.environ
            adopted.close()
            assert daemon._probe_socket(listener.getsockname()) == "live"


def test_activation_rejects_datagram_socket(monkeypatch):
    first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    inherited = os.dup(first.fileno())
    monkeypatch.setattr(daemon, "SD_LISTEN_FDS_START", inherited)
    monkeypatch.setenv("LISTEN_PID", str(os.getpid()))
    monkeypatch.setenv("LISTEN_FDS", "1")
    try:
        with pytest.raises(ValueError, match="listening Unix stream"):
            daemon._acquire_socket_systemd()
    finally:
        first.close()
        second.close()


def test_socket_probe_does_not_classify_permission_denied_as_stale(monkeypatch):
    class InaccessibleSocket:
        def settimeout(self, timeout):
            pass

        def connect(self, path):
            raise PermissionError(errno.EACCES, "Permission denied")

        def close(self):
            pass

    monkeypatch.setattr(daemon.socket, "socket", lambda *args: InaccessibleSocket())
    with pytest.raises(PermissionError):
        daemon._probe_socket("/private/reserved.sock")
