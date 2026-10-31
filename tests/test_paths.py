"""Daemon and clients must agree without a shell-specific configuration."""

import pytest

from reserved import paths
from reserved.daemon import _resolve_socket
from reserved.sdk.client import _get_socket_path


@pytest.mark.parametrize("resolve", [paths.resolve_socket_path, _resolve_socket, _get_socket_path])
@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_socket_discovery_precedence(monkeypatch, resolve, platform):
    monkeypatch.setattr(paths.sys, "platform", platform)
    monkeypatch.setattr(paths.os, "getuid", lambda: 501)
    monkeypatch.setenv("RESERVED_SOCKET", "/custom/reserved.sock")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/501")

    assert resolve("/explicit.sock") == "/explicit.sock"
    assert resolve() == "/custom/reserved.sock"
    monkeypatch.delenv("RESERVED_SOCKET")
    assert resolve() == "/run/user/501/reserved.sock"
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    expected = "/tmp/reserved-501.sock" if platform == "darwin" else "/var/run/reserved.sock"
    assert resolve() == expected


def test_empty_socket_environment_uses_default(monkeypatch):
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setenv("RESERVED_SOCKET", "")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "")
    assert paths.resolve_socket_path("") == "/var/run/reserved.sock"
