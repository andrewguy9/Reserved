"""Daemon and clients must agree without a shell-specific configuration."""

import pytest

from reserved import paths
from reserved.daemon import _resolve_socket
from reserved.sdk.client import _get_socket_override


@pytest.mark.parametrize("resolve", [paths.resolve_socket_path, _resolve_socket])
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


@pytest.mark.parametrize("platform,default", [
    ("linux", "/var/run/reserved.sock"),
    ("darwin", "/tmp/reserved-501.sock"),
])
def test_client_socket_candidates_and_strict_overrides(monkeypatch, platform, default):
    monkeypatch.setattr(paths.sys, "platform", platform)
    monkeypatch.setattr(paths.os, "getuid", lambda: 501)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/501")
    monkeypatch.setenv("RESERVED_SOCKET", "/custom.sock")
    assert paths.client_socket_paths("/explicit.sock") == ("/explicit.sock",)
    assert paths.client_socket_paths() == ("/custom.sock",)
    assert _get_socket_override() == "/custom.sock"
    monkeypatch.setenv("RESERVED_SOCKET", "")
    assert _get_socket_override() is None
    assert paths.client_socket_paths() == ("/run/user/501/reserved.sock", default)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "")
    assert paths.client_socket_paths() == (default,)
