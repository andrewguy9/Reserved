"""Socket discovery shared by the daemon, SDK, and command line tools."""

import os
import sys


SOCKET_PATH_HELP = (
    "Unix domain socket path. Defaults to RESERVED_SOCKET, then "
    "$XDG_RUNTIME_DIR/reserved.sock, then /tmp/reserved-<uid>.sock on macOS "
    "or /var/run/reserved.sock on Linux. Clients try the next default only "
    "if the previous socket is missing or refuses connections; explicit paths "
    "and RESERVED_SOCKET never fall back."
)


def _platform_socket_path() -> str:
    if sys.platform == "darwin":
        return f"/tmp/reserved-{os.getuid()}.sock"
    return "/var/run/reserved.sock"


def client_socket_override(override: str | None = None) -> str | None:
    """Keep explicit targets strict; defer automatic discovery to connect time."""
    return override or os.environ.get("RESERVED_SOCKET") or None


def client_socket_paths(override: str | None = None) -> tuple[str, ...]:
    """Ordered connection candidates, without probing or changing the filesystem."""
    if path := client_socket_override(override):
        return (path,)
    default = _platform_socket_path()
    if runtime_dir := os.environ.get("XDG_RUNTIME_DIR"):
        runtime = os.path.join(runtime_dir, "reserved.sock")
        if runtime != default:
            return (runtime, default)
    return (default,)


def resolve_socket_path(override: str | None = None) -> str:
    """Select the daemon's bind path without probing or changing the filesystem."""
    if path := client_socket_override(override):
        return path
    if runtime_dir := os.environ.get("XDG_RUNTIME_DIR"):
        return os.path.join(runtime_dir, "reserved.sock")
    return _platform_socket_path()
