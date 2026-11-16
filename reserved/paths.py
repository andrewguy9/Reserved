"""Socket discovery shared by the daemon, SDK, and command line tools."""

import os
import sys


SOCKET_PATH_HELP = (
    "Unix domain socket path. Defaults to RESERVED_SOCKET, then "
    "$XDG_RUNTIME_DIR/reserved.sock, then /tmp/reserved-<uid>.sock on macOS "
    "or /var/run/reserved.sock on Linux."
)


def resolve_socket_path(override: str | None = None) -> str:
    """Resolve a socket without probing it or changing the filesystem."""
    if override:
        return override
    if path := os.environ.get("RESERVED_SOCKET"):
        return path
    if runtime_dir := os.environ.get("XDG_RUNTIME_DIR"):
        return os.path.join(runtime_dir, "reserved.sock")
    if sys.platform == "darwin":
        return f"/tmp/reserved-{os.getuid()}.sock"
    return "/var/run/reserved.sock"
