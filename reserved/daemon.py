"""
daemon.py — Imperative shell for the reserved daemon.

Owns the single mutable reference to the current Ledger and drives the select
loop. Handles:
  - Listening socket (accept new connections)
  - Short-lived transport connections (reserve and status requests)
  - Long-lived lease fds (socketpair ends — watched for EOF = release signal)
  - Optional HTTP server for monitoring
"""

from __future__ import annotations

import array
import errno
import grp
import json
import logging
import os
import select
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from . import core, protocol
from .core import (
    Ledger,
    ReserveRequest,
)
from .protocol import (
    grant_failure_to_wire,
    grant_success_to_wire,
    parse_request,
    status_result_to_wire,
    wire_to_reserve_request,
)

logger = logging.getLogger("reserved")

# Systemd socket activation constant.
SD_LISTEN_FDS_START = 3


# ---------------------------------------------------------------------------
# SCM_RIGHTS helpers
# ---------------------------------------------------------------------------

def _send_fds(sock: socket.socket, data: bytes, fds: list[int]) -> None:
    """Send *data* together with *fds* via SCM_RIGHTS ancillary data."""
    if not fds:
        sock.sendall(data)
        return
    cmsg_level = socket.SOL_SOCKET
    cmsg_type = socket.SCM_RIGHTS
    fd_array = array.array("i", fds)
    sock.sendmsg([data], [(cmsg_level, cmsg_type, fd_array)])


def _recv_fds(sock: socket.socket, bufsize: int, maxfds: int) -> tuple[bytes, list[int]]:
    """Receive data plus up to *maxfds* file descriptors via SCM_RIGHTS."""
    fd_array = array.array("i")
    cmsg_space = socket.CMSG_SPACE(maxfds * fd_array.itemsize)
    msg, ancdata, _flags, _addr = sock.recvmsg(bufsize, cmsg_space)
    fds: list[int] = []
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fd_array.frombytes(cmsg_data[:len(cmsg_data) - (len(cmsg_data) % fd_array.itemsize)])
            fds.extend(fd_array)
    return msg, fds


# ---------------------------------------------------------------------------
# HTTP server for optional monitoring
# ---------------------------------------------------------------------------

class _StatusHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler serving /status (JSON) and / (HTML dashboard)."""

    daemon_ref: "Daemon"  # set on the class before use

    def log_message(self, fmt: str, *args: Any) -> None:  # silence access logs
        pass

    def do_GET(self) -> None:
        if self.path == "/status":
            self._serve_status_json()
        elif self.path == "/":
            self._serve_html()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def _serve_status_json(self) -> None:
        result = core.status(self.daemon_ref.ledger)
        body = protocol.status_result_to_wire(result)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self) -> None:
        result = core.status(self.daemon_ref.ledger)
        rows_devices = ""
        for d in result.devices:
            rows_devices += (
                f"<tr>"
                f"<td>{d.name}</td>"
                f"<td>{d.type}</td>"
                f"<td>{d.model or ''}</td>"
                f"<td>{d.available_ram_mb}/{d.ram_mb} MB</td>"
                f"<td>{d.available_compute_units}/{d.compute_units}</td>"
                f"</tr>\n"
            )
        rows_reservations = ""
        now = time.time()
        for r in result.reservations:
            age = int(now - r.created_at)
            age_str = f"{age // 60}m{age % 60}s"
            slot_rows = "".join(
                f"<tr><td></td><td>{s.device_name}</td>"
                f"<td>{s.ram_mb} MB</td><td>{s.compute_units}</td></tr>\n"
                for s in r.slots
            )
            rows_reservations += (
                f"<tr>"
                f"<td>{r.id}</td>"
                f"<td>{r.label}</td>"
                f"<td>{age_str}</td>"
                f"<td></td><td></td>"
                f"</tr>\n"
                + slot_rows
            )
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>reserved status</title>
<style>
body{{font-family:monospace;padding:1em}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:4px 8px;text-align:left}}
th{{background:#eee}}
h2{{margin-top:1.5em}}
</style>
</head><body>
<h1>reserved status</h1>
<h2>Devices</h2>
<table>
<tr><th>Name</th><th>Type</th><th>Model</th><th>RAM</th><th>Compute</th></tr>
{rows_devices}
</table>
<h2>Reservations</h2>
<table>
<tr><th>ID</th><th>Label</th><th>Age</th><th>Device</th><th>RAM</th></tr>
{rows_reservations if rows_reservations else '<tr><td colspan="5">(none)</td></tr>'}
</table>
</body></html>
"""
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

class Daemon:
    """
    The reserved daemon.

    Owns the single mutable ledger reference and the select loop.
    """

    def __init__(
        self,
        ledger: Ledger,
        listen_sock: socket.socket,
        http_port: int | None = None,
    ) -> None:
        self.ledger = ledger
        self._listen_sock = listen_sock
        self._http_port = http_port

        # Maps fd → (socket, reservation_id) for long-lived lease fds.
        self._lease_fds: dict[int, tuple[socket.socket, str]] = {}

        # Maps fd → socket for short-lived transport connections (reading request).
        self._transport_fds: dict[int, socket.socket] = {}

        # Buffer for partial reads on transport connections.
        self._transport_bufs: dict[int, bytes] = {}

        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Enter the event loop. Blocks until a signal or fatal error."""
        self._running = True
        self._setup_signals()

        if self._http_port is not None:
            self._start_http_server()

        logger.info("reserved daemon started, listening on %s", self._listen_sock.getsockname())

        try:
            self._loop()
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _setup_signals(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigterm)

    def _handle_sigterm(self, signum: int, frame: Any) -> None:
        logger.info("received signal %d, shutting down", signum)
        # Log active reservations so operators know what will be disrupted.
        result = core.status(self.ledger)
        if result.reservations:
            logger.info("active reservations at shutdown:")
            for r in result.reservations:
                logger.info("  %s  label=%s", r.id, r.label)
        self._running = False

    def _start_http_server(self) -> None:
        # Bind _StatusHandler.daemon_ref to self via a subclass so the handler
        # can read self.ledger without a global.
        handler_cls = type(
            "_BoundStatusHandler",
            (_StatusHandler,),
            {"daemon_ref": self},
        )
        server = HTTPServer(("0.0.0.0", self._http_port), handler_cls)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        logger.info("HTTP server started on port %d", self._http_port)

    def _loop(self) -> None:
        listen_fd = self._listen_sock.fileno()

        while self._running:
            read_fds = (
                [listen_fd]
                + list(self._transport_fds.keys())
                + list(self._lease_fds.keys())
            )
            try:
                readable, _, _ = select.select(read_fds, [], [], 1.0)
            except (ValueError, OSError) as exc:
                # Can happen if a bad fd sneaks in; log and continue.
                logger.warning("select error: %s", exc)
                continue

            for fd in readable:
                if fd == listen_fd:
                    self._accept()
                elif fd in self._transport_fds:
                    self._read_transport(fd)
                elif fd in self._lease_fds:
                    self._check_lease(fd)

    def _accept(self) -> None:
        try:
            conn, _ = self._listen_sock.accept()
        except OSError as exc:
            logger.warning("accept error: %s", exc)
            return
        conn.setblocking(False)
        fd = conn.fileno()
        self._transport_fds[fd] = conn
        self._transport_bufs[fd] = b""

    def _read_transport(self, fd: int) -> None:
        conn = self._transport_fds[fd]
        try:
            chunk = conn.recv(4096)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return
            logger.warning("recv error on transport fd %d: %s", fd, exc)
            self._close_transport(fd)
            return

        if not chunk:
            # EOF on transport — client closed without sending anything.
            self._close_transport(fd)
            return

        self._transport_bufs[fd] += chunk
        if b"\n" in self._transport_bufs[fd]:
            line, _, rest = self._transport_bufs[fd].partition(b"\n")
            self._transport_bufs[fd] = rest
            self._handle_request(fd, conn, line)

    def _handle_request(self, fd: int, conn: socket.socket, line: bytes) -> None:
        try:
            obj = parse_request(line)
        except ValueError as exc:
            logger.warning("bad request from fd %d: %s", fd, exc)
            self._close_transport(fd)
            return

        op = obj.get("op")
        if op == "reserve":
            self._handle_reserve(fd, conn, obj)
        elif op == "status":
            self._handle_status(fd, conn)
        else:
            logger.warning("unknown op %r from fd %d", op, fd)
            self._close_transport(fd)

    def _handle_reserve(self, fd: int, conn: socket.socket, obj: dict) -> None:
        try:
            request = wire_to_reserve_request(obj)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("malformed reserve request: %s", exc)
            try:
                conn.sendall(grant_failure_to_wire(
                    protocol.GrantFailure(error=f"malformed request: {exc}")
                ))
            except OSError:
                pass
            self._close_transport(fd)
            return

        new_ledger, result = core.grant(self.ledger, request)

        if not result.ok:
            logger.info("reserve rejected: %s", result.error)
            try:
                conn.sendall(grant_failure_to_wire(result))
            except OSError:
                pass
            self._close_transport(fd)
            return

        # Grant succeeded — create lease socketpairs.
        lease_socks_daemon: list[socket.socket] = []   # daemon keeps these
        lease_socks_client: list[socket.socket] = []   # sent to client
        lease_fds_client: list[int] = []

        for _ in result.reservations:
            a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            lease_socks_daemon.append(a)
            lease_socks_client.append(b)
            lease_fds_client.append(b.fileno())

        # Send JSON response + lease fds via SCM_RIGHTS.
        wire = grant_success_to_wire(result)
        conn.setblocking(True)
        try:
            _send_fds(conn, wire, lease_fds_client)
        except OSError as exc:
            logger.warning("failed to send grant response: %s", exc)
            for s in lease_socks_daemon + lease_socks_client:
                s.close()
            self._close_transport(fd)
            return

        # Close client ends now that they've been sent.
        for s in lease_socks_client:
            s.close()

        # Register lease fds in the daemon's watch table.
        self.ledger = new_ledger
        for daemon_sock, granted_res in zip(lease_socks_daemon, result.reservations):
            daemon_sock.setblocking(False)
            dfd = daemon_sock.fileno()
            self._lease_fds[dfd] = (daemon_sock, granted_res.reservation_id)
            logger.info(
                "granted reservation %s label=%s",
                granted_res.reservation_id,
                granted_res.label,
            )

        self._close_transport(fd)

    def _handle_status(self, fd: int, conn: socket.socket) -> None:
        result = core.status(self.ledger)
        wire = status_result_to_wire(result)
        conn.setblocking(True)
        try:
            conn.sendall(wire)
        except OSError as exc:
            logger.warning("failed to send status response: %s", exc)
        self._close_transport(fd)

    def _check_lease(self, fd: int) -> None:
        sock, reservation_id = self._lease_fds[fd]
        try:
            data = sock.recv(1)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return
            data = b""  # treat error as EOF
        if data == b"":
            # EOF — client closed its end; release the reservation.
            logger.info("releasing reservation %s (lease fd closed)", reservation_id)
            self.ledger = core.release(self.ledger, reservation_id)
            sock.close()
            del self._lease_fds[fd]

    def _close_transport(self, fd: int) -> None:
        sock = self._transport_fds.pop(fd, None)
        self._transport_bufs.pop(fd, None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _shutdown(self) -> None:
        logger.info("shutting down")
        for sock, _ in self._lease_fds.values():
            sock.close()
        self._lease_fds.clear()
        for sock in self._transport_fds.values():
            sock.close()
        self._transport_fds.clear()
        try:
            self._listen_sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Socket acquisition
# ---------------------------------------------------------------------------

def _probe_socket(socket_path: str) -> str:
    """
    Check whether an existing socket file is live or stale.
    Returns 'live', 'stale', or 'absent'.
    """
    try:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(1.0)
        probe.connect(socket_path)
        probe.close()
        return "live"
    except FileNotFoundError:
        return "absent"
    except (ConnectionRefusedError, OSError):
        return "stale"


def _acquire_socket(socket_path: str, socket_group: str | None) -> socket.socket:
    """Create and bind the Unix domain listening socket."""
    state = _probe_socket(socket_path)
    if state == "live":
        print(
            f"error: another instance of reserved is already running at {socket_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    if state == "stale":
        logger.warning("removing stale socket at %s", socket_path)
        os.unlink(socket_path)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(socket_path)
    os.chmod(socket_path, 0o660)
    if socket_group is not None:
        gid = grp.getgrnam(socket_group).gr_gid
        os.chown(socket_path, -1, gid)
    sock.listen(64)
    return sock


def _acquire_socket_systemd() -> socket.socket:
    """Adopt a socket passed by systemd socket activation."""
    fd = SD_LISTEN_FDS_START
    sock = socket.fromfd(fd, socket.AF_UNIX, socket.SOCK_STREAM)
    # fromfd duplicates; close the original.
    os.close(fd)
    return sock


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _resolve_config(flag: str | None) -> str:
    """Return the config file path using the documented fallback chain."""
    candidates = [
        flag,
        os.environ.get("RESERVED_CONFIG"),
        os.path.expanduser("~/.config/reserved/reserved.json"),
        "/etc/reserved/reserved.json",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    searched = [c for c in candidates if c]
    print(
        f"error: no config file found. Searched: {', '.join(searched)}\n"
        "Pass --config PATH or set RESERVED_CONFIG.",
        file=sys.stderr,
    )
    sys.exit(1)


def _resolve_socket(flag: str | None) -> str:
    """Return the socket path using the documented fallback chain."""
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    candidates = [
        flag,
        os.environ.get("RESERVED_SOCKET"),
        os.path.join(xdg_runtime, "reserved.sock") if xdg_runtime else None,
        "/var/run/reserved.sock",
    ]
    for path in candidates:
        if path:
            return path
    return "/var/run/reserved.sock"


def main() -> None:
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(
        prog="reserved",
        description="Resource reservation daemon",
    )
    parser.add_argument("--config", default=None, help=(
        "Path to reserved.json config file. "
        "Falls back to RESERVED_CONFIG env var, "
        "~/.config/reserved/reserved.json, /etc/reserved/reserved.json."
    ))
    parser.add_argument("--socket", default=None, help=(
        "Unix domain socket path. "
        "Falls back to RESERVED_SOCKET env var, "
        "$XDG_RUNTIME_DIR/reserved.sock, /var/run/reserved.sock."
    ))
    parser.add_argument("--socket-group", default=None, help="Set socket file group")
    parser.add_argument("--http", type=int, default=None, metavar="PORT",
                        help="Start HTTP monitoring server on PORT")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config_path = _resolve_config(args.config)
    with open(config_path) as f:
        config = _json.load(f)

    ledger = core.load_ledger(config)
    logger.info("loaded %d device(s) from %s", len(ledger.availability), config_path)

    # Socket acquisition.
    listen_fd_env = os.environ.get("LISTEN_FDS")
    if listen_fd_env is not None:
        logger.info("using systemd socket activation (LISTEN_FDS=%s)", listen_fd_env)
        listen_sock = _acquire_socket_systemd()
    else:
        socket_path = _resolve_socket(args.socket)
        listen_sock = _acquire_socket(socket_path, args.socket_group)
        logger.info("bound socket at %s", socket_path)

    daemon = Daemon(ledger, listen_sock, http_port=args.http)
    daemon.run()


if __name__ == "__main__":
    main()
