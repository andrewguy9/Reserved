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
from .core import Ledger
from .paths import SOCKET_PATH_HELP, resolve_socket_path as _resolve_socket
from .daemon_state import (
    Accepted,
    CloseLease,
    CloseTransport,
    CreateLease,
    Effect,
    Event,
    LeaseClosed,
    SendBytes,
    SendLeaseFds,
    TransportClosed,
    TransportReceived,
    initial_state,
    step,
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
        self._state = initial_state(ledger)
        self._listen_sock = listen_sock
        self._http_port = http_port

        # Logical handle IDs live in daemon_state. Real sockets live here.
        self._transport_socks: dict[str, socket.socket] = {}
        self._lease_socks: dict[str, socket.socket] = {}
        self._pending_client_lease_socks: dict[str, socket.socket] = {}

        self._fd_to_transport: dict[int, str] = {}
        self._fd_to_lease: dict[int, str] = {}
        self._next_connection_id = 0

        self._running = False

    @property
    def ledger(self) -> Ledger:
        return self._state.ledger

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
                + list(self._fd_to_transport.keys())
                + list(self._fd_to_lease.keys())
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
                elif fd in self._fd_to_transport:
                    self._read_transport(fd)
                elif fd in self._fd_to_lease:
                    self._check_lease(fd)

    def _accept(self) -> None:
        try:
            conn, _ = self._listen_sock.accept()
        except OSError as exc:
            logger.warning("accept error: %s", exc)
            return
        conn.setblocking(False)
        connection_id = self._new_connection_id()
        self._transport_socks[connection_id] = conn
        self._fd_to_transport[conn.fileno()] = connection_id
        self._apply_event(Accepted(connection_id))

    def _read_transport(self, fd: int) -> None:
        connection_id = self._fd_to_transport[fd]
        conn = self._transport_socks[connection_id]
        try:
            chunk = conn.recv(4096)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return
            logger.warning("recv error on transport fd %d: %s", fd, exc)
            self._apply_event(TransportClosed(connection_id))
            return

        if not chunk:
            # EOF on transport — client closed without sending anything.
            self._apply_event(TransportClosed(connection_id))
            return

        self._apply_event(TransportReceived(connection_id, chunk))

    def _check_lease(self, fd: int) -> None:
        lease_id = self._fd_to_lease[fd]
        sock = self._lease_socks[lease_id]
        try:
            data = sock.recv(1)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return
            data = b""  # treat error as EOF
        if data == b"":
            # EOF — client closed its end; release the reservation.
            logger.info("releasing lease %s (lease fd closed)", lease_id)
            self._apply_event(LeaseClosed(lease_id))

    def _new_connection_id(self) -> str:
        self._next_connection_id += 1
        return f"conn-{self._next_connection_id}"

    def _apply_event(self, event: Event) -> None:
        self._state, effects = step(self._state, event)
        self._apply_effects(effects)

    def _apply_effects(self, effects: tuple[Effect, ...]) -> None:
        for effect in effects:
            if isinstance(effect, SendBytes):
                self._send_bytes(effect)
            elif isinstance(effect, CreateLease):
                self._create_lease(effect)
            elif isinstance(effect, SendLeaseFds):
                self._send_lease_fds(effect)
            elif isinstance(effect, CloseTransport):
                self._close_transport(effect.connection_id)
            elif isinstance(effect, CloseLease):
                self._close_lease(effect.lease_id)
            else:
                raise TypeError(f"unknown daemon effect: {effect!r}")

    def _send_bytes(self, effect: SendBytes) -> None:
        conn = self._transport_socks.get(effect.connection_id)
        if conn is None:
            return
        conn.setblocking(True)
        try:
            conn.sendall(effect.data)
        except OSError as exc:
            logger.warning("failed to send response: %s", exc)

    def _create_lease(self, effect: CreateLease) -> None:
        if effect.lease_id in self._lease_socks:
            return
        daemon_sock, client_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        daemon_sock.setblocking(False)
        self._lease_socks[effect.lease_id] = daemon_sock
        self._pending_client_lease_socks[effect.lease_id] = client_sock
        self._fd_to_lease[daemon_sock.fileno()] = effect.lease_id

    def _send_lease_fds(self, effect: SendLeaseFds) -> None:
        conn = self._transport_socks.get(effect.connection_id)
        if conn is None:
            return
        client_socks = [
            self._pending_client_lease_socks[lease_id]
            for lease_id in effect.lease_ids
            if lease_id in self._pending_client_lease_socks
        ]
        conn.setblocking(True)
        try:
            _send_fds(conn, effect.data, [s.fileno() for s in client_socks])
        except OSError as exc:
            logger.warning("failed to send grant response: %s", exc)
        finally:
            for lease_id in effect.lease_ids:
                sock = self._pending_client_lease_socks.pop(lease_id, None)
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

    def _close_transport(self, connection_id: str) -> None:
        sock = self._transport_socks.pop(connection_id, None)
        if sock is None:
            return
        self._fd_to_transport.pop(sock.fileno(), None)
        try:
            sock.close()
        except OSError:
            pass

    def _close_lease(self, lease_id: str) -> None:
        pending = self._pending_client_lease_socks.pop(lease_id, None)
        if pending is not None:
            try:
                pending.close()
            except OSError:
                pass

        sock = self._lease_socks.pop(lease_id, None)
        if sock is None:
            return
        self._fd_to_lease.pop(sock.fileno(), None)
        try:
            sock.close()
        except OSError:
            pass

    def _shutdown(self) -> None:
        logger.info("shutting down")
        for lease_id in list(self._pending_client_lease_socks):
            self._close_lease(lease_id)
        for lease_id in list(self._lease_socks):
            self._close_lease(lease_id)
        for connection_id in list(self._transport_socks):
            self._close_transport(connection_id)
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
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(1.0)
        probe.connect(socket_path)
        return "live"
    except FileNotFoundError:
        return "absent"
    except ConnectionRefusedError:
        return "stale"
    finally:
        probe.close()


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


def _acquire_socket_systemd() -> socket.socket | None:
    """Adopt one listening Unix stream socket addressed to this process."""
    listen_pid = os.environ.pop("LISTEN_PID", None)
    listen_fds = os.environ.pop("LISTEN_FDS", None)
    os.environ.pop("LISTEN_FDNAMES", None)
    if listen_pid != str(os.getpid()) or listen_fds is None:
        return None
    try:
        count = int(listen_fds)
    except ValueError:
        raise ValueError("LISTEN_FDS must be an integer") from None
    if count == 0:
        return None
    if count != 1:
        raise ValueError("reserved requires exactly one systemd listening socket")
    fd = SD_LISTEN_FDS_START
    sock = socket.socket(fileno=fd)
    try:
        if (
            sock.family != socket.AF_UNIX
            or sock.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM
        ):
            raise ValueError("systemd must pass a listening Unix stream socket")
        # SO_ACCEPTCONN is unavailable for macOS Unix sockets. Systemd runs on
        # Linux; keeping adoption portable also permits fd-passing smoke checks.
        if sys.platform == "linux" and not sock.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN):
            raise ValueError("systemd must pass a listening Unix stream socket")
        sock.set_inheritable(False)
    except BaseException:
        sock.close()
        raise
    return sock


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _resolve_config(flag: str | None) -> str:
    """Return the config file path using the documented fallback chain."""
    explicit = flag or os.environ.get("RESERVED_CONFIG")
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        print(f"error: config file not found: {explicit}", file=sys.stderr)
        sys.exit(1)
    candidates = [
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
    parser.add_argument("--socket", default=None, help=SOCKET_PATH_HELP)
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
    try:
        listen_sock = _acquire_socket_systemd()
        if listen_sock is None:
            socket_path = _resolve_socket(args.socket)
            listen_sock = _acquire_socket(socket_path, args.socket_group)
            logger.info("bound socket at %s", socket_path)
        else:
            logger.info("using systemd socket activation")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: could not acquire listening socket: {exc}\n")

    daemon = Daemon(ledger, listen_sock, http_port=args.http)
    daemon.run()


if __name__ == "__main__":
    main()
