#!/usr/bin/env python3
"""
reserve-status — Print the current state of all devices and active reservations.

Usage:
    reserve-status [--socket PATH] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Ensure the package is importable when run from the repo root.
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

from reserved.sdk.client import _get_socket_path, _query_status
from reserved.sdk.errors import DaemonNotRunningError


def _format_human(status: dict) -> str:
    lines: list[str] = []
    lines.append("Devices:")
    for d in status.get("devices", []):
        model_str = f"  {d['model']}" if d.get("model") else ""
        ram_str = f"available: {d['available_ram_mb']:>6}/{d['ram_mb']} MB"
        cu = d.get("compute_units", 0)
        avail_cu = d.get("available_compute_units", 0)
        cu_str = f"  {avail_cu}/{cu} compute" if cu else ""
        lines.append(f"  {d['name']:<10}  {d['type']:<6}{model_str:<12}  {ram_str}{cu_str}")

    lines.append("")
    lines.append("Reservations:")
    reservations = status.get("reservations", [])
    if not reservations:
        lines.append("  (none)")
    else:
        now = time.time()
        for r in reservations:
            age = int(now - r.get("created_at", now))
            age_str = f"{age // 60}m{age % 60}s"
            lines.append(f"  {r['id']}  label={r['label']}  age={age_str}")
            for s in r.get("slots", []):
                cu_str = f"  {s['compute_units']} compute" if s.get("compute_units") else ""
                lines.append(f"    {s['device_name']}  {s['ram_mb']} MB{cu_str}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="reserve-status",
        description="Print current device and reservation status.",
    )
    parser.add_argument(
        "--socket",
        default=None,
        help="Daemon socket path [default: $RESERVED_SOCKET or /var/run/reserved.sock]",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output raw JSON",
    )
    args = parser.parse_args()

    socket_path = _get_socket_path(args.socket)

    try:
        status = _query_status(socket_path)
    except DaemonNotRunningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.as_json:
        print(json.dumps(status, indent=2))
    else:
        print(_format_human(status))


if __name__ == "__main__":
    main()
