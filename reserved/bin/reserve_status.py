#!/usr/bin/env python3
"""Print the current state of all devices and active reservations."""

from __future__ import annotations

import json
import os
import sys
import time

# Ensure the package is importable when run from the repo root.
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

from reserved.cli import parse_options
from reserved.paths import SOCKET_PATH_HELP
from reserved.sdk.client import _get_socket_override, _query_status
from reserved.sdk.errors import DaemonNotRunningError


_USAGE = f"""Print current device and reservation status.

Usage:
    reserve-status [--socket PATH] [--json]
    reserve-status (-h | --help)

Options:
    -h --help      Show this help.
    --socket PATH  {SOCKET_PATH_HELP}
    --json         Output raw JSON.
"""


def _format_human(status: dict) -> str:
    lines: list[str] = []
    lines.append("Devices:")
    for d in status.get("devices", []):
        ram_str = f"available: {d['available_ram_mb']:>6}/{d['ram_mb']} MB"
        cu = d.get("compute_units", 0)
        avail_cu = d.get("available_compute_units", 0)
        cu_str = f"  {avail_cu}/{cu} compute" if cu else ""
        lines.append(f"  {d['name']:<10}  {d['type']:<6}  {ram_str}{cu_str}")

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
            for s in r.get("allocations", []):
                cu_str = f"  {s['compute_units']} compute" if s.get("compute_units") else ""
                lines.append(f"    {s['name']}  {s['ram_mb']} MB{cu_str}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_options(_USAGE, argv)
    socket_path = _get_socket_override(args["--socket"])

    try:
        status = _query_status(socket_path)
    except DaemonNotRunningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    if args["--json"]:
        print(json.dumps(status, indent=2))
    else:
        print(_format_human(status))


if __name__ == "__main__":
    main()
