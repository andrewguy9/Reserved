#!/usr/bin/env python3
"""
reserve — Acquire a reservation and exec a command under it.

Usage:
    reserve LABEL [SLOT ...] [--solver consolidate|spread] [--socket PATH] -- CMD [ARGS...]

SLOT format: comma-separated key=value pairs.
    type=TYPE        Device type (cuda, metal, cpu)
    model=MODEL      Device model label
    ram=MB           RAM in MB (alias for ram_mb)
    ram_mb=MB        RAM in MB
    compute=N        Compute units (alias for compute_units)
    compute_units=N  Compute units
    name=NAME        Pin to a specific device (bypasses solver)

Examples:
    reserve train type=cuda,ram=8192 -- python train.py
    reserve infer name=cuda:2 -- python infer.py
    reserve distributed type=cuda type=cuda --solver spread -- python dist.py
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure the package is importable when run from the repo root.
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

from reserved.sdk.client import (
    DeviceConstraint,
    ExplicitSlot,
    Reservation,
    _get_socket_path,
    _query_status,
    _send_reserve,
)
from reserved.sdk.errors import DaemonNotRunningError, ReservationError
from reserved.sdk.solvers import ConsolidateSolver, SpreadSolver
from reserved.core import ReservationRequest, ReserveRequest, SlotRequest


def parse_slot(spec: str) -> DeviceConstraint | ExplicitSlot:
    """Parse a SLOT string like 'type=cuda,ram=8192' into a slot descriptor."""
    fields: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"error: invalid slot field {part!r} (expected key=value)")
        key, _, value = part.partition("=")
        fields[key.strip()] = value.strip()

    name = fields.get("name")
    if name is not None:
        # Explicit slot.
        ram_mb = int(fields.get("ram", fields.get("ram_mb", 0)))
        compute_units = int(fields.get("compute", fields.get("compute_units", 0)))
        return ExplicitSlot(name=name, ram_mb=ram_mb, compute_units=compute_units)
    else:
        # Constraint slot.
        return DeviceConstraint(
            type=fields.get("type"),
            model=fields.get("model"),
            ram_mb=int(fields.get("ram", fields.get("ram_mb", 0))),
            compute_units=int(fields.get("compute", fields.get("compute_units", 0))),
        )


def main() -> None:
    # Split argv at '--' to separate reserve args from CMD.
    argv = sys.argv[1:]
    try:
        sep = argv.index("--")
        reserve_args = argv[:sep]
        cmd_args = argv[sep + 1 :]
    except ValueError:
        reserve_args = argv
        cmd_args = []

    parser = argparse.ArgumentParser(
        prog="reserve",
        description="Acquire a reservation and exec a command under it.",
        add_help=True,
    )
    parser.add_argument("label", help="Reservation label")
    parser.add_argument(
        "slots",
        nargs="*",
        metavar="SLOT",
        help="Device constraints (type=TYPE,ram=MB,...)",
    )
    parser.add_argument(
        "--solver",
        choices=["consolidate", "spread"],
        default="consolidate",
        help="Constraint solver [default: consolidate]",
    )
    parser.add_argument(
        "--socket",
        default=None,
        help="Daemon socket path [default: $RESERVED_SOCKET or /var/run/reserved.sock]",
    )

    args = parser.parse_args(reserve_args)

    if not cmd_args:
        parser.error("no command specified after '--'")

    socket_path = _get_socket_path(args.socket)
    solver = SpreadSolver() if args.solver == "spread" else ConsolidateSolver()

    # Parse slot descriptors.
    slots: list[DeviceConstraint | ExplicitSlot] = []
    for spec in args.slots:
        slots.append(parse_slot(spec))

    label = args.label

    # Determine if all slots are explicit.
    all_explicit = all(isinstance(s, ExplicitSlot) for s in slots)

    if all_explicit or not slots:
        # Build explicit request directly.
        slot_reqs = [
            SlotRequest(name=s.name, ram_mb=s.ram_mb, compute_units=s.compute_units)
            for s in slots
            if isinstance(s, ExplicitSlot)
        ]
        req = ReserveRequest(reservations=(
            ReservationRequest(label=label, slots=tuple(slot_reqs)),
        ))
    else:
        # Run solver.
        try:
            status = _query_status(socket_path)
        except DaemonNotRunningError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)

        try:
            req = solver.solve([(label, slots)], status["devices"])
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)

    # Submit request to daemon.
    try:
        resp, fds = _send_reserve(socket_path, req, num_reservations=1)
    except DaemonNotRunningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    if not resp.get("ok"):
        print(f"error: reservation failed — {resp.get('error', 'unknown error')}", file=sys.stderr)
        # Close any stray fds.
        import os as _os
        for fd in fds:
            try:
                _os.close(fd)
            except OSError:
                pass
        sys.exit(1)

    granted = resp["reservations"][0]
    reservation_id = granted["reservation_id"]
    granted_slots = granted["slots"]

    # We have exactly one lease fd.
    lease_fd = fds[0] if fds else None

    # Build environment variables.
    # CUDA_VISIBLE_DEVICES takes physical ordinals (e.g. "2,3"), deduplicated.
    # Multiple slots on the same device are valid (shared capacity); the device
    # appears once in CUDA_VISIBLE_DEVICES. The child addresses it as cuda:0, etc.
    seen: set[str] = set()
    cuda_indices: list[str] = []
    for s in granted_slots:
        if s["cuda_ordinal"] >= 0:
            ordinal = str(s["cuda_ordinal"])
            if ordinal not in seen:
                seen.add(ordinal)
                cuda_indices.append(ordinal)
    all_device_names = [s["device_name"] for s in granted_slots]

    # Find first accelerator device (non-CPU).
    first_accel = next(
        (s["device_name"] for s in granted_slots if not s["device_name"].startswith("cpu")),
        all_device_names[0] if all_device_names else "",
    )

    env = os.environ.copy()
    if cuda_indices:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_indices)
    env["RESERVED_DEVICE"] = first_accel
    env["RESERVED_DEVICES"] = ",".join(all_device_names)
    env["RESERVED_ID"] = reservation_id
    if lease_fd is not None:
        env["RESERVED_FD"] = str(lease_fd)
        # Clear FD_CLOEXEC so CMD inherits the lease fd across exec.
        os.set_inheritable(lease_fd, True)

    # Replace this process with CMD.
    try:
        os.execvpe(cmd_args[0], cmd_args, env)
    except FileNotFoundError:
        print(f"error: command not found: {cmd_args[0]!r}", file=sys.stderr)
        if lease_fd is not None:
            os.close(lease_fd)
        sys.exit(127)


if __name__ == "__main__":
    main()
