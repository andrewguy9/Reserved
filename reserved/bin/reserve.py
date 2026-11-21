#!/usr/bin/env python3
"""Acquire a reservation and exec a command under it."""

from __future__ import annotations

import json
import os
import sys

# Ensure the package is importable when run from the repo root.
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

from reserved.cli import parse_options, usage_error
from reserved.core import DeviceAllocation, ReservationRequest, ReserveRequest, cuda_uuid
from reserved.paths import SOCKET_PATH_HELP
from reserved.sdk.client import (
    DeviceConstraint,
    _get_socket_override,
    _query_status,
    _send_reserve,
)
from reserved.sdk.errors import DaemonNotRunningError
from reserved.sdk.solvers import ConsolidateSolver, SpreadSolver


_USAGE = f"""Acquire a reservation and exec a command under it.

Usage:
    reserve [options] LABEL [DEVICE...]
    reserve (-h | --help)

Append -- CMD [ARGS...] after the options, label, and devices.

Options:
    -h --help        Show this help.
    --solver SOLVER  Select consolidate or spread [default: consolidate].
    --socket PATH    {SOCKET_PATH_HELP}

DEVICE format: comma-separated key=value pairs. Each DEVICE is one
allocation in the reservation: a constraint the solver places, or an explicit
device when name= is given.
    type=TYPE        Device type (cuda, metal, cpu)
    model=MODEL      Device model label
    ram=MB           RAM in MB (alias for ram_mb)
    compute=N        Compute units (alias for compute_units)
    name=NAME        Pin to a physical device (bypasses solver).
                     CUDA names are cuda:GPU-<uuid> as printed by nvidia-smi -L.

Examples:
    reserve train type=cuda,ram=8192 -- python train.py
    reserve infer name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456782 -- python infer.py
    reserve dist name=cuda:GPU-...80 name=cuda:GPU-...81 -- python dist.py
"""


def parse_device_spec(spec: str) -> DeviceConstraint | DeviceAllocation:
    """Parse a DEVICE string like 'type=cuda,ram=8192'.

    Returns a DeviceAllocation when the spec names a device, otherwise a
    DeviceConstraint for the solver.
    """
    fields: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"error: invalid device field {part!r} (expected key=value)")
        key, _, value = part.partition("=")
        fields[key.strip()] = value.strip()

    physical_name = fields.get("name")
    if physical_name is not None:
        # Explicit allocation.
        ram_mb = int(fields.get("ram", fields.get("ram_mb", 0)))
        compute_units = int(fields.get("compute", fields.get("compute_units", 0)))
        return DeviceAllocation(name=physical_name, ram_mb=ram_mb, compute_units=compute_units)
    else:
        # Constraint for the solver.
        return DeviceConstraint(
            type=fields.get("type"),
            model=fields.get("model"),
            ram_mb=int(fields.get("ram", fields.get("ram_mb", 0))),
            compute_units=int(fields.get("compute", fields.get("compute_units", 0))),
        )


def build_reserve_request(
    label: str,
    specs: list[DeviceConstraint | DeviceAllocation],
    solver: ConsolidateSolver | SpreadSolver,
    devices: list[dict] | None = None,
) -> ReserveRequest:
    """Build the explicit daemon request for already-parsed CLI device specs."""
    if all(isinstance(s, DeviceAllocation) for s in specs):
        # Every spec is explicit (or there are none): no solver step.
        return ReserveRequest(reservations=(
            ReservationRequest(label=label, allocations=tuple(specs)),
        ))

    if devices is None:
        raise ValueError("device status is required for device constraints")
    return solver.solve([(label, specs)], devices)


def build_exec_env(
    base_env: dict[str, str],
    granted_allocations: list[dict],
    reservation_id: str,
    lease_fd: int | None,
) -> dict[str, str]:
    """Pass granted allocations and CUDA visibility to the child process."""
    physical_names = [a["name"] for a in granted_allocations]
    cuda_uuids: list[str] = []
    for name in physical_names:
        if name.partition(":")[0] == "cuda":
            token = cuda_uuid(name)
            if token not in cuda_uuids:
                cuda_uuids.append(token)

    first_accel = next(
        (name for name in physical_names if not name.startswith("cpu")),
        physical_names[0] if physical_names else "",
    )

    env = base_env.copy()
    if cuda_uuids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_uuids)
    env["RESERVED_DEVICE"] = first_accel
    env["RESERVED_DEVICES"] = ",".join(physical_names)
    # Preserve allocation boundaries even when a configured name contains a comma.
    env["RESERVED_GRANTED_ALLOCATIONS"] = json.dumps(granted_allocations, separators=(",", ":"))
    env["RESERVED_ID"] = reservation_id
    if lease_fd is not None:
        env["RESERVED_FD"] = str(lease_fd)
    else:
        # Drop any stale lease number from the environment passed to exec.
        env.pop("RESERVED_FD", None)
    return env


def main(argv: list[str] | None = None) -> None:
    # Parse only reserve's arguments so the worker gets its exact argument vector.
    argv = sys.argv[1:] if argv is None else argv
    if "--" in argv:
        sep = argv.index("--")
        reserve_args = argv[:sep]
        cmd_args = argv[sep + 1 :]
    else:
        reserve_args = argv
        cmd_args = []

    args = parse_options(_USAGE, reserve_args)
    if not cmd_args:
        usage_error("no command specified after '--'")
    solver_name = args["--solver"]
    if solver_name not in {"consolidate", "spread"}:
        usage_error("--solver must be consolidate or spread")

    socket_path = _get_socket_override(args["--socket"])
    solver = SpreadSolver() if solver_name == "spread" else ConsolidateSolver()

    specs = [parse_device_spec(text) for text in args["DEVICE"]]
    label = args["LABEL"]

    if all(isinstance(s, DeviceAllocation) for s in specs):
        req = build_reserve_request(label, specs, solver)
    else:
        # Run solver.
        try:
            status = _query_status(socket_path)
        except DaemonNotRunningError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)

        try:
            req = build_reserve_request(label, specs, solver, status["devices"])
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
    granted_allocations = granted["allocations"]

    # We have exactly one lease fd.
    lease_fd = fds[0] if fds else None

    env = build_exec_env(os.environ, granted_allocations, reservation_id, lease_fd)
    if lease_fd is not None:
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
