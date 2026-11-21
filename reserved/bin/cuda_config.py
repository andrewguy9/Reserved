#!/usr/bin/env python3
"""Generate a reserved config for the NVIDIA GPUs torch can see on this machine."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# Ensure the package is importable when run from the repo root.
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

from reserved.cli import parse_options, usage_error
from reserved.core import load_ledger
from reserved.sdk import torch_cuda_uuid


_USAGE = """Generate a reserved config for the NVIDIA GPUs torch can see.

Usage:
    reserved-cuda-config [--ram-fraction F]
    reserved-cuda-config (-h | --help)

Options:
    -h --help         Show this help.
    --ram-fraction F  Fraction of each card's memory to commit to
                      reservations, in (0, 1] [default: 1.0].

Reads each card's GPU UUID, total memory, and streaming multiprocessor count
from torch's device properties and prints a config on stdout. Device names are
cuda:GPU-<uuid>, ram_mb is the card's memory in MiB, and compute_units is its
multiprocessor count. Run it with no CUDA_VISIBLE_DEVICES so every card is
listed. PyTorch with CUDA support must be importable.

    reserved-cuda-config > ~/.config/reserved/reserved.json
"""

MIB = 1024 * 1024


def devices_from_torch(torch_module: Any, ram_fraction: float = 1.0) -> list[dict]:
    """Return one device record per CUDA device *torch_module* reports.

    ``ram_fraction`` scales each card's memory budget and must be in (0, 1].
    """
    if not 0 < ram_fraction <= 1:
        raise ValueError(f"ram_fraction must be in (0, 1], got {ram_fraction}")
    devices = []
    for index in range(torch_module.cuda.device_count()):
        properties = torch_module.cuda.get_device_properties(index)
        devices.append({
            "name": f"cuda:{torch_cuda_uuid(properties)}",
            "type": "cuda",
            "ram_mb": int(properties.total_memory // MIB * ram_fraction),
            "compute_units": properties.multi_processor_count,
        })
    return devices


def render_config(torch_module: Any, ram_fraction: float = 1.0) -> str:
    """Return the config JSON for the visible cards, validated the way the daemon validates it."""
    config = {"devices": devices_from_torch(torch_module, ram_fraction)}
    load_ledger(config)  # raises ValueError if a name or field would be rejected
    return json.dumps(config, indent=2) + "\n"


def main(argv: list[str] | None = None) -> None:
    args = parse_options(_USAGE, argv)
    try:
        ram_fraction = float(args["--ram-fraction"])
    except ValueError:
        usage_error(f"--ram-fraction must be a number, got {args['--ram-fraction']!r}")

    try:
        import torch  # type: ignore[import]
    except ImportError as exc:
        print(f"error: PyTorch with CUDA support is required to read the cards: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        sys.stdout.write(render_config(torch, ram_fraction))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
