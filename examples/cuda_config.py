"""Generate a reserved config for the NVIDIA GPUs torch can see on this machine.

Reads each card's GPU UUID, total memory, and streaming multiprocessor count
from torch's device properties and prints a config whose device names are the
``cuda:GPU-<uuid>`` form the daemon requires. ``ram_mb`` is the card's total
memory in MiB and ``compute_units`` is its multiprocessor count. Pass
``--ram-fraction`` to commit only part of each card's memory to reservations.

Run it unmasked (no CUDA_VISIBLE_DEVICES) so every card is listed. PyTorch
with CUDA support must be importable in the running interpreter.

    python examples/cuda_config.py > ~/.config/reserved/reserved.json
    python examples/cuda_config.py --ram-fraction 0.9 > reserved.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from reserved.core import load_ledger
from reserved.sdk import torch_cuda_uuid

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ram-fraction", type=float, default=1.0,
        help="fraction of each card's memory to commit to reservations (default: 1.0)",
    )
    args = parser.parse_args(argv)
    try:
        import torch  # type: ignore[import]
    except ImportError as exc:
        print(f"error: PyTorch with CUDA support is required to read the cards: {exc}", file=sys.stderr)
        return 1
    try:
        sys.stdout.write(render_config(torch, args.ram_fraction))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
