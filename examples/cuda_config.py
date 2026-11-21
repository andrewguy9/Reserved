"""Generate a reserved config for the NVIDIA GPUs on this machine.

Reads each card's UUID, product name, and total memory from nvidia-smi and
prints a config whose device names are the ``cuda:GPU-<uuid>`` form the
daemon requires. Memory budgets default to the full card; pass
``--ram-fraction`` to commit only part of each card to reservations.

    python examples/cuda_config.py > ~/.config/reserved/reserved.json
    python examples/cuda_config.py --ram-fraction 0.9 > reserved.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from reserved.core import load_ledger

NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=uuid,name,memory.total",
    "--format=csv,noheader,nounits",
]


def model_label(product_name: str) -> str:
    """Turn an nvidia-smi product name into a config ``model`` label.

    ``NVIDIA RTX A6000`` becomes ``rtx-a6000``: the vendor prefix is dropped
    and the rest is lowercased with spaces replaced by hyphens. The label is
    informational; ``reserve-status`` shows it next to each device.
    """
    name = product_name.strip()
    if name.upper().startswith("NVIDIA "):
        name = name[len("NVIDIA "):]
    return "-".join(name.lower().split())


def parse_nvidia_smi(csv_text: str, ram_fraction: float = 1.0) -> list[dict]:
    """Convert ``nvidia-smi --query-gpu=uuid,name,memory.total`` CSV to device records.

    ``memory.total`` is reported in MiB, which is the unit of ``ram_mb``.
    ``ram_fraction`` scales each card's budget and must be in (0, 1].
    """
    if not 0 < ram_fraction <= 1:
        raise ValueError(f"ram_fraction must be in (0, 1], got {ram_fraction}")
    devices = []
    for line in csv_text.splitlines():
        if not line.strip():
            continue
        uuid, rest = line.split(",", 1)
        product_name, memory_mib = rest.rsplit(",", 1)
        devices.append({
            "name": f"cuda:{uuid.strip()}",
            "type": "cuda",
            "model": model_label(product_name),
            "ram_mb": int(int(memory_mib.strip()) * ram_fraction),
        })
    return devices


def render_config(csv_text: str, ram_fraction: float = 1.0) -> str:
    """Return the config JSON for *csv_text*, validated the way the daemon validates it."""
    config = {"devices": parse_nvidia_smi(csv_text, ram_fraction)}
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
        csv_text = subprocess.check_output(NVIDIA_SMI_QUERY, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"error: could not run nvidia-smi: {exc}", file=sys.stderr)
        return 1
    try:
        sys.stdout.write(render_config(csv_text, args.ram_fraction))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
