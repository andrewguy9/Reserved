"""The CUDA config generator turns nvidia-smi output into a config the daemon accepts."""

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "cuda_config", Path(__file__).resolve().parent.parent / "examples" / "cuda_config.py",
)
cuda_config = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cuda_config)

SAMPLE = """\
GPU-d0a59d46-9b93-1c68-c829-ef47cc823ecd, 49140
GPU-8999ca79-2402-c426-afbf-0d64aee6aa2c, 49140
"""


def test_parse_reads_uuid_and_memory():
    assert cuda_config.parse_nvidia_smi(SAMPLE) == [
        {"name": "cuda:GPU-d0a59d46-9b93-1c68-c829-ef47cc823ecd", "type": "cuda", "ram_mb": 49140},
        {"name": "cuda:GPU-8999ca79-2402-c426-afbf-0d64aee6aa2c", "type": "cuda", "ram_mb": 49140},
    ]


def test_ram_fraction_scales_each_budget_down():
    devices = cuda_config.parse_nvidia_smi(SAMPLE, ram_fraction=0.5)
    assert [d["ram_mb"] for d in devices] == [24570, 24570]


@pytest.mark.parametrize("fraction", [0, -0.1, 1.5])
def test_ram_fraction_outside_unit_interval_is_rejected(fraction):
    with pytest.raises(ValueError, match="ram_fraction"):
        cuda_config.parse_nvidia_smi(SAMPLE, ram_fraction=fraction)


def test_rendered_config_is_valid_json_the_daemon_accepts():
    text = cuda_config.render_config(SAMPLE)
    config = json.loads(text)
    assert [d["name"] for d in config["devices"]] == [
        "cuda:GPU-d0a59d46-9b93-1c68-c829-ef47cc823ecd",
        "cuda:GPU-8999ca79-2402-c426-afbf-0d64aee6aa2c",
    ]
    assert text.endswith("}\n")


def test_render_rejects_names_the_daemon_would_reject():
    with pytest.raises(ValueError, match="invalid CUDA device name"):
        cuda_config.render_config("GPU-NOT-A-UUID, 49140\n")
