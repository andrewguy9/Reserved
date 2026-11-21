"""The CUDA config generator turns torch's device properties into a config the daemon accepts."""

import importlib.util
import json
from pathlib import Path

import pytest

from tests.cuda_support import cuda_name, fake_torch

_SPEC = importlib.util.spec_from_file_location(
    "cuda_config", Path(__file__).resolve().parent.parent / "examples" / "cuda_config.py",
)
cuda_config = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cuda_config)

CARD0 = "cuda:GPU-d0a59d46-9b93-1c68-c829-ef47cc823ecd"
CARD1 = "cuda:GPU-8999ca79-2402-c426-afbf-0d64aee6aa2c"


def test_devices_carry_uuid_memory_and_multiprocessor_count():
    torch = fake_torch(CARD0, CARD1, total_memory_mib=49140, multi_processor_count=84)
    assert cuda_config.devices_from_torch(torch) == [
        {"name": CARD0, "type": "cuda", "ram_mb": 49140, "compute_units": 84},
        {"name": CARD1, "type": "cuda", "ram_mb": 49140, "compute_units": 84},
    ]


def test_memory_is_reported_in_whole_mib():
    torch = fake_torch(CARD0)
    properties = torch.cuda.get_device_properties(0)
    properties.total_memory = 2 * cuda_config.MIB + 1  # a stray byte must not round up
    torch.cuda.get_device_properties = lambda index: properties
    assert cuda_config.devices_from_torch(torch)[0]["ram_mb"] == 2


def test_ram_fraction_scales_each_budget_down():
    devices = cuda_config.devices_from_torch(fake_torch(CARD0, CARD1), ram_fraction=0.5)
    assert [d["ram_mb"] for d in devices] == [24570, 24570]
    assert [d["compute_units"] for d in devices] == [84, 84]  # compute is not scaled


@pytest.mark.parametrize("fraction", [0, -0.1, 1.5])
def test_ram_fraction_outside_unit_interval_is_rejected(fraction):
    with pytest.raises(ValueError, match="ram_fraction"):
        cuda_config.devices_from_torch(fake_torch(CARD0), ram_fraction=fraction)


def test_rendered_config_is_valid_json_the_daemon_accepts():
    text = cuda_config.render_config(fake_torch(CARD0, CARD1))
    config = json.loads(text)
    assert [d["name"] for d in config["devices"]] == [CARD0, CARD1]
    assert text.endswith("}\n")


def test_no_visible_cards_renders_an_empty_device_list():
    assert json.loads(cuda_config.render_config(fake_torch())) == {"devices": []}


def test_torch_without_device_uuids_is_rejected():
    with pytest.raises(Exception, match="does not report CUDA device UUIDs"):
        cuda_config.render_config(fake_torch(cuda_name(0), with_uuid=False))
