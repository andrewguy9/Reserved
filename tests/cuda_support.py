"""Shared CUDA test support: UUID-form device names and a stand-in torch module."""

import types
import uuid


def cuda_name(n: int) -> str:
    """A syntactically valid ``cuda:GPU-<uuid>`` name; ``n`` makes it distinct."""
    return f"cuda:GPU-00000000-0000-0000-0000-{n:012d}"


def fake_torch(*visible_names: str, with_uuid: bool = True) -> types.SimpleNamespace:
    """A torch stand-in whose CUDA devices report the UUIDs of *visible_names* in order.

    ``torch.device`` is ``str`` so tests can compare against plain device strings.
    """
    def get_device_properties(index: int) -> types.SimpleNamespace:
        properties = types.SimpleNamespace(name=f"Fake GPU {index}")
        if with_uuid:
            token = visible_names[index].partition(":")[2]  # "GPU-<uuid>"
            properties.uuid = types.SimpleNamespace(bytes=list(uuid.UUID(token[4:]).bytes))
        return properties

    cuda = types.SimpleNamespace(
        device_count=lambda: len(visible_names),
        get_device_properties=get_device_properties,
    )
    return types.SimpleNamespace(device=str, cuda=cuda)
