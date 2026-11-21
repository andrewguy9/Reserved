"""
Invariant-focused tests for the reserve CLI helpers.
"""

import json

import pytest

from reserved.bin.reserve import build_exec_env, build_reserve_request, main, parse_device_spec
from reserved.core import DeviceAllocation, ReservationRequest, ReserveRequest
from reserved.sdk.client import DeviceConstraint
from reserved.sdk.solvers import ConsolidateSolver


def test_parse_device_spec_ram_and_compute_aliases_are_equivalent():
    short = parse_device_spec("type=cuda,model=a6000,ram=8192,compute=25")
    long = parse_device_spec("type=cuda,model=a6000,ram_mb=8192,compute_units=25")

    assert isinstance(short, DeviceConstraint)
    assert isinstance(long, DeviceConstraint)
    assert short == long


def test_parse_device_spec_name_selects_explicit_allocation():
    spec = parse_device_spec("name=cuda:GPU-00000000-0000-0000-0000-000000000002,type=cuda,ram=4096,compute=50")

    assert spec == DeviceAllocation(
        name="cuda:GPU-00000000-0000-0000-0000-000000000002", ram_mb=4096, compute_units=50,
    )


def test_parse_device_spec_rejects_fields_without_key_value_shape():
    with pytest.raises(SystemExit):
        parse_device_spec("type=cuda,ram")


@pytest.mark.parametrize("argv, message", [
    (["job", "--"], "no command specified"),
    (["job", "--solver=invalid", "--", "true"], "--solver must be"),
])
def test_main_rejects_invalid_invocations(argv, message, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_build_reserve_request_for_explicit_allocations_passes_them_through():
    specs = [
        DeviceAllocation("cuda:GPU-00000000-0000-0000-0000-000000000001", ram_mb=4096, compute_units=25),
        DeviceAllocation("cpu", ram_mb=1024),
    ]

    req = build_reserve_request("job", specs, ConsolidateSolver())

    assert req.reservations == (
        ReservationRequest(label="job", allocations=(
            DeviceAllocation(name="cuda:GPU-00000000-0000-0000-0000-000000000001", ram_mb=4096, compute_units=25),
            DeviceAllocation(name="cpu", ram_mb=1024, compute_units=0),
        )),
    )


def test_build_reserve_request_without_devices_is_valid_marker_reservation():
    req = build_reserve_request("marker", [], ConsolidateSolver())

    assert req.reservations == (
        ReservationRequest(label="marker", allocations=()),
    )


def test_build_reserve_request_delegates_constraints_to_solver():
    class RecordingSolver:
        def __init__(self):
            self.calls = []

        def solve(self, specs, devices):
            self.calls.append((specs, devices))
            return ReserveRequest(reservations=(
                ReservationRequest(label=specs[0][0], allocations=(
                    DeviceAllocation(name=devices[0]["name"], ram_mb=specs[0][1][0].ram_mb),
                )),
            ))

    solver = RecordingSolver()
    devices = [{"name": "cuda:GPU-00000000-0000-0000-0000-000000000000"}]
    specs = [DeviceConstraint("cuda", ram_mb=2048)]

    req = build_reserve_request("job", specs, solver, devices)

    assert solver.calls == [([("job", specs)], devices)]
    assert req.reservations[0].allocations[0] == DeviceAllocation(name="cuda:GPU-00000000-0000-0000-0000-000000000000", ram_mb=2048)


def test_build_reserve_request_requires_status_for_constraints():
    with pytest.raises(ValueError, match="device status"):
        build_reserve_request("job", [DeviceConstraint("cuda")], ConsolidateSolver())


def test_build_exec_env_sets_reservation_metadata_without_mutating_base_env():
    base = {"KEEP": "yes"}
    granted_allocations = [
        {"name": "cpu:batch,worker", "ram_mb": 1024, "compute_units": 0},
        {"name": "cuda:GPU-00000000-0000-0000-0000-000000000002", "ram_mb": 4096, "compute_units": 0},
        {"name": "cuda:GPU-00000000-0000-0000-0000-000000000002", "ram_mb": 2048, "compute_units": 0},
        {"name": "metal:0", "ram_mb": 512, "compute_units": 0},
        {"name": "cuda:GPU-00000000-0000-0000-0000-000000000003", "ram_mb": 4096, "compute_units": 0},
    ]

    env = build_exec_env(base, granted_allocations, "abc123", 9)

    assert base == {"KEEP": "yes"}
    assert env["KEEP"] == "yes"
    assert env["RESERVED_ID"] == "abc123"
    assert env["RESERVED_FD"] == "9"
    assert env["RESERVED_DEVICE"] == "cuda:GPU-00000000-0000-0000-0000-000000000002"
    assert env["RESERVED_DEVICES"] == "cpu:batch,worker,cuda:GPU-00000000-0000-0000-0000-000000000002,cuda:GPU-00000000-0000-0000-0000-000000000002,metal:0,cuda:GPU-00000000-0000-0000-0000-000000000003"
    assert env["CUDA_VISIBLE_DEVICES"] == (
        "GPU-00000000-0000-0000-0000-000000000002,GPU-00000000-0000-0000-0000-000000000003"
    )
    assert json.loads(env["RESERVED_GRANTED_ALLOCATIONS"]) == granted_allocations


def test_build_exec_env_without_cuda_allocations_does_not_add_cuda_visible_devices():
    env = build_exec_env(
        {"RESERVED_FD": "9"},
        [{"name": "metal:0", "ram_mb": 512, "compute_units": 0}],
        "rid",
        None,
    )

    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "RESERVED_FD" not in env
    assert env["RESERVED_DEVICE"] == "metal:0"
