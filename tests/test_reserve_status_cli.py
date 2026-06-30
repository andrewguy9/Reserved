"""
Invariant-focused tests for reserve-status CLI behavior.
"""

import json
import sys

import pytest

from reserved.bin import reserve_status
from reserved.sdk.errors import DaemonNotRunningError


STATUS = {
    "devices": [
        {
            "name": "cuda:0",
            "type": "cuda",
            "model": "rtx3090",
            "ram_mb": 24576,
            "available_ram_mb": 16384,
            "compute_units": 100,
            "available_compute_units": 75,
        },
        {
            "name": "cpu",
            "type": "cpu",
            "model": None,
            "ram_mb": 65536,
            "available_ram_mb": 65536,
            "compute_units": 0,
            "available_compute_units": 0,
        },
    ],
    "reservations": [
        {
            "id": "abc123",
            "label": "train",
            "created_at": 100.0,
            "slots": [
                {"device_name": "cuda:0", "ram_mb": 8192, "compute_units": 25},
            ],
        },
    ],
}


def test_format_human_includes_devices_reservations_and_capacity(monkeypatch):
    monkeypatch.setattr(reserve_status.time, "time", lambda: 160.0)

    out = reserve_status._format_human(STATUS)

    for expected in [
        "Devices:",
        "cuda:0",
        "rtx3090",
        "16384/24576 MB",
        "75/100 compute",
        "cpu",
        "Reservations:",
        "abc123",
        "label=train",
        "age=1m0s",
        "8192 MB",
        "25 compute",
    ]:
        assert expected in out


def test_format_human_reports_empty_reservations():
    out = reserve_status._format_human({"devices": [], "reservations": []})

    assert "Devices:" in out
    assert "Reservations:" in out
    assert "(none)" in out


def test_main_json_outputs_machine_readable_status(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["reserve-status", "--socket", "/tmp/sock", "--json"])
    monkeypatch.setattr(reserve_status, "_query_status", lambda socket_path: STATUS)

    reserve_status.main()

    parsed = json.loads(capsys.readouterr().out)
    assert parsed == STATUS


def test_main_daemon_not_running_exits_2(monkeypatch, capsys):
    def raise_not_running(socket_path):
        raise DaemonNotRunningError("missing")

    monkeypatch.setattr(sys, "argv", ["reserve-status"])
    monkeypatch.setattr(reserve_status, "_query_status", raise_not_running)

    with pytest.raises(SystemExit) as exc:
        reserve_status.main()

    assert exc.value.code == 2
    assert "missing" in capsys.readouterr().err
