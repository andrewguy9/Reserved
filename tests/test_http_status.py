"""Verify the HTML dashboard through its HTTP endpoint."""

from html.parser import HTMLParser
from http.server import HTTPServer
from threading import Thread
from types import SimpleNamespace
from urllib.request import urlopen

import pytest

from reserved import core, daemon


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append([])
        elif tag == "tr":
            self.tables[-1].append([])
        elif tag in ("td", "th"):
            self.cell = ""

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.tables[-1][-1].append(self.cell)
            self.cell = None


def fetch_dashboard(ledger):
    handler = type("DashboardHandler", (daemon._StatusHandler,), {
        "daemon_ref": SimpleNamespace(ledger=ledger),
    })
    with HTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=5) as response:
                assert response.status == 200
                assert response.headers.get_content_type() == "text/html"
                html = response.read().decode("utf-8")
        finally:
            server.shutdown()
            thread.join(timeout=5)
    parser = TableParser()
    parser.feed(html)
    return html, parser.tables


@pytest.mark.parametrize("allocations, expected", [
    ((core.DeviceAllocation("cuda:GPU-00000000-0000-0000-0000-000000000000", 1024, 1),), [
        ["abc123", "infer", "1m0s", "cuda:GPU-00000000-0000-0000-0000-000000000000", "1024 MB", "1"],
    ]),
    ((core.DeviceAllocation("cuda:GPU-00000000-0000-0000-0000-000000000000", 1024, 1),
      core.DeviceAllocation("cuda:GPU-00000000-0000-0000-0000-000000000002", 2048, 0)), [
        ["abc123", "infer", "1m0s", "cuda:GPU-00000000-0000-0000-0000-000000000000", "1024 MB", "1"],
        ["abc123", "infer", "1m0s", "cuda:GPU-00000000-0000-0000-0000-000000000002", "2048 MB", "0"],
    ]),
    ((), [["abc123", "infer", "1m0s", "—", "—", "—"]]),
])
def test_reservation_values_match_their_column_headers(monkeypatch, allocations, expected):
    monkeypatch.setattr(daemon.time, "time", lambda: 160.0)
    ledger = core.Ledger(availability=(), reservations=(
        core.ActiveReservation("abc123", "infer", allocations, 100.0),
    ))

    _, tables = fetch_dashboard(ledger)

    assert tables[1] == [["ID", "Label", "Age", "Device", "RAM", "Compute"], *expected]


def test_empty_reservations_span_all_columns():
    html, tables = fetch_dashboard(core.Ledger(availability=(), reservations=()))

    assert tables[1] == [
        ["ID", "Label", "Age", "Device", "RAM", "Compute"],
        ["(none)"],
    ]
    assert 'colspan="6"' in html


def test_device_text_and_reservation_labels_are_displayed_literally():
    name = "cpu<&>"
    label = '<script>alert("job")</script> & train'
    ledger = core.load_ledger({"devices": [
        {"name": name, "type": "cpu", "model": "<board>", "ram_mb": 2048},
    ]})
    ledger, result = core.grant(ledger, core.ReserveRequest(reservations=(
        core.ReservationRequest(label, (core.DeviceAllocation(name, ram_mb=256),)),
    )))
    assert result.ok

    html, tables = fetch_dashboard(ledger)

    assert tables[0][1] == [name, "cpu", "<board>", "1792/2048 MB", "0/0"]
    assert tables[1][1][1] == label
    assert tables[1][1][3:] == [name, "256 MB", "0"]
    assert "<script>" not in html
    assert "<board>" not in html
