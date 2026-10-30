"""Worker recommendations against real ledger snapshots and SDK acquisitions."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import socket

import pytest

from reserved import core, protocol
from reserved.sdk import (
    Backlog, DeviceConstraint, ExplicitSlot, OwnedWorker, Request, Reservation,
    ReservationError, WorkerPlan, WorkerShortfall, plan_workers, recommend_workers,
)
from reserved.sdk import client


def make_ledger(count=2, *, ram_mb=24576, compute_units=100):
    return core.load_ledger({"devices": [
        {"name": f"cuda:{index}", "type": "cuda", "model": "test-gpu",
         "ram_mb": ram_mb, "compute_units": compute_units}
        for index in range(count)
    ]})


def snapshot(ledger):
    return protocol.parse_response(protocol.status_result_to_wire(core.status(ledger)))


def backlog(workload_id="a", **changes):
    return replace(Backlog(
        workload_id, queued_requests=10000,
        worker=DeviceConstraint("cuda", ram_mb=8192, compute_units=50),
        max_workers=8, batch_size=16, batch_seconds=2,
        startup_seconds=30, shutdown_seconds=5,
    ), **changes)


def hold(ledger, name="cuda:0", *, workload_id="a", state="accepting",
         buffered_requests=0, ram_mb=8192, compute_units=50):
    ledger, result = core.grant(ledger, core.ReserveRequest((
        core.ReservationRequest(workload_id, (core.SlotRequest(name, ram_mb, compute_units),)),
    )))
    assert result.ok
    worker = OwnedWorker(result.reservations[0].reservation_id, workload_id,
                         state, buffered_requests)
    return ledger, worker


def test_large_queue_uses_both_devices_without_mutating_inputs():
    status = snapshot(make_ledger())
    original = deepcopy(status)
    demand = backlog()
    requirement = deepcopy(demand.worker)

    plan = plan_workers([demand], [], status)

    assert Counter(start.slot.name for start in plan.start) == {"cuda:0": 2, "cuda:1": 2}
    assert plan.shortfall == {"a": WorkerShortfall(4, ("capacity",))}
    assert plan.keep == plan.drain == plan.expired == ()
    assert status == original
    assert demand.worker == requirement


def test_polling_held_workers_does_not_subtract_their_capacity_twice():
    ledger, owned = hold(make_ledger())
    plan = plan_workers([backlog()], [owned], snapshot(ledger))
    assert plan.keep == (owned.reservation_id,)
    assert len(plan.start) == 3
    assert Counter(start.slot.name for start in plan.start) == {"cuda:0": 1, "cuda:1": 2}


def test_multiple_workloads_share_capacity_and_ignore_input_order():
    a, b = backlog("a"), backlog("b")
    status = snapshot(make_ledger())
    plan = plan_workers([b, a], [], status)
    reversed_status = {"devices": list(reversed(status["devices"])), "reservations": []}
    assert plan == plan_workers([a, b], [], reversed_status)
    assert [start.workload_id for start in plan.start] == ["a", "b", "a", "b"]


def test_unplaceable_workload_does_not_block_another():
    impossible = backlog("a", worker=DeviceConstraint("metal", ram_mb=8192))
    plan = plan_workers([impossible, backlog("b")], [], snapshot(make_ledger()))
    assert [start.workload_id for start in plan.start] == ["b"] * 4
    assert plan.shortfall["a"] == WorkerShortfall(8, ("capacity",))


def test_hardware_filter_and_ram_are_respected():
    ledger = make_ledger(compute_units=1000)
    demand = backlog(worker=DeviceConstraint("cuda", model="test-gpu", ram_mb=16384))
    plan = plan_workers([demand], [], snapshot(ledger))
    assert Counter(start.slot.name for start in plan.start) == {"cuda:0": 1, "cuda:1": 1}
    missing = replace(demand, worker=DeviceConstraint("cuda", model="other", ram_mb=16384))
    assert plan_workers([missing], [], snapshot(ledger)).start == ()


def test_batch_capacity_and_worker_cap_bound_small_queues():
    plan = plan_workers([backlog(queued_requests=17, startup_seconds=0,
                                 shutdown_seconds=0, max_workers=1)], [],
                        snapshot(make_ledger()))
    assert len(plan.start) == 1
    assert plan.shortfall == {}


def test_small_queue_bootstraps_one_worker_but_does_not_expand_a_warm_pool():
    small = backlog(queued_requests=2)
    assert len(plan_workers([small], [], snapshot(make_ledger())).start) == 1
    ledger, worker = hold(make_ledger(), buffered_requests=16)
    plan = plan_workers([small], [worker], snapshot(ledger))
    assert plan.keep == (worker.reservation_id,)
    assert plan.start == plan.drain == ()


def test_startup_guard_accounts_for_work_already_buffered():
    ledger, worker = hold(make_ledger())
    demand = backlog(queued_requests=500)
    plan = plan_workers([demand], [worker], snapshot(ledger))
    assert plan.start == ()
    assert plan.shortfall == {"a": WorkerShortfall(1, ("startup_cost",))}

    busy = replace(worker, buffered_requests=240)
    plan = plan_workers([demand], [busy], snapshot(ledger))
    assert len(plan.start) == 1
    assert plan.shortfall == {}


def test_loading_workers_suppress_duplicate_expansion():
    ledger, worker = hold(make_ledger(), state="starting")
    plan = plan_workers([backlog(queued_requests=1000)], [worker], snapshot(ledger))
    assert plan.keep == (worker.reservation_id,)
    assert plan.start == plan.drain == ()
    assert plan.shortfall == {"a": WorkerShortfall(3, ("starting",))}


def test_loading_workers_do_not_replace_ready_workers_during_retirement():
    ledger, loading = hold(make_ledger(), state="starting")
    ledger, ready = hold(ledger, "cuda:1", buffered_requests=16)
    plan = plan_workers([backlog(queued_requests=100)], [loading, ready], snapshot(ledger))
    assert plan.keep == tuple(sorted((loading.reservation_id, ready.reservation_id)))
    assert plan.start == plan.drain == ()


def test_queue_tapers_accepting_pool_while_buffered_workers_keep_their_leases():
    ledger = make_ledger()
    workers = []
    for index, buffered in enumerate((128, 64, 32, 16)):
        ledger, worker = hold(ledger, f"cuda:{index // 2}", buffered_requests=buffered)
        workers.append(worker)
    status = snapshot(ledger)
    original = deepcopy(status)

    for queued, next_worker in zip((600, 300, 100, 0), reversed(workers)):
        plan = plan_workers([backlog(queued_requests=queued)], workers, status)
        assert plan.drain == (next_worker.reservation_id,)
        assert plan.start == ()
        workers = [replace(worker, state="draining") if worker.reservation_id in plan.drain
                   else worker for worker in workers]

    plan = plan_workers([backlog(queued_requests=0)], workers, status)
    assert plan.drain == plan.start == ()
    assert set(plan.keep) == {worker.reservation_id for worker in workers}
    assert status == original


def test_drain_ties_are_deterministic():
    ledger, first = hold(make_ledger())
    ledger, second = hold(ledger)
    plan = plan_workers([backlog(queued_requests=100)], [second, first], snapshot(ledger))
    assert plan.drain == (min(first.reservation_id, second.reservation_id),)


def test_draining_is_not_reversed_if_more_work_arrives_and_counts_against_cap():
    ledger, retiring = hold(make_ledger(), state="draining", buffered_requests=16)
    ledger, ready = hold(ledger, "cuda:1", buffered_requests=16)
    plan = plan_workers([backlog(max_workers=2)], [retiring, ready], snapshot(ledger))
    assert set(plan.keep) == {retiring.reservation_id, ready.reservation_id}
    assert plan.start == plan.drain == ()
    assert plan.shortfall == {"a": WorkerShortfall(1, ("worker_limit",))}


def test_draining_and_stopping_do_not_provide_free_capacity_for_other_models():
    ledger, worker = hold(make_ledger(1), ram_mb=24576, compute_units=100)
    demands = [backlog("a", queued_requests=0), backlog("b")]
    plan = plan_workers(demands, [worker], snapshot(ledger))
    assert plan.drain == (worker.reservation_id,)
    assert plan.start == ()

    for state in ("draining", "stopping"):
        plan = plan_workers(demands, [replace(worker, state=state)], snapshot(ledger))
        assert plan.keep == (worker.reservation_id,)
        assert plan.start == plan.drain == ()
    released = core.release(ledger, worker.reservation_id)
    assert len(plan_workers(demands, [], snapshot(released)).start) == 2


def test_unlisted_reservations_are_fixed_even_when_their_labels_match():
    ledger, _ = hold(make_ledger(1), ram_mb=24576, compute_units=100)
    plan = plan_workers([backlog()], [], snapshot(ledger))
    assert plan.start == plan.keep == plan.drain == ()
    assert plan.shortfall == {"a": WorkerShortfall(8, ("capacity",))}


def test_zero_cap_drains_ready_workers_but_keeps_loading_workers():
    ledger, ready = hold(make_ledger(), buffered_requests=32)
    ledger, loading = hold(ledger, "cuda:1", state="starting")
    plan = plan_workers([backlog(max_workers=0)], [ready, loading], snapshot(ledger))
    assert plan.drain == (ready.reservation_id,)
    assert plan.keep == (loading.reservation_id,)
    assert plan.start == ()
    assert plan.shortfall == {}


def test_missing_owned_id_suppresses_all_actions():
    ledger, worker = hold(make_ledger())
    missing = OwnedWorker("expired", "a", "accepting", 16)
    plan = plan_workers([backlog(queued_requests=0), backlog("b")],
                        [worker, missing], snapshot(ledger))
    assert plan == WorkerPlan(expired=("expired",))


@pytest.mark.parametrize("changes", [
    {"workload_id": ""}, {"queued_requests": -1}, {"queued_requests": True},
    {"queued_requests": 1.5}, {"max_workers": -1}, {"batch_size": 0},
    {"batch_size": False}, {"batch_seconds": 0}, {"batch_seconds": float("nan")},
    {"startup_seconds": -1}, {"startup_seconds": float("inf")},
    {"shutdown_seconds": float("nan")}, {"shutdown_seconds": "5"},
    {"worker": DeviceConstraint("cuda")},
    {"worker": DeviceConstraint(None, ram_mb=8192)},
    {"worker": DeviceConstraint("cuda", ram_mb=8192, compute_units=-1)},
    {"worker": ExplicitSlot("cuda:0", ram_mb=8192)},
    {"worker": [DeviceConstraint("cuda", ram_mb=8192)]},
    {"startup_seconds": 1e308, "shutdown_seconds": 1e308},
])
def test_invalid_backlog_declarations_fail_clearly(changes):
    with pytest.raises(ValueError):
        plan_workers([backlog(**changes)], [], snapshot(make_ledger()))


@pytest.mark.parametrize("worker", [
    OwnedWorker("", "a"), OwnedWorker("id", "unknown"),
    OwnedWorker("id", "a", "unknown"), OwnedWorker("id", "a", buffered_requests=-1),
    OwnedWorker("id", "a", "stopping", buffered_requests=1),
])
def test_invalid_worker_declarations_fail_clearly(worker):
    with pytest.raises(ValueError):
        plan_workers([backlog()], [worker], snapshot(make_ledger()))


def test_duplicate_workloads_and_workers_are_rejected():
    status = snapshot(make_ledger())
    with pytest.raises(ValueError, match="duplicate workload"):
        plan_workers([backlog(), backlog()], [], status)
    with pytest.raises(ValueError, match="duplicate reservation"):
        plan_workers([backlog()], [OwnedWorker("id", "a")] * 2, status)


def test_owned_claim_must_cover_the_declared_requirement():
    ledger, worker = hold(make_ledger(), ram_mb=4096)
    with pytest.raises(ValueError, match="does not cover"):
        plan_workers([backlog()], [worker], snapshot(ledger))
    ledger, result = core.grant(make_ledger(), core.ReserveRequest((
        core.ReservationRequest("a", (core.SlotRequest("cuda:0", 8192, 50),
                                      core.SlotRequest("cuda:1", 8192, 50))),
    )))
    worker = OwnedWorker(result.reservations[0].reservation_id, "a")
    with pytest.raises(ValueError, match="exactly one slot"):
        plan_workers([backlog()], [worker], snapshot(ledger))


def test_empty_managed_set_leaves_other_reservations_alone():
    ledger, _ = hold(make_ledger())
    assert plan_workers([], [], snapshot(ledger)) == WorkerPlan()


@pytest.fixture
def ledger_transport(monkeypatch):
    """Fake IO only: real SDK requests, ledger grants, protocol, and lease EOF."""
    class Transport:
        ledger = make_ledger()

        def __init__(self):
            self.leases = {}
            self.paths = []

        def query(self, path):
            self.paths.append(path)
            for reservation_id, lease in list(self.leases.items()):
                try:
                    closed = lease.recv(1, socket.MSG_DONTWAIT) == b""
                except BlockingIOError:
                    closed = False
                if closed:
                    self.ledger = core.release(self.ledger, reservation_id)
                    self.leases.pop(reservation_id).close()
            return snapshot(self.ledger)

        def reserve(self, path, request, num_reservations):
            assert num_reservations == len(request.reservations)
            self.ledger, result = core.grant(self.ledger, request)
            if not result.ok:
                return protocol.parse_response(protocol.grant_failure_to_wire(result)), []
            fds = []
            for reservation in result.reservations:
                local, remote = socket.socketpair()
                self.leases[reservation.reservation_id] = remote
                fds.append(local.detach())
            return protocol.parse_response(protocol.grant_success_to_wire(result)), fds

    transport = Transport()
    monkeypatch.setattr(client, "_query_status", transport.query)
    monkeypatch.setattr(client, "_send_reserve", transport.reserve)
    yield transport
    for lease in transport.leases.values():
        lease.close()


def test_recommend_acquire_poll_and_recover_from_an_acquisition_race(ledger_transport,
                                                                  monkeypatch):
    monkeypatch.setenv("RESERVED_SOCKET", "/example.sock")
    demand = backlog(max_workers=2)
    initial = recommend_workers([demand], [])
    reservations = Request(*[
        Reservation(start.workload_id, start.slot) for start in initial.start
    ]).acquire()
    try:
        owned = [OwnedWorker(reservation.reservation_id, "a") for reservation in reservations]
        again = recommend_workers([demand], owned)
        assert len(again.keep) == 2
        assert again.start == ()
        assert ledger_transport.paths[-1] == "/example.sock"

        demand = replace(demand, max_workers=4)
        owned = [replace(worker, state="accepting", buffered_requests=16) for worker in owned]
        proposal = recommend_workers([demand], owned, socket_path="/override.sock")
        assert len(proposal.start) == 2
        assert ledger_transport.paths[-1] == "/override.sock"

        ledger_transport.ledger, _ = hold(
            ledger_transport.ledger, "cuda:0", workload_id="other",
            ram_mb=16384, compute_units=50,
        )
        before = ledger_transport.ledger
        with pytest.raises(ReservationError):
            Request(*[Reservation(start.workload_id, start.slot)
                      for start in proposal.start]).acquire()
        assert ledger_transport.ledger == before
        for reservation in reservations:
            reservation.check()

        fresh = recommend_workers([demand], owned)
        assert set(fresh.keep) == {worker.reservation_id for worker in owned}
        assert [start.slot.name for start in fresh.start] == ["cuda:1"]
    finally:
        for reservation in reservations:
            reservation.release()
    remaining = ledger_transport.query("/example.sock")["reservations"]
    assert [reservation["label"] for reservation in remaining] == ["other"]


def test_recommend_propagates_daemon_connection_errors(monkeypatch):
    from reserved.sdk import DaemonNotRunningError

    def unavailable(path):
        raise DaemonNotRunningError(path)

    monkeypatch.setattr(client, "_query_status", unavailable)
    with pytest.raises(DaemonNotRunningError):
        recommend_workers([backlog()], [], socket_path="/missing.sock")
