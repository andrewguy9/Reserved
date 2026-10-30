"""Advisory worker sizing from a shared queue, worker buffers, and daemon status.

The planner never acquires or releases resources. Applications apply starts
through Request, and drain buffered work before unloading and closing leases.
Timing estimates drive a conservative heuristic, not a throughput guarantee.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from . import client
from .client import DeviceConstraint, ExplicitSlot
from .solvers import SpreadSolver


WorkerState = Literal["starting", "accepting", "draining", "stopping"]
ShortfallReason = Literal["capacity", "startup_cost", "starting", "worker_limit"]


@dataclass(frozen=True)
class Backlog:
    """One workload's unassigned queue and rough per-worker requirements.

    queued_requests excludes work already buffered or executing on workers.
    batch_seconds includes normal batch processing; startup/shutdown describe
    loading and teardown. All compatible devices use these same estimates.
    """

    workload_id: str
    queued_requests: int
    worker: DeviceConstraint
    max_workers: int
    batch_size: int
    batch_seconds: float
    startup_seconds: float
    shutdown_seconds: float


@dataclass(frozen=True)
class OwnedWorker:
    """A held worker lease and its application-reported lifecycle state.

    buffered_requests includes both prefetched work and executing requests.
    Draining and stopping workers keep their leases until teardown finishes.
    """

    reservation_id: str
    workload_id: str
    state: WorkerState = "starting"
    buffered_requests: int = 0


@dataclass(frozen=True)
class WorkerStart:
    """Construct Reservation(workload_id, slot), then acquire before launching."""

    workload_id: str
    slot: ExplicitSlot


@dataclass(frozen=True)
class WorkerShortfall:
    """Uncovered target workers and the constraints encountered this round."""

    count: int
    reasons: tuple[ShortfallReason, ...]


@dataclass(frozen=True)
class WorkerPlan:
    """Recommendations for one snapshot; no resources are held by this object.

    keep continues each worker's current state, including retirement already
    underway. drain contains new retirement requests, not releases. shortfall
    omits workloads whose target is covered by kept, loading, or planned workers.
    Any expired ID suppresses all actions until the application reconciles.
    """

    keep: tuple[str, ...] = ()
    drain: tuple[str, ...] = ()
    start: tuple[WorkerStart, ...] = ()
    shortfall: dict[str, WorkerShortfall] = field(default_factory=dict)
    expired: tuple[str, ...] = ()


@dataclass
class _Placement:
    backlog: Backlog
    target: int
    covered: int
    budget: int
    reasons: set[ShortfallReason] = field(default_factory=set)
    added: int = 0
    blocked: bool = False


def _count(name: str, value: int, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _seconds(name: str, value: float, *, positive: bool = False) -> None:
    try:
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and (value > 0 if positive else value >= 0)
        )
    except OverflowError:
        valid = False
    if not valid:
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {bound} number")


def _identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _cost(backlog: Backlog) -> tuple[float, float]:
    rate = backlog.batch_size / backlog.batch_seconds
    window = max(
        backlog.batch_seconds, backlog.startup_seconds + backlog.shutdown_seconds
    )
    work = rate * window
    if not math.isfinite(rate) or rate <= 0 or not math.isfinite(work) or work <= 0:
        raise ValueError("timing estimates must produce finite positive worker costs")
    return rate, work


def _validate_inputs(backlogs: Sequence[Backlog], workers: Sequence[OwnedWorker]) -> None:
    ids: set[str] = set()
    for backlog in backlogs:
        if not isinstance(backlog, Backlog):
            raise ValueError("backlogs must contain Backlog objects")
        _identifier("workload_id", backlog.workload_id)
        if backlog.workload_id in ids:
            raise ValueError(f"duplicate workload_id: {backlog.workload_id!r}")
        ids.add(backlog.workload_id)
        _count("queued_requests", backlog.queued_requests)
        _count("max_workers", backlog.max_workers)
        _count("batch_size", backlog.batch_size, 1)
        _seconds("batch_seconds", backlog.batch_seconds, positive=True)
        _seconds("startup_seconds", backlog.startup_seconds)
        _seconds("shutdown_seconds", backlog.shutdown_seconds)
        if not isinstance(backlog.worker, DeviceConstraint):
            raise ValueError("worker must be a single DeviceConstraint")
        _identifier("worker.type", backlog.worker.type)
        if backlog.worker.model is not None:
            _identifier("worker.model", backlog.worker.model)
        _count("worker.ram_mb", backlog.worker.ram_mb, 1)
        _count("worker.compute_units", backlog.worker.compute_units)
        _cost(backlog)

    reservation_ids: set[str] = set()
    for worker in workers:
        if not isinstance(worker, OwnedWorker):
            raise ValueError("workers must contain OwnedWorker objects")
        _identifier("reservation_id", worker.reservation_id)
        if worker.reservation_id in reservation_ids:
            raise ValueError(f"duplicate reservation_id: {worker.reservation_id!r}")
        reservation_ids.add(worker.reservation_id)
        _identifier("worker.workload_id", worker.workload_id)
        if worker.workload_id not in ids:
            raise ValueError(f"unknown workload_id: {worker.workload_id!r}")
        if worker.state not in ("starting", "accepting", "draining", "stopping"):
            raise ValueError(f"unknown worker state: {worker.state!r}")
        _count("buffered_requests", worker.buffered_requests)
        if worker.state == "stopping" and worker.buffered_requests:
            raise ValueError("stopping workers must have no buffered requests")


def _validate_claim(
    worker: OwnedWorker,
    backlog: Backlog,
    reservation: dict,
    devices: dict[str, dict],
) -> None:
    slots = reservation["slots"]
    if len(slots) != 1:
        raise ValueError(f"worker {worker.reservation_id!r} must have exactly one slot")
    slot = slots[0]
    device = devices.get(slot["device_name"])
    requirement = backlog.worker
    if (
        device is None
        or device["type"] != requirement.type
        or (requirement.model is not None and device.get("model") != requirement.model)
        or slot["ram_mb"] < requirement.ram_mb
        or slot["compute_units"] < requirement.compute_units
    ):
        raise ValueError(
            f"reservation {worker.reservation_id!r} does not cover its workload's requirement"
        )


def plan_workers(
    backlogs: Sequence[Backlog],
    workers: Sequence[OwnedWorker],
    status: Mapping[str, Any],
) -> WorkerPlan:
    """Plan against a parsed daemon status without IO or input mutation.

    Keep reporting acquired workers while loading and retiring. A drain starts
    an application-managed transition: cease dispatch, finish buffered batches
    (including partial batches), unload, and only then close the lease.

    ValueError reports malformed declarations or incompatible owned claims.
    Missing reservation IDs instead produce an action-free expired plan.
    """
    backlogs, workers = tuple(backlogs), tuple(workers)
    _validate_inputs(backlogs, workers)
    by_id = {backlog.workload_id: backlog for backlog in backlogs}
    devices = {device["name"]: dict(device) for device in status["devices"]}
    reservations = {reservation["id"]: reservation for reservation in status["reservations"]}
    if len(devices) != len(status["devices"]) or len(reservations) != len(status["reservations"]):
        raise ValueError("status contains duplicate device or reservation IDs")
    for device in devices.values():
        _count("available_ram_mb", device["available_ram_mb"])
        _count("available_compute_units", device["available_compute_units"])

    expired = tuple(sorted(worker.reservation_id for worker in workers
                           if worker.reservation_id not in reservations))
    if expired:
        return WorkerPlan(expired=expired)

    grouped: dict[str, list[OwnedWorker]] = defaultdict(list)
    for worker in workers:
        _validate_claim(worker, by_id[worker.workload_id],
                        reservations[worker.reservation_id], devices)
        grouped[worker.workload_id].append(worker)

    drain: set[str] = set()
    placements: list[_Placement] = []
    for backlog in sorted(backlogs, key=lambda item: item.workload_id):
        owned = grouped[backlog.workload_id]
        accepting = sorted(
            (worker for worker in owned if worker.state == "accepting"),
            key=lambda worker: (worker.buffered_requests, worker.reservation_id),
        )
        starting = sum(worker.state == "starting" for worker in owned)
        rate, work = _cost(backlog)
        target = min(backlog.max_workers, math.ceil(backlog.queued_requests / work))
        surplus = max(0, len(accepting) - target)
        drain.update(worker.reservation_id for worker in accepting[:surplus])
        retained = accepting[surplus:]
        covered = len(retained) + starting
        needed = max(0, target - covered)
        cap_space = max(0, backlog.max_workers - len(owned))
        reasons: set[ShortfallReason] = set()
        if cap_space < needed:
            reasons.add("worker_limit")
        if starting:
            affordable = 0
            if needed:
                reasons.add("starting")
        else:
            queue_at_ready = max(0, backlog.queued_requests - sum(
                max(0, rate * backlog.startup_seconds - worker.buffered_requests)
                for worker in retained
            ))
            affordable = math.floor(queue_at_ready / work)
            if backlog.queued_requests and not retained:
                affordable = max(1, affordable)
            if affordable < needed:
                reasons.add("startup_cost")
        placements.append(_Placement(
            backlog, target, covered, min(needed, cap_space, affordable), reasons
        ))

    available = [devices[name] for name in sorted(devices)]
    solver = SpreadSolver()
    starts: list[WorkerStart] = []
    while True:
        candidates = sorted(
            (item for item in placements if not item.blocked and item.added < item.budget),
            key=lambda item: (item.covered + item.added, item.backlog.workload_id),
        )
        if not candidates:
            break
        for item in candidates:
            backlog = item.backlog
            try:
                request = solver.solve([(backlog.workload_id, [backlog.worker])], available)
            except ValueError:
                item.blocked = True
                item.reasons.add("capacity")
                continue
            slot = request.reservations[0].slots[0]
            starts.append(WorkerStart(backlog.workload_id, ExplicitSlot(
                slot.name, ram_mb=slot.ram_mb, compute_units=slot.compute_units
            )))
            devices[slot.name]["available_ram_mb"] -= slot.ram_mb
            devices[slot.name]["available_compute_units"] -= slot.compute_units
            item.added += 1

    shortfall = {}
    for item in placements:
        count = max(0, item.target - item.covered - item.added)
        if count:
            shortfall[item.backlog.workload_id] = WorkerShortfall(
                count, tuple(sorted(item.reasons))
            )
    return WorkerPlan(
        keep=tuple(sorted(worker.reservation_id for worker in workers
                          if worker.reservation_id not in drain)),
        drain=tuple(sorted(drain)),
        start=tuple(starts),
        shortfall=shortfall,
    )


def recommend_workers(
    backlogs: Sequence[Backlog],
    workers: Sequence[OwnedWorker],
    *,
    socket_path: str | None = None,
) -> WorkerPlan:
    """Read one daemon snapshot and return an advisory plan.

    Uses the SDK's usual socket selection and daemon-not-running errors.
    Acquisitions can still lose a race; keep surviving workers and replan.
    """
    return plan_workers(
        backlogs, workers, client._query_status(client._get_socket_path(socket_path))
    )
