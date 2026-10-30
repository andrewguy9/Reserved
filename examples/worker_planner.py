"""Dummy batched services using real reservations and the worker planner.

Start reserved with examples/worker_planner.json, then run this script against
the same socket. Loading, batching, and unloading are simulated at 50x speed.
No model runtime or GPU is required.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

from reserved.sdk import (
    Backlog, DeviceConstraint, OwnedWorker, Request, Reservation,
    ReservationError, ReservationExpiredError, recommend_workers,
)


TIME_SCALE = 0.02
WORK = Backlog(
    workload_id="dummy-model", queued_requests=83,
    worker=DeviceConstraint("cpu", ram_mb=256, compute_units=25),
    max_workers=4, batch_size=4, batch_seconds=2,
    startup_seconds=3, shutdown_seconds=1,
)


@dataclass
class DummyWorker:
    reservation: Reservation
    state: str = "starting"
    buffered_requests: int = 0

    @property
    def reservation_id(self) -> str:
        assert self.reservation.reservation_id is not None
        return self.reservation.reservation_id

    def report(self) -> OwnedWorker:
        return OwnedWorker(self.reservation_id, WORK.workload_id,
                           self.state, self.buffered_requests)


async def main() -> None:
    queued = WORK.queued_requests
    completed = 0
    workers: dict[str, DummyWorker] = {}
    tasks: list[asyncio.Task] = []

    async def serve(worker: DummyWorker) -> None:
        nonlocal queued, completed
        try:
            await asyncio.sleep(WORK.startup_seconds * TIME_SCALE)
            worker.state = "accepting"
            print(f"ready {worker.reservation_id}")
            while True:
                worker.reservation.check()
                if worker.state == "accepting":
                    # Prefetch up to two batches. Assigned work leaves the shared
                    # queue, but remains buffered until processing completes.
                    take = min(queued, 2 * WORK.batch_size - worker.buffered_requests)
                    queued -= take
                    worker.buffered_requests += take
                if worker.buffered_requests:
                    batch = min(WORK.batch_size, worker.buffered_requests)
                    if batch < WORK.batch_size:
                        print(f"flush {worker.reservation_id}: partial batch of {batch}")
                    await asyncio.sleep(WORK.batch_seconds * TIME_SCALE)
                    worker.buffered_requests -= batch
                    completed += batch
                elif worker.state == "draining":
                    break
                else:
                    await asyncio.sleep(TIME_SCALE / 2)
        finally:
            # A failed/cancelled dummy service returns its unfinished work.
            queued += worker.buffered_requests
            worker.buffered_requests = 0
            worker.state = "stopping"
            try:
                await asyncio.sleep(WORK.shutdown_seconds * TIME_SCALE)
            finally:
                worker.reservation.release()
                workers.pop(worker.reservation_id, None)
                print(f"released {worker.reservation_id}")

    try:
        while queued or workers:
            for task in tasks:
                if task.done() and task.exception() is not None:
                    raise task.exception()
            plan = recommend_workers(
                [replace(WORK, queued_requests=queued)],
                [worker.report() for worker in workers.values()],
            )
            if plan.expired:
                raise ReservationExpiredError(f"expired reservations: {plan.expired}")
            for reservation_id in plan.drain:
                worker = workers[reservation_id]
                worker.state = "draining"
                print(f"drain {reservation_id}: queued={queued}, "
                      f"buffered={worker.buffered_requests}")
            if plan.start:
                try:
                    leases = Request(*[
                        Reservation(start.workload_id, start.slot)
                        for start in plan.start
                    ]).acquire()
                except ReservationError:
                    print("capacity changed; polling again")
                else:
                    for lease in leases:
                        worker = DummyWorker(lease)
                        workers[worker.reservation_id] = worker
                        tasks.append(asyncio.create_task(serve(worker)))
                        print(f"start {worker.reservation_id}")
            await asyncio.sleep(TIME_SCALE / 2)
        await asyncio.gather(*tasks)
        print(f"completed {completed} requests; all worker leases released")
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
