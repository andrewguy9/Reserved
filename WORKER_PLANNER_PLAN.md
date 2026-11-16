# Worker planner reference

The SDK recommends which workers to keep, start, or drain. An application owns
its queue, processes, and leases. The planner uses the same placement code as
`SpreadSolver`; `Request.acquire()` grants or rejects the explicit assignments
atomically against the live ledger.

## Inputs and results

Call `recommend_workers(backlogs, workers=(), socket_path=...)` to read one
daemon snapshot, or `plan_workers(backlogs, workers, status)` with an existing
parsed snapshot. Declare a `Backlog` for every managed workload, even after its
queue reaches zero. `workload_id` identifies a model and serving configuration;
`DeviceConstraint.model` filters hardware. Each worker claims one device slot
with a positive RAM budget covering loading and peak batch memory. Compute
units are the operator's accounting budget.

Report each held reservation as an `OwnedWorker` with its workload ID, state
(`starting`, `accepting`, `draining`, or `stopping`), and unfinished buffered
request count. A request belongs either to the shared queue or to one worker's
buffer, including in-flight batches. Unlisted reservations remain fixed.
Use a new workload ID if its resource requirements change.

| Result | Meaning |
| --- | --- |
| `keep` | IDs of workers continuing in their current state. |
| `drain` | IDs of accepting workers that should stop receiving new work. |
| `start` | Workload IDs and `ExplicitSlot` placements ready for `Reservation`. |
| `shortfall` | Unmet worker targets, with counts and reasons (`capacity`, `startup_cost`, `starting`, `worker_limit`). |
| `expired` | Supplied IDs missing from the snapshot. Other results are empty until the application reconciles. |

## How the planner sizes workers

With default timing values, it seeks up to one worker per unassigned request,
bounded by the worker cap and free capacity. For batched services, set
`batch_size`, `batch_seconds`, `startup_seconds`, and `shutdown_seconds`:

```text
rate = batch_size / batch_seconds
work_per_worker = rate * max(batch_seconds, startup_seconds + shutdown_seconds)
target = min(max_workers, ceil(queued_requests / work_per_worker))
```

Surplus accepting workers drain in order of smallest buffer, then reservation
ID. Loading workers count toward expansion but don't replace ready workers
until accepting. Draining and stopping workers count against the cap and keep
their capacity until release.

For expansion, estimate how much of the shared queue the retained workers can
process during startup, after their own buffers. Each new worker needs at least
`work_per_worker` requests left by then. A nonempty queue with no accepting or
loading worker can bootstrap one, subject to cap and capacity. Further expansion
waits while workers are loading.

Eligible workloads take turns in order of fewest accepting and starting workers,
including planned starts, with ties broken by workload ID. Each requested slot
uses the shared spread placement calculation and consumes its snapshot capacity
once. This is a greedy placement rule; it doesn't rearrange assignments or
optimize request ordering.

## Apply recommendations

Use one application controller to apply the plan. Register every acquired lease
as `starting` before launching workers. If acquisition fails because capacity
changed, retain surviving workers and get a fresh plan. For a drain, stop
dispatch and prefetch, finish the buffer (including any partial batch), unload,
and then close the lease. Keep reporting the worker until release; hardware
becomes available once the daemon observes the closure.

See the [SDK snippet](README.md#planning-a-pool-of-batched-workers) and the
[runnable CPU example](examples/README.md#batched-worker-planning).
