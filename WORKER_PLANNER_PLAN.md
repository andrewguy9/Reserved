# Reserved Worker Planner Project Definition

## Goal

Let an application declare its unfinished work and worker resource requirements,
then ask the SDK how many workers to keep, drain, and start on the current machine.
The application acquires reservations and runs the workers itself. Workers retire
by ceasing to accept new work, draining their assigned batches, and unloading
before releasing their reservations.

The first version combines capacity-based sizing with a small estimate of useful
work per worker. It preserves existing workers where possible, phases workers
out as the shared queue empties, and avoids starting services for a short tail.
It uses a conservative heuristic; maximum-throughput and completion-time
optimization are later work.

The initial SDK implementation is in `reserved.sdk.planner`. The sections below
define its scope and lifecycle contract.

## First version scope

- One machine, using the existing daemon and reservation protocol.
- Multiple independent workloads, each with a count of requests awaiting dispatch.
- One independently releasable reservation and one device slot per worker.
- Workers buffer and process batches. Requests within a workload are
  interchangeable for estimating work.
- Each workload has one fixed resource requirement shared by all its workers.
- Rough batch, startup, and shutdown estimates shared by a workload's workers.
- The SDK recommends changes. The application owns request dispatch, model
  loading, processes, polling, and reservation lifetimes.

A workload ID is an application-defined identifier for a model and its serving
configuration. The planner treats it as opaque. In the existing
`DeviceConstraint`, `model` continues to mean the hardware model, such as
`rtx3090`, rather than the inference model.

## Application inputs

Each planning call supplies the complete set of workloads being managed by this
controller, including workloads whose backlog has reached zero.

| Backlog field | Meaning |
| --- | --- |
| `workload_id` | Unique application-defined workload identifier. |
| `queued_requests` | Nonnegative count of requests still in the shared queue, excluding work assigned to any worker. |
| `worker` | A single `DeviceConstraint` describing the resources required by each worker. |
| `max_workers` | Nonnegative application limit on workers for this workload. |
| `batch_size` | Typical number of requests processed in one batch. Positive integer. |
| `batch_seconds` | Rough processing time for a typical batch. Positive number. |
| `startup_seconds` | Rough time from acquiring a reservation until a new worker can process work. Nonnegative number. |
| `shutdown_seconds` | Rough unload and teardown time after assigned work has drained. Nonnegative number. |

These estimates can be naive application-supplied constants. They need not be
benchmarks. V1 assumes the same performance for the compatible devices selected
by a workload's constraint. Applications with materially different configurations
can declare separate workloads.

For v1, the constraint must specify a device type, positive `ram_mb`, and
nonnegative `compute_units`. The RAM declaration must cover the worker's peak
use, including model loading and buffered batches. A hardware model filter is
optional. Compute units retain their existing meaning as an operator-defined
budget; they are not a
throughput measurement. The worker cap lets the application limit concurrency
when its compute requirement is only approximate.

The application also describes the reservations it already holds for these
workers:

| Owned worker field | Meaning |
| --- | --- |
| `reservation_id` | ID of the worker's current reservation. |
| `workload_id` | Workload currently served by that worker. |
| `state` | One of `starting`, `accepting`, `draining`, or `stopping`. |
| `buffered_requests` | Nonnegative count of assigned but unfinished requests, including prefetched batches and requests currently executing. |

`accepting` means the worker is ready to receive additional work; it may already
be busy. `draining` means new dispatch has ceased and assigned work is finishing.
`stopping` means assigned work is complete and unloading or teardown is underway.
All states retain their reservations. Each worker's lease covers that worker
alone. Changing a workload's resource requirements requires a new workload ID.

Total unfinished work is `queued_requests + sum(buffered_requests)`. Work must
appear in exactly one place. Batch buffering should not make the shared queue's
depletion look like completed work.

The SDK queries daemon status to resolve the supplied reservation IDs to their
device assignments and claimed resources. Reservations absent from the supplied
owned-worker list remain fixed, including any other reservations held by the same
application. An ID identifies a reservation for planning; the application still
needs its existing lease handle to release it.

Duplicate workload or reservation IDs, unknown workload references, invalid
counts or states, nonfinite timing estimates, and unsupported worker requirements
are input errors. A stopping worker must have no unfinished buffered requests.

## Recommendation API

`reserved.sdk.planner` provides a pure planning function and a thin status-query
wrapper. Both are also exported from `reserved.sdk`:

```python
def plan_workers(backlogs, workers, status) -> WorkerPlan: ...
def recommend_workers(backlogs, workers, *, socket_path=None) -> WorkerPlan: ...
```

`plan_workers` accepts the parsed daemon status snapshot. It performs no IO and
changes no reservations. `recommend_workers` obtains one snapshot from the
daemon, then calls the same function.

The plan contains:

| Plan field | Meaning |
| --- | --- |
| `keep` | Owned reservation IDs that continue in their current lifecycle state. |
| `drain` | Accepting workers that should cease taking new work, finish their buffers, then unload and release. |
| `start` | `WorkerStart` specifications with `workload_id` and `slot`, where `slot` is an `ExplicitSlot` ready for constructing an SDK reservation. |
| `shortfall` | Map of workload ID to `WorkerShortfall(count, reasons)`, omitting covered targets. Reasons are `capacity`, `startup_cost`, `starting`, or `worker_limit`; more than one may apply. |
| `expired` | Supplied reservation IDs missing from the daemon snapshot. |

If any supplied reservation is missing, return its ID in `expired` and leave
`keep`, `drain`, `start`, and `shortfall` empty. The application reconciles the
affected workers before requesting another recommendation. This includes handling
a daemon restart, which invalidates existing leases.

The plan is advisory and holds no resources. Repeated calls with the same inputs
and snapshot produce the same result.

## Queue sizing and graceful retirement

Use a coarse amount of work per accepting worker rather than equating one
request to one worker:

```text
requests_per_second = batch_size / batch_seconds
work_window_seconds = max(batch_seconds, startup_seconds + shutdown_seconds)
work_per_worker = requests_per_second * work_window_seconds
target_accepting = min(max_workers, ceil(queued_requests / work_per_worker))
```

With a positive queue and a positive cap, the target is at least one accepting
worker. An empty shared queue has a target of zero, even while assigned batches
are still running. The work window is a fixed conservative rule based on the
declared overheads, rather than a new optimization parameter.

Keep starting workers while startup finishes, and count them when considering
further starts. They do not replace ready capacity until they become accepting.
Among accepting workers, retain up to the target and recommend draining the
surplus, choosing workers with the smallest buffers first and breaking ties by
reservation ID. This preserves busier workers to serve the shared queue while
smaller buffers finish and free their reservations sooner.

Draining is a persistent lifecycle transition, not an immediate stop command.
Later polls continue to report those workers as draining or stopping. They are
never counted as available to serve the shared queue, never credited as free
hardware, and never switched back to accepting by the planner. They finish their
assigned work even if the target falls again. The application must flush partial
final batches rather than wait indefinitely for a normal batch to fill.

This heuristic trades some tail parallelism for earlier resource release. It
does not promise the fastest possible final request. The application remains
responsible for the order in which requests enter workers' buffers.

## Starting and placement rules

Before adding workers, estimate how much shared-queue work will remain after
startup. For each accepting worker retained by the plan, allow time for its own
buffer before estimating how much of the shared queue it can consume:

```text
queue_at_ready = max(0, queued_requests - sum(
    max(0, requests_per_second * startup_seconds - worker.buffered_requests)
    for worker in retained_accepting_workers
))
```

Allow at most `floor(queue_at_ready / work_per_worker)` new workers across the
whole plan for that workload, further limited by the accepting target, worker
cap, and hardware capacity. If there is shared-queue work, a positive cap, and no
accepting or starting worker, allow one bootstrap worker even for a small queue.
While a workload already has workers starting, wait for them to become ready
before recommending more starts for that workload. These rules avoid repeated
expansion during model loading and starting services for work the existing pool
is likely to finish soon.

Use the following placement rules:

1. Count all owned workers against `max_workers`, including starting, draining,
   and stopping workers. A lowered cap is reached gradually through draining.
2. Place additional workers against the snapshot's currently free capacity.
   Existing reservations already consume capacity in that snapshot; do not
   subtract them again or credit reservations listed in `drain`.
3. Consider eligible workloads with fewer accepting and starting workers first,
   including newly planned starts and breaking ties by workload ID. Try each
   once per pass, recomputing the order between passes. This fixed sharing rule
   needs no request priorities.
4. For each new worker, use the existing spread placement policy and subtract
   its claimed RAM and compute from a simulated copy of availability. Sort
   devices by name before placement to make ties stable.
5. Repeat until eligible additions are placed or a complete pass places no
   workers. Record unmet targets and their reasons in `shortfall`.

Existing workers keep their device assignments. V1 retires workers because a
workload is winding down or its cap was lowered; model replacement to favor
another workload is later work. Greedy placement may leave a capacity shortfall
that a different arrangement could avoid.

## Application integration

The following input uses the SDK API:

```python
from reserved.sdk import DeviceConstraint
from reserved.sdk.planner import Backlog, recommend_workers

plan = recommend_workers(
    backlogs=[
        Backlog(
            workload_id="model-a-fp16",
            queued_requests=1000,
            worker=DeviceConstraint("cuda", ram_mb=8192, compute_units=50),
            max_workers=8,
            batch_size=16,
            batch_seconds=2,
            startup_seconds=30,
            shutdown_seconds=5,
        ),
    ],
    workers=[],
)
```

The application follows this control loop:

1. Collect backlog counts and the state of every worker reservation it manages.
2. Request a recommendation. Resolve expired leases before applying any changes.
3. For `drain`, stop further dispatch and prefetch for that worker and mark it
   draining. Allow already assigned batches to finish, including partial final
   batches. Report its state on later polls while other workers keep serving.
   After its buffer empties, mark it stopping, unload its model, finish teardown,
   and release its lease. Never release merely because the plan requested drain.
4. Construct `Reservation(start.workload_id, start.slot)` for each `start` entry.
   Acquire these together through the existing `Request(...).acquire()` API.
5. Register each acquired reservation before starting its worker. If startup
   fails, clean up the worker and release the reservation once its resources are
   no longer in use. Keep lease handles alive for all running workers.
6. Repeat after applying the plan, on backlog changes, or at an application-chosen
   polling interval.

Only one controller applies recommendations to a given set of workers. Starting
and draining can span many polls; the controller records each lifecycle change
before asking for another plan. Initiating a drain does not block further polls
or starts using independent free capacity. Acquired workers must appear
immediately, for example as
`OwnedWorker(reservation_id=..., workload_id=..., state="starting",
buffered_requests=0)`, so they are counted rather than duplicated.

Acquisition can fail if another application claims capacity after the snapshot.
The application keeps its surviving workers and requests a fresh recommendation;
there is no queue or automatic retry loop in the planner.

Capacity from retiring workers becomes available to planning only after the
daemon observes the released leases. The application polls again to use that
capacity. The first release therefore needs no atomic reservation replacement.

## Example behavior

Assume two CUDA devices, each with 24,576 MB of reservable RAM and 100 compute
units. A worker requires 8,192 MB and 50 compute units. Its typical batch is 16
requests in two seconds, startup is 30 seconds, and teardown is five seconds.
The workload initially has 10,000 queued requests and a cap of eight workers.
The resulting work window is 35 seconds, or 280 requests per accepting worker.

| Situation | Recommendation |
| --- | --- |
| Both devices are free and the queue is large | Start four workers, two per device; capacity limits further starts. |
| Those four workers are loading | Keep their reservations and start none while they become ready. |
| Four workers are accepting; 600 requests remain in the shared queue, plus their own buffers | Target three accepting workers. Ask the worker with the smallest buffer to drain; it keeps its reservation while finishing. |
| The shared queue reaches 300, then 100 | Target two accepting workers, then one. Retire additional workers gradually, without interrupting buffers already draining. |
| The shared queue is empty but batches remain in every worker | Drain all remaining accepting workers. Buffered requests continue processing and reservations are released individually after teardown. |
| Only two requests remain unassigned with a warm worker available | Let the warm worker finish the tail; avoid starting another service for those requests. |
| Model A has no queued work but still has draining workers, while B needs their hardware | Let A finish and unload; start B only on a later call after the daemon reports free capacity. |

The four-worker recommendation follows declared capacity. Its throughput depends
on the application's worker requirements and behavior when sharing each device.

## Delivery and acceptance

Deliver the planner module, public SDK exports, a runnable example using dummy
workers, and SDK documentation. Keep daemon and wire-protocol behavior unchanged.

Acceptance checks should establish that:

- A large workload scales from zero workers to the capacity and worker limits.
- Multiple workloads receive deterministic recommendations under the fixed
  sharing rule, and placement continues when one workload cannot fit.
- Repeated polling counts existing workers correctly and does not duplicate them.
- Simulated starts respect combined RAM and compute limits and hardware filters.
- Shared-queue and assigned work are counted separately, including buffered
  batches and in-flight work.
- The accepting pool tapers as the queue shrinks; draining workers finish their
  buffers and partial final batches before unload and release.
- Starting workers prevent duplicate expansion; small tails avoid extra startups.
- Draining persists across polls and is not reversed by the next recommendation.
- Other reservations remain fixed and draining workers do not supply immediate
  placement capacity.
- Missing IDs require reconciliation, and malformed declarations fail clearly.
- A narrow integration test covers status lookup, acquiring recommended starts,
  and replanning after acquisition or a rejected request.

The example should simulate model loading, buffered batches, gradual draining,
and unloading, returning each worker's resources only when its lifecycle ends.
Most allocation tests belong at the pure planning-function layer.

## Deferred capabilities

Measured hardware-specific service profiles, batch-size optimization, elapsed
startup prediction, worker migration, replacing useful workers, multi-device
workers, shared model-weight reservations, and optimized completion times can
follow after applications use the basic loop. Request ordering continues to
belong to the application.
