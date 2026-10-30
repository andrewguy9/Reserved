# reserved examples

This directory contains small configs and scripts for trying `reserved` locally.

## Getting started

Run commands from the repository root. Install this checkout to use its SDK and
worker planner:

```bash
python3 -m pip install -e .
```

- [CPU without a GPU](#cpu-without-a-gpu) demonstrates acquiring a single lease
  and inspecting capacity while it is held.
- [Batched worker planning](#batched-worker-planning) demonstrates an application
  sizing its worker pool, buffering batches, and gracefully draining services.
- [Apple Silicon Metal](#apple-silicon-metal) demonstrates a single reservation
  for an MPS workload.

## CPU without a GPU

Start with the [getting-started walkthrough](../GETTING_STARTED.md#try-a-reservation-without-a-gpu)
for environment setup, the two-terminal commands, and expected status output.
It uses `cpu.json` and `cpu_job.py` to claim 256 MB from a 1024 MB accounting
budget and show status while the reservation is held. It needs no PyTorch or GPU.

## Batched worker planning

Use this example to integrate `recommend_workers()` into an application that
starts and manages its own workers. It declares a backlog, per-worker resource
requirements, rough batch and startup timings, and existing worker states. The
planner recommends which workers to keep, start, or drain on each poll.

The [dummy worker script](worker_planner.py) uses the [CPU config](worker_planner.json),
so no model runtime or GPU is needed.

**1. Start the daemon** in one terminal:

```bash
reserved --config examples/worker_planner.json --socket /tmp/planner.sock
```

**2. Run the worker pool** in another terminal:

```bash
RESERVED_SOCKET=/tmp/planner.sock python3 examples/worker_planner.py
```

The example uses real reservation leases and asyncio tasks as dummy services.
It simulates model loading, prefetches two batches per worker, and processes
83 requests. As the shared queue empties, workers stop accepting new work,
finish their buffers (including partial batches), unload, and release their
leases independently. Timing runs at 50 times the declared speed.

The planner only recommends changes. The example application owns worker
startup, dispatch, lifecycle state, and lease cleanup. See
[the project definition](../WORKER_PLANNER_PLAN.md) for the sizing heuristic.
The [SDK guide](../README.md#planning-a-pool-of-batched-workers) explains the
input and output types and how to acquire recommended reservations.

## Apple Silicon Metal

Start a daemon with the example hardware config:

```bash
reserved --config examples/m2_mac.json --socket /tmp/reserved.sock
```

In another shell, run the example SDK client:

```bash
RESERVED_SOCKET=/tmp/reserved.sock python3 examples/metal_job.py
```

While it is waiting, inspect the held reservation:

```bash
RESERVED_SOCKET=/tmp/reserved.sock reserve-status
```
