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
- [NVIDIA CUDA](#nvidia-cuda) demonstrates reservations for one or two GPUs.
- [Apple Silicon Metal](#apple-silicon-metal) demonstrates a single reservation
  for an MPS workload.

## CPU without a GPU

Start with the [getting-started walkthrough](../GETTING_STARTED.md#try-a-reservation-without-a-gpu)
for environment setup, the two-terminal commands, and expected status output.
It uses `cpu.json` and `cpu_job.py` to claim 256 MB from a 1024 MB accounting
budget and show status while the reservation is held. It needs no PyTorch or GPU.

The same walkthrough shows how to [wrap an existing command](../GETTING_STARTED.md#wrap-a-command-that-knows-nothing-about-reserved)
and how to [adopt the lease in a Python worker](../GETTING_STARTED.md#adopt-the-lease-in-a-python-worker)
with `inherited_lease_job.py`.

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

## NVIDIA CUDA

Run these commands from the repository root on a Linux machine with NVIDIA GPUs.
Use an environment with ReserveD installed and a CUDA-enabled build of PyTorch
for the worker. See the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

Check which GPUs the machine exposes and how much memory they have:

```bash
nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id --format=csv
```

The example configs assume GPU `0` for one card and GPUs `0` and `1` for two
cards. Adjust the `cuda:N` names and `ram_mb` budgets to your machine. These
budgets are entries in the reservation ledger. Choose them based on the memory
you can commit to participating workloads. This example uses CUDA's PCI bus
ordering. Verify that the configured ordinals identify the intended cards.
See [NVIDIA's CUDA environment variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#device-enumeration-and-properties).

In the first terminal, start a demo daemon. Use [cuda.json](cuda.json) for one
card or substitute [cuda_dual.json](cuda_dual.json) for two cards. Stop the demo
daemon with Ctrl-C before switching configs.

```bash
export RESERVED_SOCKET="/tmp/reserved-cuda-demo-$(id -u).sock"
reserved --config examples/cuda.json
```

In the second terminal, use the same environment and socket:

```bash
export RESERVED_SOCKET="/tmp/reserved-cuda-demo-$(id -u).sock"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
reserve-status
```

### One card

Ask ReserveD for any configured CUDA GPU with at least 1024 MB of available
budget. The worker adopts the lease and runs a small tensor operation on the
granted GPU as logical `cuda:0`.

```bash
reserve cuda-demo type=cuda,ram=1024 -- python examples/cuda_job.py
```

### Two distinct cards

With `cuda_dual.json` loaded, name each physical card to claim 1024 MB on each
in one reservation. Both claims are granted together or the job does not start.
Two `type=cuda` slots can share a card under the default solver. Naming the cards
guarantees distinct grants.

```bash
reserve cuda-dual-demo name=cuda:0,ram=1024 name=cuda:1,ram=1024 -- python examples/cuda_job.py
```

The worker prints one physical-to-logical mapping per grant. For example, if
you request `cuda:1` before `cuda:0`, the first grant maps to PyTorch's `cuda:0`
and the second to `cuda:1`. The worker releases the reservation after the tensor
operations finish. Run `reserve-status` again to see the capacity return.

To check allocation without PyTorch, run either command with `reserve-status`
after `--` in place of `python examples/cuda_job.py`. It prints the reservation
and its slots while the lease is held.

## Apple Silicon Metal

Start a daemon with the example hardware config:

```bash
reserved --config examples/m2_mac.json
```

In another shell, run the example SDK client:

```bash
python3 examples/metal_job.py
```

While it is waiting, inspect the held reservation:

```bash
reserve-status
```
