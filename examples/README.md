# reserved examples

This directory contains small configs and scripts for trying `reserved` locally.

## CPU without a GPU

Start with the [getting-started walkthrough](../GETTING_STARTED.md#try-a-reservation-without-a-gpu)
for environment setup, the two-terminal commands, and expected status output.
It uses `cpu.json` and `cpu_job.py` to claim 256 MB from a 1024 MB accounting
budget and show status while the reservation is held. It needs no PyTorch or GPU.

The same walkthrough shows how to [wrap an existing command](../GETTING_STARTED.md#wrap-a-command-that-knows-nothing-about-reserved)
and how to [adopt the lease in a Python worker](../GETTING_STARTED.md#adopt-the-lease-in-a-python-worker)
with `inherited_lease_job.py`. `launch_worker.py` starts that same worker from
Python with `Reservation.popen()` instead of the `reserve` command.

## NVIDIA CUDA

Run these commands from the repository root on a Linux machine with NVIDIA GPUs.
Use an environment with ReserveD installed and a CUDA-enabled build of PyTorch
for the worker. See the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

Check which GPUs the machine exposes and their UUIDs:

```bash
nvidia-smi -L
```

The example configs use placeholder UUIDs. Generate a config for the cards
actually in the machine instead of editing them by hand, from the same
environment that has PyTorch, with no `CUDA_VISIBLE_DEVICES` set so every card
is listed:

```bash
reserved-generate-config > /tmp/reserved-cuda.json
```

`reserved-generate-config` detects NVIDIA GPUs through torch and reads each card's GPU UUID, total memory, and
streaming multiprocessor count from torch's device properties. It names the
devices `cuda:GPU-<uuid>`, sets `ram_mb` to the card's memory in MiB, and sets
`compute_units` to its multiprocessor count (84 on an RTX A6000). A UUID names
one specific card, so no device-ordering setting is needed. Pass
`--ram-fraction 0.9` to commit only part of each card's memory; budgets are
entries in the reservation ledger and should reflect the memory you can commit
to participating workloads. Use the generated names in the `reserve` commands
below.

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
Two `type=cuda` constraints can share a card under the default solver. Naming the cards
guarantees distinct grants.

```bash
reserve cuda-dual-demo name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456780,ram=1024 name=cuda:GPU-0a1b2c3d-4e5f-6789-abcd-ef0123456781,ram=1024 -- python examples/cuda_job.py
```

The worker prints one physical-to-logical mapping per grant. For example, if
you name the second card before the first, the first grant maps to PyTorch's
`cuda:0` and the second to `cuda:1`. The worker releases the reservation after the tensor
operations finish. Run `reserve-status` again to see the capacity return.

To check allocation without PyTorch, run either command with `reserve-status`
after `--` in place of `python examples/cuda_job.py`. It prints the reservation
and its allocations while the lease is held.

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
