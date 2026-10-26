# reserved examples

This directory contains small configs and scripts for trying `reserved` locally.

## CPU without a GPU

Start with the [getting-started walkthrough](../GETTING_STARTED.md#try-a-reservation-without-a-gpu)
for environment setup, the two-terminal commands, and expected status output.
It uses `cpu.json` and `cpu_job.py` to claim 256 MB from a 1024 MB accounting
budget and show status while the reservation is held. It needs no PyTorch or GPU.

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
