# reserved examples

This directory contains small configs and scripts for trying `reserved` locally.

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
