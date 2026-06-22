# reserved

A single-machine, best-effort resource reservation daemon for physical hardware (GPUs, CPU, RAM). Processes claim hardware capacity before running, and release it automatically when they exit — even on crash.

`reserved` is opt-in coordination between willing processes. It does not enforce anything against programs that don't participate. It succeeds or fails immediately — there is no queue.

---

## Install

```bash
pip install reserved
```

Requires Python 3.11+.

---

## Quick start

**1. Create a hardware config** describing the devices on your machine:

```json
{
  "devices": [
    { "name": "cuda:0", "type": "cuda", "model": "rtx3090", "ram_mb": 24576, "compute_units": 100 },
    { "name": "cuda:1", "type": "cuda", "model": "rtx3090", "ram_mb": 24576, "compute_units": 100 },
    { "name": "cpu",    "type": "cpu",                       "ram_mb": 65536 }
  ]
}
```

Save it to `~/.config/reserved/reserved.json` (or pass `--config PATH`).

**2. Start the daemon:**

```bash
reserved
```

On macOS, where `/var/run` requires root:

```bash
reserved --socket /tmp/reserved.sock
export RESERVED_SOCKET=/tmp/reserved.sock
```

**3. Check status:**

```bash
reserve-status
```

---

## `reserve` CLI

Wraps any command under a reservation. The reservation is held for the lifetime of the command.

```
reserve LABEL [SLOT ...] [--solver consolidate|spread] [--socket PATH] -- CMD [ARGS...]
```

Each `SLOT` is a comma-separated set of constraints:

| Field | Effect |
| --- | --- |
| `type=TYPE` | Device type: `cuda`, `metal`, or `cpu` |
| `model=MODEL` | Device model label (e.g. `rtx3090`) |
| `ram=MB` | Minimum available RAM in MB |
| `compute=N` | Minimum available compute units |
| `name=NAME` | Pin to a specific device by name |

```bash
# Any CUDA device
reserve infer type=cuda -- python infer.py

# Apple Silicon (Metal/MPS)
reserve infer type=metal -- python infer.py

# Needs 30GB VRAM
reserve train type=cuda,ram=30720 -- python train.py

# Specific model
reserve infer type=cuda,model=rtx3090 -- python infer.py

# Pin to a specific device
reserve infer name=cuda:2 -- python infer.py

# Two CUDA devices (distributed job)
reserve distributed type=cuda type=cuda -- python distributed.py

# Spread across cards for throughput
reserve pipeline --solver spread type=cuda type=cuda -- python pipeline.py
```

**Environment variables set before exec:**

| Variable | Value |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | Physical CUDA ordinals of granted devices (e.g. `2` or `2,3`) |
| `RESERVED_DEVICE` | Physical name of the first granted accelerator |
| `RESERVED_DEVICES` | Comma-separated physical names of all granted devices |
| `RESERVED_ID` | Reservation ID |
| `RESERVED_FD` | File descriptor number of the lease fd |

---

## Python SDK

```python
from reserved.sdk.client import Reservation, Request, DeviceConstraint, ExplicitSlot
from reserved.sdk.errors import ReservationError, DaemonNotRunningError, ReservationExpiredError
```

### Single reservation

```python
# Any CUDA device
with Reservation("infer", DeviceConstraint("cuda")) as r:
    model.to(r.devices[0].torch_device())
    run_model()

# Metal (Apple Silicon)
with Reservation("infer", DeviceConstraint("metal")) as r:
    model.to(r.devices[0].torch_device())   # torch.device("mps")
    run_model()

# Needs 30GB VRAM
with Reservation("infer", DeviceConstraint("cuda", ram_mb=30720)) as r:
    model.to(r.devices[0].torch_device())
    ...

# Two CUDA devices, released together
with Reservation("distributed", DeviceConstraint("cuda"), DeviceConstraint("cuda")) as r:
    devices = r.torch_devices()
    run_distributed(devices)
```

### Multiple reservations — atomic acquire, independent release

```python
# Stable diffusion: checkpoint held for the session, LoRA freed after each inference
req = Request(
    Reservation("checkpoint", DeviceConstraint("cuda", ram_mb=8192)),
    Reservation("lora",       DeviceConstraint("cuda", ram_mb=2048)),
)
checkpoint, lora = req.acquire()

with checkpoint:
    load_checkpoint(checkpoint.devices[0].torch_device())
    with lora:
        load_lora(lora.devices[0].torch_device())
        run_inference()
    # lora released here, VRAM returned to pool
# checkpoint released here

# TTS pipeline: 5 actors acquired atomically, freed one by one as each finishes
actors = Request(*[
    Reservation(f"actor-{i}", DeviceConstraint("cuda", ram_mb=4096))
    for i in range(5)
]).acquire()

for actor, line in zip(actors, script):
    device = actor.devices[0].torch_device()
    render(device, line)
    actor.release()
```

### Explicit placement (orchestrator)

```python
# Pin directly to named devices — bypasses solver
with Reservation("infer", ExplicitSlot("cuda:2", ram_mb=8192)) as r:
    model.to(r.devices[0].torch_device())
    run_model()
```

### Checking for daemon restart

Call `check()` at natural checkpoints. Raises `ReservationExpiredError` if the daemon has restarted and the lease is gone.

```python
with Reservation("train", DeviceConstraint("cuda", ram_mb=16384)) as r:
    for epoch in range(num_epochs):
        r.check()
        train_epoch(model, data)
```

### Errors

- `ReservationError` — raised by `acquire()` when the request is rejected (insufficient resources).
- `DaemonNotRunningError` — raised when the daemon socket is not found.
- `ReservationExpiredError` — raised by `check()` when the daemon has restarted.

---

## `reserve-status`

```bash
reserve-status [--socket PATH] [--json]
```

```
Devices:
  cuda:0  cuda  rtx3090  available: 16384/24576 MB  100/100 compute
  cuda:1  cuda  rtx3090  available: 24576/24576 MB  100/100 compute

Reservations:
  a3f9c2  label=train  age=2m14s
    cuda:0  8192 MB
```

Use `--json` for scripting:

```bash
reserve-status --json | jq '.reservations | length'
```

---

## Resource config

Devices have the following fields:

| Field | Description |
| --- | --- |
| `name` | Unique identifier. CUDA: must be `cuda:N` matching `nvidia-smi` index. Metal: use `metal:0`. |
| `type` | `cuda`, `metal`, or `cpu` |
| `model` | Optional label for the hardware class |
| `ram_mb` | Total reservable RAM (VRAM for CUDA, unified memory for Metal, system RAM for CPU) |
| `compute_units` | Optional opaque budget — convention between operator and callers |

**Apple Silicon (macOS) config:**

```json
{
  "devices": [
    { "name": "metal:0", "type": "metal", "model": "m3-max", "ram_mb": 36864, "compute_units": 100 },
    { "name": "cpu",     "type": "cpu",                       "ram_mb": 65536 }
  ]
}
```

Config is discovered in this order:
1. `--config PATH` flag
2. `RESERVED_CONFIG` env var
3. `~/.config/reserved/reserved.json`
4. `/etc/reserved/reserved.json`

---

## Deployment

### macOS (launchd)

```bash
# 1. Create config
mkdir -p ~/.config/reserved
# edit ~/.config/reserved/reserved.json

# 2. Install launchd agent
cat > ~/Library/LaunchAgents/com.reserved.daemon.plist <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.reserved.daemon</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/local/bin/reserved</string>
    <string>--socket</string><string>/tmp/reserved.sock</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/reserved.log</string>
  <key>StandardErrorPath</key><string>/tmp/reserved.log</string>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.reserved.daemon.plist

# 3. Add to shell profile
echo 'export RESERVED_SOCKET=/tmp/reserved.sock' >> ~/.zshrc
```

### Linux — system daemon (systemd, root)

```bash
# 1. Install
pip install reserved
sudo mkdir -p /etc/reserved
# edit /etc/reserved/reserved.json

# 2. Create service account and group
sudo groupadd gpu-users
sudo useradd --system --no-create-home --group reserved reserved

# 3. Install unit files
sudo tee /etc/systemd/system/reserved.socket > /dev/null <<'EOF'
[Unit]
Description=reserved resource reservation daemon socket
[Socket]
ListenStream=/var/run/reserved.sock
SocketGroup=gpu-users
SocketMode=0660
[Install]
WantedBy=sockets.target
EOF

sudo tee /etc/systemd/system/reserved.service > /dev/null <<'EOF'
[Unit]
Description=reserved resource reservation daemon
Requires=reserved.socket
After=reserved.socket
[Service]
ExecStart=/usr/bin/reserved --config /etc/reserved/reserved.json
User=reserved
Group=reserved
Restart=on-failure
[Install]
WantedBy=multi-user.target
EOF

# 4. Enable
sudo systemctl daemon-reload
sudo systemctl enable --now reserved.socket

# 5. Add users
sudo usermod -aG gpu-users $USER
```

### Linux — user daemon (systemd, no root)

```bash
# 1. Install and create config
pip install reserved
mkdir -p ~/.config/reserved
# edit ~/.config/reserved/reserved.json

# 2. Install unit files
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/reserved.socket <<'EOF'
[Unit]
Description=reserved resource reservation daemon socket
[Socket]
ListenStream=%t/reserved.sock
[Install]
WantedBy=sockets.target
EOF

cat > ~/.config/systemd/user/reserved.service <<'EOF'
[Unit]
Description=reserved resource reservation daemon
Requires=reserved.socket
After=reserved.socket
[Service]
ExecStart=%h/.local/bin/reserved
Restart=on-failure
[Install]
WantedBy=default.target
EOF

# 3. Enable
systemctl --user daemon-reload
systemctl --user enable --now reserved.socket
echo 'export RESERVED_SOCKET="${XDG_RUNTIME_DIR}/reserved.sock"' >> ~/.bashrc
```

---

## What reserved does NOT do

- Queue requests — callers retry if they want to wait
- Persist state — restart means a clean slate
- Enforce against non-participating processes
- Multi-machine coordination
