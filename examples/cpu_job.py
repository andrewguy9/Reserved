"""Claim CPU capacity and show status while the lease is open.

Start the demo daemon and set RESERVED_SOCKET as described in
GETTING_STARTED.md, then run this script with the installed environment.
No GPU, PyTorch, or model download is needed.
"""

import subprocess
import sys

from reserved.sdk import DeviceConstraint, Reservation


with Reservation("cpu-example", DeviceConstraint("cpu", ram_mb=256)) as reservation:
    reservation.check()
    print(f"holding 256 MB on {reservation.devices[0].device_name}", flush=True)
    subprocess.run([sys.executable, "-m", "reserved.bin.reserve_status"], check=True)

print("lease closed; the daemon will return the capacity")
