"""Acquire a CPU reservation and start inherited_lease_job.py under it."""

import sys
from pathlib import Path

from reserved import sdk


if __name__ == "__main__":
    worker_script = Path(__file__).with_name("inherited_lease_job.py")
    with sdk.Reservation("launcher", sdk.DeviceConstraint(name="cpu", ram_mb=256)) as reservation:
        print(f"holding {reservation.reservation_id}; starting the worker")
        worker = reservation.popen([sys.executable, str(worker_script)])
        returncode = worker.wait()
        print(f"worker exited with {returncode}; releasing the lease")
