"""
Example: acquire an Apple Silicon Metal/MPS reservation from Python.

Start a daemon with:

    reserved --config examples/m2_mac.json

Then run:

    python3 examples/metal_job.py
"""

from reserved.sdk.client import DeviceConstraint, Reservation


with Reservation("metal-example", DeviceConstraint("metal", ram_mb=10240, compute_units=100)) as r:
    print(f"got {r.devices[0].device_name}, reservation {r.reservation_id}")
    input("press enter to release...")
