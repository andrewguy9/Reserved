"""Worker launched by Reservation.popen_python() in tests; reports its interpreter state."""

import json
import os
import sys

import reserved
from reserved import sdk
from reserved.sdk.client import _query_status


def main() -> None:
    try:
        import shadowed_module  # noqa: F401
        shadow_imported = True
    except ModuleNotFoundError:
        shadow_imported = False
    with sdk.adopt_inherited_lease() as lease:
        lease.check()
        active = [r["id"] for r in _query_status(os.environ["RESERVED_SOCKET"])["reservations"]]
        print(json.dumps({
            "prefix": sys.prefix,
            "reserved_file": reserved.__file__,
            "safe_path": bool(sys.flags.safe_path),
            "path0": sys.path[0],
            "shadow_imported": shadow_imported,
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "path_head": os.environ["PATH"].split(os.pathsep)[0],
            "pythonpath": os.environ.get("PYTHONPATH"),
            "pythonhome": os.environ.get("PYTHONHOME"),
            "mark": os.environ["MARK"],
            "argv": sys.argv[1:],
            "id": lease.reservation_id,
            "active": active,
        }))


if __name__ == "__main__":
    main()
