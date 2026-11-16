"""Print a launchd plist using absolute paths, including XML escaping."""

import argparse
import os
from pathlib import Path
import plistlib
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    args = parser.parse_args()
    venv = args.venv.expanduser().absolute()
    home = Path.home()
    agent = {
        "Label": "com.reserved.daemon",
        "ProgramArguments": [
            str(venv / "bin" / "python"),
            "-m", "reserved.daemon",
            "--config", str(home / ".config" / "reserved" / "reserved.json"),
            "--socket", f"/tmp/reserved-{os.getuid()}.sock",
        ],
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(home / "Library" / "Logs" / "reserved.log"),
        "StandardErrorPath": str(home / "Library" / "Logs" / "reserved.log"),
    }
    plistlib.dump(agent, sys.stdout.buffer)


if __name__ == "__main__":
    main()
