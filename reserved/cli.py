"""Parse command help and preserve exit status 2 for usage errors."""

import sys
from typing import NoReturn

from docopt import DocoptExit, ParsedOptions, docopt


def parse_options(usage: str, argv: list[str] | None = None) -> ParsedOptions:
    try:
        return docopt(usage, argv=argv)
    except DocoptExit as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from None


def usage_error(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)
