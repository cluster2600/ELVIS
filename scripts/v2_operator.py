"""Source-only dispatcher for the ELVIS V2 paper-migration operator.

This module deliberately exposes only the four bounded PostgreSQL operator
commands.  It does not start a trading runtime and it does not authorize an
ACTIVE cut-over.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from typing import Protocol

_VERSION = "2.0.0-alpha.2"
_COMMANDS = {
    "bootstrap": (
        "scripts.postgres_bootstrap",
        "stage or reconcile roles and schema on an operator-selected target",
    ),
    "cutover-preflight": (
        "scripts.postgres_cutover_preflight",
        "inspect a stopped source clone and a fresh target without activation",
    ),
    "import-snapshot": (
        "scripts.postgres_legacy_snapshot_import",
        "import a bounded legacy snapshot into a disposable target",
    ),
    "reconcile-snapshot": (
        "scripts.postgres_legacy_snapshot_reconciliation",
        "assess imported snapshot evidence without activating a runtime",
    ),
}


class _CommandModule(Protocol):
    def main(self, argv: Sequence[str] | None = None) -> int: ...


def _write_help() -> None:
    command_lines = "\n".join(
        f"  {name:<20} {description}" for name, (_, description) in _COMMANDS.items()
    )
    print(f"""ELVIS V2 operator preview {_VERSION}

Usage:
  elvis-v2-operator <command> [arguments]
  python -m scripts.v2_operator <command> [arguments]

Commands:
{command_lines}

Status:
  ACTIVE NO-GO. Paper/migration preview only. This dispatcher cannot start a
  trading runtime or authorize live execution. Run '<command> --help' for the
  bounded command's required files and explicit confirmation flags.
""")


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one exact operator command without command abbreviation."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments in (["--help"], ["-h"]):
        _write_help()
        return 0
    if arguments == ["--version"]:
        print(_VERSION)
        return 0
    if not arguments:
        _write_help()
        return 2

    command = arguments.pop(0)
    command_spec = _COMMANDS.get(command)
    if command_spec is None:
        print(f"unknown operator command: {command}", file=sys.stderr)
        _write_help()
        return 2

    module = importlib.import_module(command_spec[0])
    command_module: _CommandModule = module
    return int(command_module.main(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
