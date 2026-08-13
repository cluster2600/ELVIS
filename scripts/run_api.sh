#!/bin/bash

# Python 3.14-only compatibility API launcher; installs no dependencies.
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

if ! command -v python3.14 >/dev/null 2>&1; then
    echo "Python 3.14 is required." >&2
    exit 2
fi

exec python3.14 trading/scripts/run_api.py "$@"
