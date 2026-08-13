#!/bin/bash

# Python 3.14-only entry point for the canonical paper-training pipeline.
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

if ! command -v python3.14 >/dev/null 2>&1; then
    echo "Python 3.14 is required." >&2
    exit 2
fi

exec python3.14 -m training.train_models "$@"
