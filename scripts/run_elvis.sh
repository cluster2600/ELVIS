#!/bin/bash

# Retained compatibility paper-runtime launcher. V2 runtime activation remains NO-GO.
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

if ! command -v python3.14 >/dev/null 2>&1; then
    echo "Python 3.14 is required." >&2
    exit 2
fi

log_level="INFO"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            if [[ "${2:-}" != "paper" ]]; then
                echo "Only paper mode is supported." >&2
                exit 2
            fi
            shift 2
            ;;
        --log-level)
            log_level="${2:?missing value for --log-level}"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--mode paper] [--log-level LEVEL]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

exec python3.14 main.py --mode paper --log-level "$log_level"
