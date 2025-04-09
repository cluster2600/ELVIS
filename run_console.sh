#!/bin/bash

# Print header
echo "============================================="
echo "Starting ELVIS Trading System Console Dashboard"
echo "============================================="
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add the project root to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Activate virtual environment if it exists
if [ -d "venv310" ]; then
    echo "Activating virtual environment..."
    source venv310/bin/activate
fi

# Run the console dashboard
echo "🚀 Starting console dashboard..."
python -m main --mode paper --dashboard console

echo ""
echo "============================================="
echo "Console Dashboard Session Ended"
echo "=============================================" 