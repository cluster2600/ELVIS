#!/bin/bash

# Run ELVIS with console dashboard enabled
# This script runs the ELVIS bot in paper trading mode with the console dashboard enabled

# ASCII Art for ELVIS
echo " _______  _        __      __  _____   _____ "
echo "|  ____| | |       \ \    / / |_   _| / ____|"
echo "| |__    | |        \ \  / /    | |  | (___  "
echo "|  __|   | |         \ \/ /     | |   \___ \ "
echo "| |____  | |____      \  /     _| |_  ____) |"
echo "|______| |______|      \/     |_____||_____/ "
echo ""
echo "Enhanced Leveraged Virtual Investment System - Console Dashboard Mode"
echo ""
echo "Press 'q' to quit the dashboard"
echo ""

# Check if Python virtual environment exists
VENV_DIR="venv310"
if [ -d "$VENV_DIR" ]; then
    echo "Using Python virtual environment: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    echo "Warning: Python virtual environment not found. Using system Python."
fi

# Install dependencies (macOS doesn't need windows-curses)
pip install websocket-client psutil >/dev/null 2>&1 || {
    echo "Error: Failed to install required Python packages."
    exit 1
}

# Set working directory to nested project root
cd /Users/maxime/BTC_BOT/BTC_BOT

# Run the bot in paper trading mode with Ensemble strategy
echo "Starting ELVIS in paper trading mode with Ensemble strategy..."
# Redirect stderr to /dev/null to prevent log messages from appearing in the terminal
python main.py --mode paper --strategy ensemble --leverage 125 --log-level INFO "$@" 2>/dev/null

# Check exit status
if [ $? -eq 0 ]; then
    echo "ELVIS dashboard stopped successfully."
else
    echo "Error: ELVIS dashboard encountered an issue."
fi

# Deactivate virtual environment if it was activated
if [ -d "$VENV_DIR" ]; then
    deactivate
fi
