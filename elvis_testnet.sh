#!/bin/bash

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run main.py with specified arguments
echo "Starting ELVIS in paper mode with console dashboard..."
python main.py --mode paper --log-level DEBUG