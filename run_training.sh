#!/bin/bash

# Check if Python 3.10 is installed
if ! command -v python3.10 &> /dev/null; then
    echo "Python 3.10 is not installed. Please install Python 3.10 first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv310" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv venv310
fi

# Activate virtual environment
source venv310/bin/activate

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Run the training script
echo "Starting model training..."
python trading/scripts/train_models.py

# Deactivate virtual environment
deactivate

echo "Model training completed." 