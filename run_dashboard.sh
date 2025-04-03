#!/bin/bash

# Print header
echo "============================================="
echo "Starting ELVIS Trading System Dashboard"
echo "============================================="
echo ""

# Function to check Python version
check_python_version() {
    local version=$(python3.10 --version 2>&1 | awk '{print $2}')
    if [[ $version == 3.10.* ]]; then
        return 0
    else
        return 1
    fi
}

# Check if Python 3.10 is installed and is the correct version
if ! command -v python3.10 &> /dev/null || ! check_python_version; then
    echo "❌ Python 3.10 not found or incorrect version. Please install Python 3.10 first."
    echo "You can install it using: brew install python@3.10"
    echo "After installation, make sure to run: brew link python@3.10"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv310" ]; then
    echo "❌ Virtual environment not found. Creating it..."
    python3.10 -m venv venv310
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv310/bin/activate
if [ $? -eq 0 ]; then
    echo "✅ Virtual environment activated"
    echo "📦 Python version: $(python --version)"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Install requirements if not already installed
echo "📦 Checking requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements"
    exit 1
fi

# Run the dashboard
echo "🚀 Starting dashboard..."
echo "📊 Mode: Paper Trading"
echo "🤖 Strategy: Ensemble"
echo "📈 Leverage: 125x"
echo "📝 Log Level: INFO"
echo "🖥️ Dashboard: Console"

# Run the main script with proper arguments
python main.py \
    --mode paper \
    --strategy ensemble \
    --leverage 125 \
    --log-level INFO \
    --dashboard console

# Deactivate virtual environment when done
deactivate

echo ""
echo "============================================="
echo "Dashboard Session Ended"
echo "============================================="
