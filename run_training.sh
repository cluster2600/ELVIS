#!/bin/bash

# Exit on any error
set -e

# Function to prefix messages with a timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Error handler
function handle_error() {
    log "❌ [ERROR] An error occurred during the training process."
    log "Cleaning up..."
    deactivate || true
    exit 1
}

trap 'handle_error' ERR

log "🔍 Checking if Python 3.10 is installed..."
if ! command -v python3.10 &> /dev/null; then
    log "❌ Python 3.10 is not installed. Please install it and rerun this script."
    exit 1
fi
log "✅ Python 3.10 is available."

# Virtual environment setup
if [ ! -d "venv310" ]; then
    log "📦 Creating virtual environment in ./venv310..."
    python3.10 -m venv venv310
else
    log "🌀 Virtual environment already exists."
fi

log "🚀 Activating virtual environment..."
source venv310/bin/activate

log "🐍 Python version in use: $(python --version)"

log "📚 Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

export PYTHONPATH=$(pwd):$PYTHONPATH
log "🛠️ PYTHONPATH set to: $PYTHONPATH"

log "📊 Processing trade history from database..."
python training/data/trade_history_processor.py

log "🎯 Starting model training with trade history..."
log "➡️ Command: python training/train_models.py --config training/config/model_config.yaml --include-trade-history"
python training/train_models.py --config training/config/model_config.yaml --include-trade-history

log "✅ Model training completed successfully."

log "🧹 Deactivating virtual environment..."
deactivate