#!/bin/bash

echo "Installing Binance Futures Connector..."

# Activate virtual environment if it exists
if [ -d "venv_new" ]; then
    echo "Activating venv_new..."
    source venv_new/bin/activate
fi

# Install the futures connector
pip install binance-futures-connector

echo "Binance Futures Connector installed successfully!"
echo "The bot now supports both spot and futures trading with better API connectivity."