#!/bin/bash

echo "Installing Binance Futures Connector..."

# Activate virtual environment if it exists
if [ -d "venv314" ]; then
    echo "Activating venv314..."
    source venv314/bin/activate
fi

# Install the futures connector
pip install binance-futures-connector

echo "Binance Futures Connector installed successfully!"
echo "The bot now supports both spot and futures trading with better API connectivity."