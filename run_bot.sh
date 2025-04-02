#!/bin/bash
if [ -f /Users/maxime/BTC_BOT/BTC_BOT/.env ]; then
    source /Users/maxime/BTC_BOT/BTC_BOT/.env
    echo ".env file loaded successfully. Exporting variables..."
    export $(grep -v '^#' /Users/maxime/BTC_BOT/BTC_BOT/.env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi
source /Users/maxime/BTC_BOT/BTC_BOT/venv/bin/activate
echo "Loaded environment variables before running the script:"
echo "Environment variables after loading .env:"
cat /Users/maxime/BTC_BOT/BTC_BOT/.env
echo "Starting your_bot_script.py..."
python /Users/maxime/BTC_BOT/BTC_BOT/your_bot_script.py
