#!/bin/bash

# Script to set up secure configuration loading for your_bot_script.py and copy credentials

# Define variables
VENV_PATH="/Users/maxime/BTC_BOT/BTC_BOT/venv314"
SCRIPT_PATH="/Users/maxime/BTC_BOT/BTC_BOT/your_bot_script.py"
ENV_FILE="/Users/maxime/BTC_BOT/BTC_BOT/.env"
SECRETS_FILE="/Users/maxime/BTC_BOT/BTC_BOT/secrets.txt"
APIKEYS_FILE="/Users/maxime/BTC_BOT/BTC_BOT/apikeys.txt"
BACKUP_SCRIPT="$SCRIPT_PATH.bak"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Check if secrets.txt and apikeys.txt exist
if [ ! -f "$SECRETS_FILE" ]; then
    echo "Error: $SECRETS_FILE not found. Please ensure it exists with TELEGRAM_TOKEN and TELEGRAM_CHAT_ID."
    exit 1
fi
if [ ! -f "$APIKEYS_FILE" ]; then
    echo "Error: $APIKEYS_FILE not found. Please ensure it exists with BINANCE_API_KEY and BINANCE_API_SECRET."
    exit 1
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Install python-dotenv
echo "Installing python-dotenv..."
pip install python-dotenv

# Extract credentials from secrets.txt and apikeys.txt
echo "Copying credentials to $ENV_FILE..."
TELEGRAM_TOKEN=$(grep "TELEGRAM_TOKEN" "$SECRETS_FILE" | cut -d'=' -f2- | tr -d ' ')
TELEGRAM_CHAT_ID=$(grep "TELEGRAM_CHAT_ID" "$SECRETS_FILE" | cut -d'=' -f2- | tr -d ' ')
BINANCE_API_KEY=$(grep "BINANCE_API_KEY" "$APIKEYS_FILE" | cut -d'=' -f2- | tr -d ' ')
BINANCE_API_SECRET=$(grep "BINANCE_API_SECRET" "$APIKEYS_FILE" | cut -d'=' -f2- | tr -d ' ')

# Validate that all credentials were found
if [ -z "$TELEGRAM_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ] || [ -z "$BINANCE_API_KEY" ] || [ -z "$BINANCE_API_SECRET" ]; then
    echo "Error: One or more credentials could not be extracted from $SECRETS_FILE or $APIKEYS_FILE."
    echo "Ensure they contain: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, BINANCE_API_KEY, BINANCE_API_SECRET"
    exit 1
fi

# Create or overwrite .env file with credentials
cat <<EOL > "$ENV_FILE"
# Binance API credentials
BINANCE_API_KEY=$BINANCE_API_KEY
BINANCE_API_SECRET=$BINANCE_API_SECRET

# Telegram credentials
TELEGRAM_TOKEN=$TELEGRAM_TOKEN
TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID
EOL
echo "Created $ENV_FILE with copied credentials."

# Backup the original script
if [ -f "$SCRIPT_PATH" ]; then
    cp "$SCRIPT_PATH" "$BACKUP_SCRIPT"
    echo "Backed up original script to $BACKUP_SCRIPT"
else
    echo "Error: $SCRIPT_PATH not found"
    exit 1
fi

# Modify your_bot_script.py to use environment variables
echo "Updating $SCRIPT_PATH to use environment variables..."
sed -i '' '/import os/a\
import dotenv\
dotenv.load_dotenv()' "$SCRIPT_PATH"

sed -i '' '/secrets = read_config("secrets.txt")/d' "$SCRIPT_PATH"
sed -i '' '/api_keys = read_config("apikeys.txt")/d' "$SCRIPT_PATH"
sed -i '' '/TELEGRAM_TOKEN = secrets.get("TELEGRAM_TOKEN")/d' "$SCRIPT_PATH"
sed -i '' '/TELEGRAM_CHAT_ID = secrets.get("TELEGRAM_CHAT_ID")/d' "$SCRIPT_PATH"
sed -i '' '/BINANCE_API_KEY = api_keys.get("BINANCE_API_KEY")/d' "$SCRIPT_PATH"
sed -i '' '/BINANCE_API_SECRET = api_keys.get("BINANCE_API_SECRET")/d' "$SCRIPT_PATH"
sed -i '' "/if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:/i\
# Load environment variables\
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')\
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')\
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')\
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')" "$SCRIPT_PATH"

# Optionally remove read_config function if not used elsewhere
# sed -i '' '/def read_config(filename: str) -> Dict[str, str]:/,/    return config/d' "$SCRIPT_PATH"

# Verify changes
echo "Verifying changes in $SCRIPT_PATH..."
grep -A 5 "Load environment variables" "$SCRIPT_PATH"

# Deactivate the virtual environment
deactivate

echo "Setup complete. Credentials copied to $ENV_FILE."
echo "Test the script with: source $VENV_PATH/bin/activate && python3 $SCRIPT_PATH"