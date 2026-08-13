#!/bin/bash

# ELVIS Research-Based Strategy Runner
# Based on Bonenkamp (2021) academic paper methodology


# Resolve repo root so this script works from any cwd (moved into scripts/).
cd "$(dirname "$0")/.." || exit 1
echo "🔬 ELVIS Research-Based Trading Strategy"
echo "========================================"
echo "📊 Paper-only research strategy with binary classification"
echo "🎯 Historical results are not forward-looking performance claims"
echo ""

# Default settings
STRATEGY_MODE="research"
SOCIAL_DATA_ENABLED="true"
ROLLING_TRAINING_ENABLED="true"
MODE="paper"
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-social)
            SOCIAL_DATA_ENABLED="false"
            echo "📱 Social data disabled"
            shift
            ;;
        --no-rolling)
            ROLLING_TRAINING_ENABLED="false"
            echo "🔄 Rolling training disabled"
            shift
            ;;
        --debug)
            LOG_LEVEL="DEBUG"
            echo "🐛 Debug logging enabled"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --no-social     Disable social features (Twitter + Google Trends)"
            echo "  --no-rolling    Disable rolling training (use static model)"
            echo "  --debug         Enable debug logging"
            echo "  -h, --help      Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                    # Full research strategy with all features"
            echo "  $0 --no-social       # Research strategy without social data"
            echo "  $0 --debug           # Debug mode for troubleshooting"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Configuration:"
echo "   Strategy Mode: $STRATEGY_MODE"
echo "   Social Data: $SOCIAL_DATA_ENABLED"
echo "   Rolling Training: $ROLLING_TRAINING_ENABLED"
echo "   Trading Mode: $MODE"
echo "   Log Level: $LOG_LEVEL"
echo ""

echo "🎯 Starting ELVIS with Research-Based Strategy..."
echo "   Mode: paper only"
echo "   Method: Binary classification (BUY/SELL only)"
echo "   Basis: Bonenkamp (2021) academic research"
echo ""

# Export environment variables and run
export STRATEGY_MODE="$STRATEGY_MODE"
export SOCIAL_DATA_ENABLED="$SOCIAL_DATA_ENABLED"
export ROLLING_TRAINING_ENABLED="$ROLLING_TRAINING_ENABLED"

# Run the bot
python3.14 main.py --mode "$MODE" --log-level "$LOG_LEVEL"
