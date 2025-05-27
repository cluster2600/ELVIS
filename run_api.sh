#!/bin/bash

# ELVIS Trading Bot - API Runner Script
# This script starts the REST API server

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting ELVIS Trading Bot API...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if needed
echo -e "${YELLOW}Checking dependencies...${NC}"
pip install -q -r requirements.txt

# Default values
PORT=5000
HOST="0.0.0.0"
DEBUG=""
WORKERS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}API Configuration:${NC}"
echo -e "  Host: ${HOST}"
echo -e "  Port: ${PORT}"
echo -e "  Workers: ${WORKERS}"
if [ ! -z "$DEBUG" ]; then
    echo -e "  ${YELLOW}Debug Mode: ENABLED${NC}"
fi

echo -e "\n${GREEN}API Documentation will be available at:${NC}"
echo -e "  http://localhost:${PORT}/api/docs"
echo -e "\n${GREEN}Health Check endpoint:${NC}"
echo -e "  http://localhost:${PORT}/health"

# Run the API
echo -e "\n${GREEN}Starting API server...${NC}"
python trading/scripts/run_api.py --host $HOST --port $PORT --workers $WORKERS $DEBUG
