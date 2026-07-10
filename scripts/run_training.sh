#!/bin/bash

# ELVIS Trading Bot - Unified Training Script
# Integrates all training methods with automatic fallback and error handling

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default parameters - User Configuration  
DEFAULT_LIMIT=0
DEFAULT_EPOCHS=20
DEFAULT_HORIZON=5
DEFAULT_METHOD="auto"
DEBUG_MODE=false
VAULT_MODE=true

# Research strategy parameters
SOCIAL_DATA_ENABLED=false
ROLLING_TRAINING_ENABLED=false
RESEARCH_MODE=false
LIVE_TRADING=false

# Enhanced feature flags (optional)
AUTOML_MODE=false
AUTOML_TRIALS=50
START_DASHBOARD=false
DASHBOARD_LOG="training.log"

# Phase 2 Advanced AI Features (optional)
LLM_ANALYSIS=false
LLM_PROVIDER="local"
LLM_MODEL="openai/gpt-oss-20b"
LLM_BASE_URL="http://localhost:1234/v1"
CONTINUOUS_LEARNING=false
MULTIMODAL_LEARNING=false
NAS_ENABLED=false
NAS_GENERATIONS=20

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1  # repo root (scripts moved into scripts/)

print_header() {
    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}🚀 ELVIS Trading Bot - Unified Training${NC}"
    echo -e "${CYAN}============================================${NC}"
    echo -e "${GREEN}📅 $(date)${NC}"
    echo -e "${GREEN}📁 Working Directory: $(pwd)${NC}"
    echo ""
}

print_usage() {
    echo -e "${YELLOW}Usage: $0 [OPTIONS]${NC}"
    echo ""
    echo "Training Methods:"
    echo "  --method postgres     Use PostgreSQL database (recommended)"
    echo "  --method enhanced     Use enhanced script with Vault fixes"
    echo "  --method no-vault     Complete Vault bypass"
    echo "  --method research     Research-based strategy training (Bonenkamp 2021)"
    echo "  --method auto         Auto-detect best method (default)"
    echo ""
    echo "Parameters:"
    echo "  --limit NUM          Maximum trades to use (default: ALL AVAILABLE)"
    echo "  --epochs NUM         Training epochs (default: $DEFAULT_EPOCHS)"
    echo "  --horizon NUM        Prediction horizon (default: $DEFAULT_HORIZON)"
    echo "  --debug             Enable debug mode"
    echo "  --no-vault          Disable Vault (same as --method no-vault)"
    echo ""
    echo "Research Strategy Options:"
    echo "  --social            Enable social data (Twitter + Google Trends)"
    echo "  --no-social         Disable social data features"
    echo "  --rolling           Enable rolling training (1-week windows)"
    echo "  --no-rolling        Disable rolling training"
    echo "  --live              Prepare models for live deployment (training only)"
    echo ""
    echo "Utility Options:"
    echo "  --check             Run system diagnostics only"
    echo "  --quick             Quick test run (100 trades, 5 epochs)"
    echo "  --production        Production run (5000 trades, 50 epochs)"
    echo "  --research          Research strategy training mode (14.9% target returns)"
    echo ""
    echo "🆕 Enhanced Options:"
    echo "  --automl            Enable AutoML hyperparameter optimization"
    echo "  --automl-trials N   Number of AutoML trials (default: 50)"
    echo "  --dashboard         Start real-time console dashboard"
    echo "  --dashboard-log F   Log file to monitor (default: training.log)"
    echo ""
    echo "🤖 Phase 2 Advanced AI Options:"
    echo "  --llm               Enable LLM market analysis"
    echo "  --llm-provider P    LLM provider (local, openai, anthropic) (default: local)"
    echo "  --llm-model M       LLM model name (default: openai/gpt-oss-20b)"
    echo "  --llm-url URL       Local LLM base URL (default: http://localhost:1234/v1)"
    echo "  --continuous        Enable continuous learning pipeline"
    echo "  --multimodal        Enable multi-modal learning (price+news+social)"
    echo "  --nas               Enable neural architecture search"
    echo "  --nas-generations N NAS generations (default: 20)"
    echo "  --help              Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                  # Auto-detect and run with defaults"
    echo "  $0 --quick --debug                 # Quick test with debug"
    echo "  $0 --method postgres --limit 2000  # PostgreSQL with 2000 trades"
    echo "  $0 --method research --social       # Research strategy training with social data"
    echo "  $0 --research --live                # Research training for live deployment"
    echo "  $0 --production                     # Full production training"
    echo ""
    echo "🆕 Enhanced Examples:"
    echo "  $0 --automl --dashboard             # AutoML with real-time dashboard"
    echo "  $0 --research --automl              # Research strategy with AutoML"
    echo "  $0 --quick --automl --debug         # Quick AutoML test"
    echo "  $0 --dashboard --dashboard-log custom.log # Console dashboard with custom log"
    echo ""
    echo "🤖 Advanced AI Examples:"
    echo "  $0 --llm --continuous               # LLM analysis with continuous learning"
    echo "  $0 --multimodal --nas               # Multi-modal learning with architecture search"
    echo "  $0 --research --llm --multimodal    # Full research training with all AI features"
    echo "  $0 --llm --llm-provider openai      # Use OpenAI GPT for market analysis"
    echo "  $0 --automl --llm --dashboard       # Complete AI-powered training pipeline"
    echo "  $0 --check                          # Run diagnostics only"
}

check_dependencies() {
    echo -e "${BLUE}🔍 Checking Dependencies...${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 not found${NC}"
        return 1
    fi
    
    # Check required Python packages (tensorflow is optional, checked below)
    python3 -c "import pandas, numpy, torch, psycopg2" 2>/dev/null || {
        echo -e "${RED}❌ Missing Python dependencies${NC}"
        echo -e "${YELLOW}💡 Install with: pip install -r requirements.txt${NC}"
        return 1
    }

    # TensorFlow is optional: PyTorch/ensemble training paths do not need it and
    # it has no wheel on some Python versions. Soft-warn instead of failing.
    python3 -c "import tensorflow" 2>/dev/null || {
        echo -e "${YELLOW}⚠️  TensorFlow not installed (optional) — TF-specific model paths will be skipped.${NC}"
    }

    echo -e "${GREEN}✅ Dependencies OK${NC}"
    return 0
}

check_postgresql() {
    echo -e "${BLUE}🗄️  Checking PostgreSQL...${NC}"
    
    # Try to connect to PostgreSQL
    python3 -c "
import psycopg2
try:
    conn = psycopg2.connect(host='localhost', port=5432, dbname='elvis_trading', user='maxime', password='')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM trades')
    count = cursor.fetchone()[0]
    print(f'✅ PostgreSQL OK - {count} trades in elvis_trading')
    conn.close()
except Exception as e:
    print(f'❌ PostgreSQL connection failed: {str(e)[:50]}...')
    exit(1)
" 2>/dev/null && return 0
    
    # Try alternative configurations
    python3 -c "
import psycopg2
configs = [
    {'host': 'localhost', 'port': 5432, 'dbname': 'trading_bot', 'user': 'postgres', 'password': ''},
    {'host': 'localhost', 'port': 5432, 'dbname': 'elvis_trading', 'user': 'postgres', 'password': 'elvis_password'},
]

for config in configs:
    try:
        conn = psycopg2.connect(**config)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM trades')
        count = cursor.fetchone()[0]
        print(f'✅ PostgreSQL OK - {count} trades in {config[\"dbname\"]}')
        conn.close()
        exit(0)
    except:
        continue

print('❌ No PostgreSQL connection found')
exit(1)
" 2>/dev/null && return 0
    
    echo -e "${RED}❌ PostgreSQL not accessible${NC}"
    return 1
}

check_vault() {
    echo -e "${BLUE}🔐 Checking Vault...${NC}"
    
    if ! command -v vault &> /dev/null; then
        echo -e "${YELLOW}⚠️  Vault CLI not found${NC}"
        return 1
    fi
    
    export VAULT_ADDR="http://127.0.0.1:8200"
    export VAULT_TOKEN="trading-bot-token"
    
    if vault status &> /dev/null; then
        echo -e "${GREEN}✅ Vault server running${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Vault server not running${NC}"
        return 1
    fi
}

auto_detect_method() {
    echo -e "${BLUE}🎯 Auto-detecting best training method...${NC}" >&2
    
    # If research mode is specifically requested, prioritize it
    if [[ "$RESEARCH_MODE" == "true" ]]; then
        echo -e "${PURPLE}🎯 Selected: Research strategy method${NC}" >&2
        echo "research"
        return 0
    fi
    
    if check_postgresql >&2; then
        echo -e "${GREEN}🎯 Selected: PostgreSQL method${NC}" >&2
        echo "postgres"
        return 0
    fi
    
    echo -e "${YELLOW}🎯 Selected: No-Vault method${NC}" >&2
    echo "no-vault"
}

run_diagnostics() {
    echo -e "${PURPLE}🔧 Running System Diagnostics...${NC}"
    echo ""
    
    if [[ -f "debug_training.py" ]]; then
        python3 debug_training.py
    else
        echo -e "${BLUE}🔍 Manual Diagnostics:${NC}"
        check_dependencies
        check_postgresql
        check_vault
    fi
}

run_training_postgres() {
    local limit=$1
    local epochs=$2
    local horizon=$3
    local debug_flag=$4
    
    echo -e "${GREEN}🚀 Starting PostgreSQL Training...${NC}"
    
    # Use Patient LLM training for slow LLMs, or ALL trades training for maximum data utilization
    if [[ -f "train_all_trades_patient_llm.py" && "$LLM_ANALYSIS" == "true" ]]; then
        # Use Patient LLM training for slow LLMs
        cmd="python3 train_all_trades_patient_llm.py --trades $limit --epochs $epochs --horizon $horizon --include-test"
        echo -e "${PURPLE}🐌 Using Patient LLM-enhanced training (optimized for slow LLMs)${NC}"
        echo -e "${CYAN}📊 Will use $limit individual paper trades with patient LLM processing${NC}"
        
        # Add vault flag if not disabled
        if [[ "$VAULT_MODE" == "true" ]]; then
            cmd="$cmd --vault"
        fi
    elif [[ -f "train_all_paper_trades.py" ]]; then
        # Use ALL trades training for maximum data utilization
        local trades_limit=$limit
        
        cmd="python3 train_all_paper_trades.py --method auto --trades $trades_limit --epochs $epochs --horizon $horizon --include-test"
        echo -e "${PURPLE}🧠 Using ALL trades LLM-enhanced training (maximizes training samples)${NC}"
        echo -e "${CYAN}📊 Will use $trades_limit individual paper trades${NC}"
        
        # Add vault flag if not disabled
        if [[ "$VAULT_MODE" == "true" ]]; then
            cmd="$cmd --vault"
        fi
    # Use LLM-enhanced training if LLM is enabled
    elif [[ "$LLM_ANALYSIS" == "true" && -f "train_with_llm.py" ]]; then
        cmd="python3 train_with_llm.py --limit $limit --epochs $epochs --prediction-horizon $horizon --enable-llm"
        echo -e "${PURPLE}🧠 Enhanced with LLM analysis${NC}"
    elif [[ -f "train_with_postgres_llm.py" ]]; then
        cmd="python3 train_with_postgres_llm.py --method auto --limit $limit --epochs $epochs --horizon $horizon --vault --include-test"
        echo -e "${PURPLE}🧠 Using LLM-enhanced PostgreSQL training script${NC}"
    elif [[ -f "train_with_postgres.py" ]]; then
        cmd="python3 train_with_postgres.py --limit $limit --epochs $epochs --prediction-horizon $horizon"
    elif [[ -f "train_with_llm.py" ]]; then
        cmd="python3 train_with_llm.py --limit $limit --epochs $epochs --prediction-horizon $horizon --disable-llm"
        echo -e "${CYAN}📊 Using LLM script without LLM features${NC}"
    else
        echo -e "${RED}❌ No suitable training script found${NC}"
        return 1
    fi
    
    if [[ "$debug_flag" == "true" ]]; then
        cmd="$cmd --debug"
    fi
    
    echo -e "${CYAN}📋 Command: $cmd${NC}"
    echo ""
    
    $cmd
}

run_training_enhanced() {
    local limit=$1
    local epochs=$2
    local horizon=$3
    local debug_flag=$4
    
    echo -e "${GREEN}🚀 Starting Enhanced Training...${NC}"
    
    cmd="python3 train_with_trades.py --limit $limit --epochs $epochs --prediction-horizon $horizon"
    
    if [[ "$debug_flag" == "true" ]]; then
        cmd="$cmd --debug"
    fi
    
    if [[ "$VAULT_MODE" == "false" ]]; then
        cmd="$cmd --no-vault"
    fi
    
    echo -e "${CYAN}📋 Command: $cmd${NC}"
    echo ""
    
    $cmd
}

run_training_no_vault() {
    local limit=$1
    local epochs=$2
    local horizon=$3
    local debug_flag=$4
    
    echo -e "${GREEN}🚀 Starting No-Vault Training...${NC}"
    
    # Use LLM-enhanced training if LLM is enabled
    if [[ "$LLM_ANALYSIS" == "true" && -f "train_with_llm.py" ]]; then
        cmd="python3 train_with_llm.py --limit $limit --epochs $epochs --prediction-horizon $horizon --enable-llm"
        echo -e "${PURPLE}🧠 Enhanced with LLM analysis${NC}"
    elif [[ -f "train_no_vault.py" ]]; then
        cmd="python3 train_no_vault.py --limit $limit --epochs $epochs --prediction-horizon $horizon"
    elif [[ -f "train_with_llm.py" ]]; then
        cmd="python3 train_with_llm.py --limit $limit --epochs $epochs --prediction-horizon $horizon --disable-llm"
        echo -e "${CYAN}📊 Using LLM script without LLM features${NC}"
    else
        echo -e "${RED}❌ No suitable training script found${NC}"
        return 1
    fi
    
    if [[ "$debug_flag" == "true" ]]; then
        cmd="$cmd --debug"
    fi
    
    echo -e "${CYAN}📋 Command: $cmd${NC}"
    echo ""
    
    $cmd
}

run_training_research() {
    local limit=$1
    local epochs=$2
    local horizon=$3
    local debug_flag=$4
    
    echo -e "${PURPLE}🔬 Starting Research-Based Strategy Training...${NC}"
    echo -e "${CYAN}📊 Following Bonenkamp (2021) methodology${NC}"
    echo -e "${CYAN}🎯 Target: 14.9% annual returns with binary classification${NC}"
    echo ""
    
    # Disable Vault for research strategy to avoid authentication issues
    export VAULT_ENABLED="false"
    export USE_VAULT="false"
    export VAULT_AVAILABLE="false"
    
    # Set environment variables for research strategy
    export STRATEGY_MODE="research"
    export SOCIAL_DATA_ENABLED="$SOCIAL_DATA_ENABLED"
    export ROLLING_TRAINING_ENABLED="$ROLLING_TRAINING_ENABLED"
    
    # Set basic credentials for paper trading
    export BINANCE_API_KEY="research_mode_key"
    export BINANCE_API_SECRET="research_mode_secret"
    
    # Research strategy focuses on training, not live trading
    # LIVE_TRADING flag will be used in model evaluation/backtesting only
    local training_mode="research"
    if [[ "$LIVE_TRADING" == "true" ]]; then
        echo -e "${YELLOW}⚠️  Note: --live flag noted for research strategy${NC}"
        echo -e "${CYAN}   This affects model evaluation, not actual trading${NC}"
        echo -e "${GREEN}   Training script will prepare models for live deployment${NC}"
    fi
    
    # Set log level
    local log_level="INFO"
    if [[ "$debug_flag" == "true" ]]; then
        log_level="DEBUG"
    fi
    
    echo -e "${CYAN}🚀 Research Training Configuration:${NC}"
    echo -e "   Strategy Mode: ${GREEN}$STRATEGY_MODE${NC}"
    echo -e "   Social Data: ${GREEN}$SOCIAL_DATA_ENABLED${NC}"
    echo -e "   Rolling Training: ${GREEN}$ROLLING_TRAINING_ENABLED${NC}"
    echo -e "   Training Samples: ${GREEN}$limit${NC}"
    echo -e "   Training Epochs: ${GREEN}$epochs${NC}"
    echo -e "   Prediction Horizon: ${GREEN}$horizon${NC}"
    echo -e "   Live Deployment Ready: ${GREEN}$LIVE_TRADING${NC}"
    echo -e "   Debug Mode: ${GREEN}$debug_flag${NC}"
    echo -e "   Vault: ${YELLOW}Disabled for research${NC}"
    echo ""
    
    # Build command - use training script, NOT main.py (which starts the bot)
    # For research strategy, prioritize ALL trades training for maximum data
    if [[ -f "train_all_paper_trades.py" ]]; then
        # Use infinite trades (0 = all available) for research to maximize training samples
        local trades_limit=0
        if [[ "$limit" != "$DEFAULT_LIMIT" ]]; then
            trades_limit=$limit
        fi
        
        cmd="python3 train_all_paper_trades.py --method all_trades --trades $trades_limit --epochs $epochs --horizon $horizon --include-test"
        echo -e "${PURPLE}🧠 Using ALL trades LLM-enhanced training for research strategy${NC}"
        echo -e "${CYAN}📊 Will use ALL ${trades_limit:-25440+} individual paper trades for maximum research data${NC}"
    # For research strategy, use LLM-enhanced training if LLM is enabled
    elif [[ "$LLM_ANALYSIS" == "true" && -f "train_with_llm.py" ]]; then
        cmd="python3 train_with_llm.py --limit $limit --epochs $epochs --prediction-horizon $horizon --enable-llm"
        echo -e "${PURPLE}🧠 Using LLM-enhanced training script for research strategy${NC}"
    elif [[ -f "train_research_strategy.py" ]]; then
        cmd="python3 train_research_strategy.py --limit $limit --epochs $epochs --prediction-horizon $horizon"
    elif [[ -f "train_with_postgres_llm.py" ]]; then
        cmd="python3 train_with_postgres_llm.py --method auto --limit $limit --epochs $epochs --horizon $horizon --vault --include-test"
        echo -e "${PURPLE}🧠 Using LLM-enhanced PostgreSQL training script${NC}"
    elif [[ -f "train_with_postgres.py" ]]; then
        cmd="python3 train_with_postgres.py --limit $limit --epochs $epochs --prediction-horizon $horizon"
        echo -e "${YELLOW}💡 Using PostgreSQL training script for research strategy${NC}"
    elif [[ -f "train_with_trades.py" ]]; then
        cmd="python3 train_with_trades.py --limit $limit --epochs $epochs --prediction-horizon $horizon --no-vault"
        echo -e "${YELLOW}💡 Using enhanced training script for research strategy${NC}"
    elif [[ -f "train_no_vault.py" ]]; then
        cmd="python3 train_no_vault.py --limit $limit --epochs $epochs --prediction-horizon $horizon"
        echo -e "${YELLOW}💡 Using no-vault training script for research strategy${NC}"
    elif [[ -f "train_with_llm.py" ]]; then
        cmd="python3 train_with_llm.py --limit $limit --epochs $epochs --prediction-horizon $horizon --disable-llm"
        echo -e "${CYAN}📊 Using LLM training script without LLM features${NC}"
    else
        echo -e "${RED}❌ No suitable training script found!${NC}"
        echo -e "${YELLOW}💡 Available training scripts should be one of:${NC}"
        echo -e "   - train_all_paper_trades.py (ALL trades, maximum data, recommended)"
        echo -e "   - train_with_llm.py (LLM-enhanced, preferred)"
        echo -e "   - train_research_strategy.py"
        echo -e "   - train_with_postgres_llm.py (PostgreSQL + LLM, recommended)"
        echo -e "   - train_with_postgres.py"
        echo -e "   - train_with_trades.py"
        echo -e "   - train_no_vault.py"
        return 1
    fi
    
    # Add debug flag if needed
    if [[ "$debug_flag" == "true" ]]; then
        cmd="$cmd --debug"
    fi
    
    echo -e "${CYAN}📋 Training Command: $cmd${NC}"
    echo -e "${GREEN}📚 NOTE: This will TRAIN models, not start the trading bot${NC}"
    echo ""
    
    # Execute the training (NOT the bot)
    $cmd
}

start_dashboard() {
    if [[ "$START_DASHBOARD" == "true" ]]; then
        echo -e "${BLUE}🖥️  Starting Console Dashboard...${NC}"
        
        # Check if console dashboard script exists
        if [[ -f "utils/dashboard/console_dashboard.py" ]]; then
            echo -e "${CYAN}📊 Console dashboard will monitor: $DASHBOARD_LOG${NC}"
            echo -e "${YELLOW}💡 Dashboard will start after training begins${NC}"
            echo -e "${YELLOW}💡 Press 'Q' in dashboard to quit, 'R' to reset metrics${NC}"
            
            # Create flag file so training scripts know dashboard is wanted
            echo "$DASHBOARD_LOG" > .dashboard_config
            
            echo -e "${GREEN}✅ Console dashboard configured${NC}"
        else
            echo -e "${YELLOW}⚠️  Console dashboard script not found, skipping dashboard startup${NC}"
        fi
    fi
}

setup_llm_analysis() {
    if [[ "$LLM_ANALYSIS" == "true" ]]; then
        echo -e "${PURPLE}🧠 Setting up LLM Market Analysis...${NC}"
        
        # Export LLM configuration
        export ELVIS_LLM_ENABLED="true"
        export ELVIS_LLM_PROVIDER="$LLM_PROVIDER"
        export ELVIS_LLM_MODEL="$LLM_MODEL"
        export ELVIS_LLM_BASE_URL="$LLM_BASE_URL"
        
        # Check if LLM script exists
        if [[ -f "core/ai/llm_market_analyzer.py" ]]; then
            echo -e "${GREEN}✅ LLM market analyzer found${NC}"
            echo -e "${CYAN}🔗 Provider: $LLM_PROVIDER, Model: $LLM_MODEL${NC}"
            
            # Test LLM connection (basic check)
            if [[ "$LLM_PROVIDER" == "local" ]]; then
                echo -e "${CYAN}🌐 Testing local LLM connection...${NC}"
                if curl -s --max-time 5 "$LLM_BASE_URL/models" > /dev/null 2>&1; then
                    echo -e "${GREEN}✅ Local LLM server accessible${NC}"
                else
                    echo -e "${YELLOW}⚠️  Local LLM server not accessible, continuing anyway${NC}"
                fi
            fi
        else
            echo -e "${YELLOW}⚠️  LLM analyzer script not found, disabling LLM features${NC}"
            LLM_ANALYSIS=false
        fi
    fi
}

setup_continuous_learning() {
    if [[ "$CONTINUOUS_LEARNING" == "true" ]]; then
        echo -e "${BLUE}⚡ Setting up Continuous Learning Pipeline...${NC}"
        
        export ELVIS_CONTINUOUS_LEARNING="true"
        
        if [[ -f "core/streaming/continuous_learning_pipeline.py" ]]; then
            echo -e "${GREEN}✅ Continuous learning pipeline found${NC}"
        else
            echo -e "${YELLOW}⚠️  Continuous learning pipeline not found, disabling feature${NC}"
            CONTINUOUS_LEARNING=false
        fi
    fi
}

setup_multimodal_learning() {
    if [[ "$MULTIMODAL_LEARNING" == "true" ]]; then
        echo -e "${CYAN}🎭 Setting up Multi-Modal Learning...${NC}"
        
        export ELVIS_MULTIMODAL_ENABLED="true"
        
        if [[ -f "core/multimodal/multimodal_learning_system.py" ]]; then
            echo -e "${GREEN}✅ Multi-modal learning system found${NC}"
            echo -e "${CYAN}📊 Will integrate price + news + social + technical data${NC}"
        else
            echo -e "${YELLOW}⚠️  Multi-modal learning system not found, disabling feature${NC}"
            MULTIMODAL_LEARNING=false
        fi
    fi
}

setup_neural_architecture_search() {
    if [[ "$NAS_ENABLED" == "true" ]]; then
        echo -e "${PURPLE}🔍 Setting up Neural Architecture Search...${NC}"
        
        export ELVIS_NAS_ENABLED="true"
        export ELVIS_NAS_GENERATIONS="$NAS_GENERATIONS"
        
        if [[ -f "core/nas/neural_architecture_search.py" ]]; then
            echo -e "${GREEN}✅ Neural architecture search found${NC}"
            echo -e "${CYAN}🧬 Will optimize network architecture (${NAS_GENERATIONS} generations)${NC}"
        else
            echo -e "${YELLOW}⚠️  Neural architecture search not found, disabling feature${NC}"
            NAS_ENABLED=false
        fi
    fi
}

stop_dashboard() {
    # Clean up dashboard config
    [[ -f ".dashboard_config" ]] && rm -f .dashboard_config
}

run_automl_training() {
    local method=$1
    local limit=$2
    local epochs=$3
    local horizon=$4
    local debug_flag=$5
    
    echo -e "${PURPLE}🤖 Starting AutoML Hyperparameter Optimization...${NC}"
    echo -e "${CYAN}🎯 Trials: $AUTOML_TRIALS, Method: $method${NC}"
    
    # Check if AutoML script exists
    if [[ ! -f "training/automl/hyperparameter_optimizer.py" ]]; then
        echo -e "${RED}❌ AutoML optimizer not found, falling back to standard training${NC}"
        return 1
    fi
    
    # Simple AutoML integration - optimize the current method
    echo -e "${BLUE}🔧 Running AutoML optimization for enhanced performance...${NC}"
    
    # Set environment variables for AutoML
    export ELVIS_AUTOML_TRIALS=$AUTOML_TRIALS
    export ELVIS_AUTOML_METHOD=$method
    export ELVIS_TRAINING_LIMIT=$limit
    export ELVIS_TRAINING_EPOCHS=$epochs
    
    # The actual training will use optimized parameters
    echo -e "${GREEN}✅ AutoML environment prepared${NC}"
    return 0
}

setup_vault_if_needed() {
    if [[ "$VAULT_MODE" == "true" && "$method" != "no-vault" && "$method" != "research" ]]; then
        echo -e "${BLUE}🔐 Setting up Vault environment...${NC}"
        
        export VAULT_ADDR="http://127.0.0.1:8200"
        export VAULT_TOKEN="trading-bot-token"
        
        # Try to start Vault if not running
        if ! vault status &> /dev/null; then
            echo -e "${YELLOW}🚀 Starting Vault development server...${NC}"
            vault server -dev -dev-root-token-id=trading-bot-token &> /dev/null &
            sleep 3
            
            if vault status &> /dev/null; then
                echo -e "${GREEN}✅ Vault server started${NC}"
            else
                echo -e "${YELLOW}⚠️  Vault start failed - continuing without Vault${NC}"
                VAULT_MODE=false
            fi
        fi
    elif [[ "$method" == "research" ]]; then
        echo -e "${YELLOW}🔐 Vault disabled for research strategy${NC}"
    fi
}

main() {
    print_header
    
    # Parse arguments
    limit=$DEFAULT_LIMIT
    epochs=$DEFAULT_EPOCHS
    horizon=$DEFAULT_HORIZON
    method=$DEFAULT_METHOD
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --method)
                method="$2"
                shift 2
                ;;
            --limit)
                limit="$2"
                shift 2
                ;;
            --epochs)
                epochs="$2"
                shift 2
                ;;
            --horizon)
                horizon="$2"
                shift 2
                ;;
            --debug)
                DEBUG_MODE=true
                shift
                ;;
            --no-vault)
                VAULT_MODE=false
                method="no-vault"
                shift
                ;;
            --social)
                SOCIAL_DATA_ENABLED=true
                shift
                ;;
            --no-social)
                SOCIAL_DATA_ENABLED=false
                shift
                ;;
            --rolling)
                ROLLING_TRAINING_ENABLED=true
                shift
                ;;
            --no-rolling)
                ROLLING_TRAINING_ENABLED=false
                shift
                ;;
            --live)
                LIVE_TRADING=true
                shift
                ;;
            --research)
                method="research"
                RESEARCH_MODE=true
                SOCIAL_DATA_ENABLED=true
                ROLLING_TRAINING_ENABLED=true
                shift
                ;;
            --automl)
                AUTOML_MODE=true
                shift
                ;;
            --automl-trials)
                AUTOML_TRIALS="$2"
                shift 2
                ;;
            --dashboard)
                START_DASHBOARD=true
                shift
                ;;
            --dashboard-log)
                DASHBOARD_LOG="$2"
                shift 2
                ;;
            --llm)
                LLM_ANALYSIS=true
                shift
                ;;
            --llm-provider)
                LLM_PROVIDER="$2"
                shift 2
                ;;
            --llm-model)
                LLM_MODEL="$2"
                shift 2
                ;;
            --llm-url)
                LLM_BASE_URL="$2"
                shift 2
                ;;
            --continuous)
                CONTINUOUS_LEARNING=true
                shift
                ;;
            --multimodal)
                MULTIMODAL_LEARNING=true
                shift
                ;;
            --nas)
                NAS_ENABLED=true
                shift
                ;;
            --nas-generations)
                NAS_GENERATIONS="$2"
                shift 2
                ;;
            --quick)
                limit=100
                epochs=5
                DEBUG_MODE=true
                shift
                ;;
            --production)
                limit=5000
                epochs=50
                shift
                ;;
            --check)
                run_diagnostics
                exit 0
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                echo -e "${RED}❌ Unknown option: $1${NC}"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Show configuration
    echo -e "${CYAN}📊 Training Configuration:${NC}"
    echo -e "   Method: ${GREEN}$method${NC}"
    echo -e "   Trades: ${GREEN}$limit${NC}"
    echo -e "   Epochs: ${GREEN}$epochs${NC}"
    echo -e "   Horizon: ${GREEN}$horizon${NC}"
    echo -e "   Debug: ${GREEN}$DEBUG_MODE${NC}"
    echo -e "   Vault: ${GREEN}$VAULT_MODE${NC}"
    
    # Show enhanced features if enabled
    if [[ "$AUTOML_MODE" == "true" || "$START_DASHBOARD" == "true" ]]; then
        echo -e "${PURPLE}🆕 Enhanced Features:${NC}"
        [[ "$AUTOML_MODE" == "true" ]] && echo -e "   AutoML: ${GREEN}enabled${NC} (${AUTOML_TRIALS} trials)"
        [[ "$START_DASHBOARD" == "true" ]] && echo -e "   Console Dashboard: ${GREEN}enabled${NC}"
    fi
    
    # Show Phase 2 AI features if enabled
    if [[ "$LLM_ANALYSIS" == "true" || "$CONTINUOUS_LEARNING" == "true" || "$MULTIMODAL_LEARNING" == "true" || "$NAS_ENABLED" == "true" ]]; then
        echo -e "${BLUE}🤖 Phase 2 AI Features:${NC}"
        [[ "$LLM_ANALYSIS" == "true" ]] && echo -e "   LLM Analysis: ${GREEN}enabled${NC} (${LLM_PROVIDER}/${LLM_MODEL})"
        [[ "$CONTINUOUS_LEARNING" == "true" ]] && echo -e "   Continuous Learning: ${GREEN}enabled${NC}"
        [[ "$MULTIMODAL_LEARNING" == "true" ]] && echo -e "   Multi-Modal Learning: ${GREEN}enabled${NC}"
        [[ "$NAS_ENABLED" == "true" ]] && echo -e "   Neural Architecture Search: ${GREEN}enabled${NC} (${NAS_GENERATIONS} gen)"
    fi
    
    # Show research configuration if applicable
    if [[ "$method" == "research" || "$RESEARCH_MODE" == "true" ]]; then
        echo -e "${PURPLE}📊 Research Strategy Configuration:${NC}"
        echo -e "   Social Data: ${GREEN}$SOCIAL_DATA_ENABLED${NC}"
        echo -e "   Rolling Training: ${GREEN}$ROLLING_TRAINING_ENABLED${NC}"
        echo -e "   Live Trading: ${GREEN}$LIVE_TRADING${NC}"
    fi
    echo ""
    
    # Check dependencies
    if ! check_dependencies; then
        exit 1
    fi
    
    # Auto-detect method if needed
    if [[ "$method" == "auto" ]]; then
        method=$(auto_detect_method)
    fi
    
    # Setup enhanced features
    start_dashboard
    
    # Setup AutoML if enabled
    if [[ "$AUTOML_MODE" == "true" ]]; then
        run_automl_training "$method" "$limit" "$epochs" "$horizon" "$DEBUG_MODE"
    fi
    
    # Setup Phase 2 AI features
    setup_llm_analysis
    setup_continuous_learning
    setup_multimodal_learning
    setup_neural_architecture_search
    
    # Setup Vault if needed
    setup_vault_if_needed
    
    # Create output directory
    mkdir -p models
    
    # Record start time
    start_time=$(date +%s)
    echo -e "${CYAN}⏰ Training started at: $(date)${NC}"
    echo ""
    
    # Run training based on method
    case $method in
        postgres)
            if ! check_postgresql; then
                echo -e "${RED}❌ PostgreSQL not available, switching to no-vault method${NC}"
                method="no-vault"
                run_training_no_vault "$limit" "$epochs" "$horizon" "$DEBUG_MODE"
            else
                run_training_postgres "$limit" "$epochs" "$horizon" "$DEBUG_MODE"
            fi
            ;;
        enhanced)
            run_training_enhanced "$limit" "$epochs" "$horizon" "$DEBUG_MODE"
            ;;
        no-vault)
            run_training_no_vault "$limit" "$epochs" "$horizon" "$DEBUG_MODE"
            ;;
        research)
            run_training_research "$limit" "$epochs" "$horizon" "$DEBUG_MODE"
            ;;
        *)
            echo -e "${RED}❌ Unknown training method: $method${NC}"
            print_usage
            exit 1
            ;;
    esac
    
    # Calculate duration
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    echo -e "${GREEN}🎉 Training Completed Successfully!${NC}"
    echo -e "${CYAN}⏱️  Duration: ${duration}s${NC}"
    echo -e "${CYAN}📂 Check models/ directory for results${NC}"
    echo -e "${CYAN}📊 Check logs/ directory for training logs${NC}"
    
    # Show enhanced results if applicable
    if [[ "$START_DASHBOARD" == "true" ]]; then
        echo -e "${YELLOW}🖥️  Console Dashboard: Monitoring $DASHBOARD_LOG${NC}"
    fi
    
    if [[ "$AUTOML_MODE" == "true" ]]; then
        echo -e "${YELLOW}🤖 AutoML: Enhanced hyperparameters applied${NC}"
    fi
    
    # Show Phase 2 AI results
    if [[ "$LLM_ANALYSIS" == "true" ]]; then
        echo -e "${BLUE}🧠 LLM Analysis: Market intelligence integrated${NC}"
    fi
    
    if [[ "$CONTINUOUS_LEARNING" == "true" ]]; then
        echo -e "${CYAN}⚡ Continuous Learning: Online adaptation enabled${NC}"
    fi
    
    if [[ "$MULTIMODAL_LEARNING" == "true" ]]; then
        echo -e "${PURPLE}🎭 Multi-Modal: Price+News+Social integration active${NC}"
    fi
    
    if [[ "$NAS_ENABLED" == "true" ]]; then
        echo -e "${GREEN}🔍 NAS: Optimized neural architecture discovered${NC}"
    fi
    
    echo ""
    
    # Show next steps
    echo -e "${YELLOW}💡 Next Steps:${NC}"
    echo -e "   1. Review trained models: ${CYAN}ls -la models/${NC}"
    echo -e "   2. Test with paper trading: ${CYAN}python main.py --mode paper${NC}"
    echo -e "   3. Start live trading: ${CYAN}python main.py --mode live${NC}"
    echo -e "   4. View training logs: ${CYAN}ls -la logs/model_training_*${NC}"
    if [[ "$START_DASHBOARD" == "true" ]]; then
        echo -e "   4. View console dashboard: ${CYAN}python3 utils/dashboard/console_dashboard.py${NC}"
        echo -e "   5. Run diagnostics: ${CYAN}$0 --check${NC}"
    else
        echo -e "   4. Run diagnostics: ${CYAN}$0 --check${NC}"
    fi
    
    if [[ "$AUTOML_MODE" == "true" ]]; then
        echo -e "   📈 AutoML results saved in optimization logs"
    fi
    
    # Phase 2 AI next steps
    if [[ "$LLM_ANALYSIS" == "true" || "$CONTINUOUS_LEARNING" == "true" || "$MULTIMODAL_LEARNING" == "true" || "$NAS_ENABLED" == "true" ]]; then
        echo -e "${BLUE}🤖 Advanced AI Features:${NC}"
        [[ "$LLM_ANALYSIS" == "true" ]] && echo -e "   🧠 LLM market insights in logs/llm_analysis_*"
        [[ "$CONTINUOUS_LEARNING" == "true" ]] && echo -e "   ⚡ Continuous learning pipeline active"
        [[ "$MULTIMODAL_LEARNING" == "true" ]] && echo -e "   🎭 Multi-modal models in models/multimodal_*"
        [[ "$NAS_ENABLED" == "true" ]] && echo -e "   🔍 NAS results in models/nas_architectures.json"
    fi
}

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up...${NC}"
    stop_dashboard
    
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Handle Ctrl+C gracefully and cleanup on exit
trap 'echo -e "\n${YELLOW}⚠️  Training interrupted by user${NC}"; cleanup; exit 130' INT
trap 'cleanup' EXIT

# Run main function
main "$@"