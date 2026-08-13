#!/bin/bash

# ELVIS Trading Bot - Ansible Setup Script
# This script runs the Ansible playbook to install all dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required commands
check_requirements() {
    print_status "Checking requirements..."
    
    if ! command_exists ansible-playbook; then
        print_error "Ansible is not installed. Please install Ansible first:"
        echo "  Ubuntu/Debian: sudo apt update && sudo apt install ansible"
        echo "  CentOS/RHEL: sudo yum install ansible"
        echo "  macOS: brew install ansible"
        echo "  or: pip install ansible"
        exit 1
    fi
    
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3 first."
        exit 1
    fi
    
    print_success "Requirements check passed"
}

# Function to install Ansible Galaxy collections
install_galaxy_requirements() {
    print_status "Installing Ansible Galaxy collections..."
    
    if [ -f "requirements.yml" ]; then
        ansible-galaxy collection install -r requirements.yml --force
        print_success "Ansible Galaxy requirements installed"
    else
        print_warning "requirements.yml not found, skipping Galaxy installation"
    fi
}

# Function to run the playbook
run_playbook() {
    local environment=${1:-development}
    local extra_vars=""
    local playbook_file="playbook.yml"
    
    print_status "Running Ansible playbook for environment: $environment"
    
    # Check if custom variables file exists
    if [ -f "vars/${environment}.yml" ]; then
        extra_vars="--extra-vars @vars/${environment}.yml"
    fi
    
    # Run the playbook
    ansible-playbook \
        --inventory inventory.yml \
        --limit $environment \
        $extra_vars \
        $playbook_file
    
    if [ $? -eq 0 ]; then
        print_success "Ansible playbook completed successfully!"
    else
        print_error "Ansible playbook failed!"
        exit 1
    fi
}

# Function to run Docker deployment
run_docker_deployment() {
    print_status "Running Docker-based deployment..."
    
    # Check if Docker is installed
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first:"
        echo "  Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    # Run the Docker playbook
    ansible-playbook \
        --inventory inventory.yml \
        --limit development \
        docker_playbook.yml
    
    if [ $? -eq 0 ]; then
        print_success "Docker deployment completed successfully!"
        print_status "Services are starting up, please wait 30-60 seconds..."
        print_status "Then access the services at:"
        echo "  - Trading Bot API: http://localhost:5050"
        echo "  - Grafana Monitoring: http://localhost:3000 (admin/admin)"
        echo "  - Prometheus Metrics: http://localhost:9090"
    else
        print_error "Docker deployment failed!"
        exit 1
    fi
}

# Function to test the connection
test_connection() {
    local environment=${1:-development}
    
    print_status "Testing connection to $environment hosts..."
    ansible all --inventory inventory.yml --limit $environment -m ping
    
    if [ $? -eq 0 ]; then
        print_success "Connection test passed"
    else
        print_warning "Connection test failed, but continuing..."
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [ENVIRONMENT]"
    echo ""
    echo "ENVIRONMENT:"
    echo "  development    Install on localhost (default)"
    echo "  staging        Install on staging servers"
    echo "  production     Install on production servers"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help     Show this help message"
    echo "  -t, --test     Test connection only"
    echo "  -v, --verbose  Verbose output"
    echo "  --check        Run in check mode (dry run)"
    echo "  --skip-galaxy  Skip Galaxy collection installation (preinstalled only)"
    echo "  --docker       Use Docker deployment (recommended)"
    echo ""
    echo "Examples:"
    echo "  $0 --docker                 # Docker deployment (recommended)"
    echo "  $0                          # Install on localhost"
    echo "  $0 development              # Install on development environment"
    echo "  $0 production --check       # Dry run on production"
    echo "  $0 --test staging           # Test connection to staging"
}

# Main execution
main() {
    local environment="development"
    local test_only=false
    local verbose=false
    local check_mode=false
    local skip_galaxy=false
    local use_docker=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -t|--test)
                test_only=true
                shift
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            --check)
                check_mode=true
                shift
                ;;
            --skip-galaxy)
                skip_galaxy=true
                shift
                ;;
            --docker)
                use_docker=true
                shift
                ;;
            development|staging|production)
                environment=$1
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Change to ansible directory
    cd "$(dirname "$0")"
    
    print_status "Starting ELVIS Trading Bot setup..."
    print_status "Environment: $environment"
    print_warning "Legacy compatibility automation only; not a V2 production cut-over."
    
    # Check requirements
    check_requirements
    
    # Install Galaxy requirements unless skipped
    if [ "$skip_galaxy" = false ]; then
        install_galaxy_requirements
    fi
    
    # Test connection
    if [ "$test_only" = true ]; then
        test_connection $environment
        exit 0
    fi
    
    # Run the appropriate deployment
    if [ "$use_docker" = true ]; then
        run_docker_deployment
    elif [ "$check_mode" = true ]; then
        print_warning "Running in check mode (dry run)"
        ansible-playbook --check --inventory inventory.yml --limit $environment playbook.yml
    else
        run_playbook $environment
    fi
    
    print_success "Setup completed!"
    echo ""
    echo "🔐 ELVIS Trading Bot with Enterprise Security Installed!"
    echo ""
    echo "📋 Next steps:"
    echo "1. 🔑 Setup HashiCorp Vault secrets (REQUIRED):"
    echo "   cd .. && ./setup_vault.sh"
    echo ""
    echo "2. 🔐 Configure your real API keys in Vault:"
    echo "   vault kv put secret/trading/api-keys \\"
    echo "     binance-api-key=YOUR_REAL_API_KEY \\"
    echo "     binance-api-secret=YOUR_REAL_API_SECRET"
    echo ""
    echo "3. 🚀 Start ELVIS with enterprise security:"
    echo "   export VAULT_ADDR=http://127.0.0.1:8200"
    echo "   export VAULT_TOKEN=trading-bot-token"
    echo "   python main.py --mode dashboard"
    echo ""
    echo "4. ✅ Verify security status:"
    echo "   Dashboard should show: ✅ Vault 3ms (connected)"
    echo "   Overall system health: 88%+"
    echo ""
    echo "📊 Additional services:"
    echo "- Monitor with Grafana: http://localhost:3000 (admin/admin)"
    echo "- Check API documentation: http://localhost:5050/api/docs"
    echo "- Prometheus metrics: http://localhost:9090"
    echo ""
    echo "📚 Security documentation:"
    echo "- Complete guide: ../SECURITY.md"
    echo "- Setup instructions: ../docs/VAULT_SETUP.md"
    echo ""
    echo "🎯 Features enabled:"
    echo "✅ Enterprise-grade secret management"
    echo "✅ AES-256-GCM encryption"  
    echo "✅ Real-time security monitoring"
    echo "✅ Maximum speed trading (zero cooldowns)"
    echo "✅ Comprehensive API health dashboard"
}

# Run main function
main "$@"
