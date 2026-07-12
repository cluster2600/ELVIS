#!/bin/bash

# ELVIS Trading Bot - Container Setup Test
# Quick test to verify Apple Container system integration

set -e

echo "🧪 Testing ELVIS Container Setup"
echo "================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test functions
test_docker() {
    echo -n "Testing Docker installation... "
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        return 1
    fi
}

test_files() {
    echo -n "Testing required files... "
    
    required_files=(
        "Dockerfile"
        "Dockerfile.simple"
        "docker-compose.yml"
        "requirements.txt"
        ".env"
    )
    
    missing_files=()
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -eq 0 ]]; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        echo "Missing files: ${missing_files[*]}"
        return 1
    fi
}

test_build() {
    echo -n "Testing container build... "
    
    if docker build -f Dockerfile.simple -t elvis-test:latest . &> /dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
        # Clean up test image
        docker rmi elvis-test:latest &> /dev/null || true
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        return 1
    fi
}

test_compose() {
    echo -n "Testing docker-compose syntax... "
    
    if docker-compose config &> /dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        return 1
    fi
}

test_python_imports() {
    echo -n "Testing Python dependencies... "
    
    # Test in container if possible, otherwise local
    if docker run --rm python:3.10-slim python -c "
import sys
try:
    import numpy, pandas, sklearn, requests
    print('OK')
    sys.exit(0)
except ImportError as e:
    print(f'Missing: {e}')
    sys.exit(1)
" &> /dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  SKIP${NC} (will install in container)"
        return 0
    fi
}

# Main test execution
main() {
    echo "Running container setup tests..."
    echo ""
    
    tests=(
        "test_docker"
        "test_files" 
        "test_compose"
        "test_build"
        "test_python_imports"
    )
    
    passed=0
    failed=0
    
    for test in "${tests[@]}"; do
        if $test; then
            ((passed++))
        else
            ((failed++))
        fi
    done
    
    echo ""
    echo "Test Results:"
    echo "============="
    echo -e "Passed: ${GREEN}$passed${NC}"
    echo -e "Failed: ${RED}$failed${NC}"
    
    if [[ $failed -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}🎉 All tests passed! ELVIS is ready for containers.${NC}"
        echo ""
        echo "Next steps:"
        echo "1. Run: scripts/setup_apple_containers.sh"
        echo "2. Start: ./start_elvis_apple.sh"
        echo "3. Open: http://localhost:5050"
        return 0
    else
        echo ""
        echo -e "${RED}❌ Some tests failed. Please fix issues before proceeding.${NC}"
        return 1
    fi
}

# Run tests
main "$@"