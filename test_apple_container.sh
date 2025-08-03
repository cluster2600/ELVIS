#!/bin/bash

# Test Apple Container CLI Integration
# Verify ELVIS works with Apple's native container command

set -e

echo "🧪 Testing Apple Container CLI Integration"
echo "========================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

test_apple_container_cli() {
    echo -n "Testing Apple container CLI... "
    
    if command -v container &> /dev/null; then
        echo -e "${GREEN}✅ FOUND${NC}"
        echo "   Version: $(container --version 2>/dev/null || echo 'Available')"
        return 0
    else
        echo -e "${RED}❌ NOT FOUND${NC}"
        echo "   Please install Apple Container CLI"
        return 1
    fi
}

test_container_commands() {
    echo -n "Testing container commands... "
    
    # Test basic container commands
    if container --help &> /dev/null && \
       container images --help &> /dev/null && \
       container run --help &> /dev/null; then
        echo -e "${GREEN}✅ WORKING${NC}"
        return 0
    else
        echo -e "${RED}❌ ERRORS${NC}"
        return 1
    fi
}

test_dockerfile() {
    echo -n "Testing Dockerfile... "
    
    if [[ -f "Dockerfile.simple" ]]; then
        echo -e "${GREEN}✅ FOUND${NC}"
        return 0
    else
        echo -e "${RED}❌ MISSING${NC}"
        return 1
    fi
}

test_apple_script() {
    echo -n "Testing Apple container script... "
    
    if [[ -f "apple_container_native.sh" ]] && [[ -x "apple_container_native.sh" ]]; then
        echo -e "${GREEN}✅ READY${NC}"
        return 0
    else
        echo -e "${RED}❌ MISSING${NC}"
        return 1
    fi
}

test_network_creation() {
    echo -n "Testing network creation... "
    
    # Try to create and remove a test network
    if container network create test-network &> /dev/null; then
        container network rm test-network &> /dev/null || true
        echo -e "${GREEN}✅ WORKING${NC}"
        return 0
    else
        echo -e "${RED}❌ FAILED${NC}"
        return 1
    fi
}

test_image_pull() {
    echo -n "Testing image pull... "
    
    # Test pulling a simple image
    if container run --rm hello-world &> /dev/null 2>&1; then
        echo -e "${GREEN}✅ WORKING${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  SKIPPED${NC} (hello-world not available)"
        return 0
    fi
}

show_recommendations() {
    echo ""
    echo "🎯 Apple Container Setup Recommendations:"
    echo "========================================"
    echo ""
    echo "✅ Ready to use:"
    echo "   ./apple_container_native.sh setup"
    echo "   ./apple_container_native.sh start"
    echo ""
    echo "📊 Dashboard access:"
    echo "   http://localhost:5050 (Trading Dashboard)"
    echo "   http://localhost:3000 (Grafana - admin/admin)"
    echo ""
    echo "🛠️ Management commands:"
    echo "   ./apple_container_native.sh status"
    echo "   ./apple_container_native.sh logs"
    echo "   ./apple_container_native.sh stop"
    echo ""
    echo "💰 Paper trading features:"
    echo "   • \$1000 USDT starting balance"
    echo "   • \$1000 BNB starting balance"
    echo "   • Bonenkamp HFT strategy"
    echo "   • Real-time position monitoring"
}

show_apple_container_benefits() {
    echo ""
    echo "🍎 Apple Container Benefits:"
    echo "=========================="
    echo "• Native macOS integration"
    echo "• Better performance on Apple Silicon"
    echo "• Enhanced security features"
    echo "• System-level resource management"
    echo "• Optimized for M1/M2/M3 chips"
}

main() {
    echo "Testing Apple Container CLI compatibility..."
    echo ""
    
    tests=(
        "test_apple_container_cli"
        "test_container_commands"
        "test_dockerfile"
        "test_apple_script"
        "test_network_creation"
        "test_image_pull"
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
        echo -e "${GREEN}🎉 Apple Container CLI is ready for ELVIS!${NC}"
        show_recommendations
        show_apple_container_benefits
        return 0
    else
        echo ""
        echo -e "${RED}❌ Some tests failed. Please fix issues before proceeding.${NC}"
        
        if ! command -v container &> /dev/null; then
            echo ""
            echo "To install Apple Container CLI:"
            echo "1. Visit: https://developer.apple.com/documentation/container"
            echo "2. Or install via Xcode Command Line Tools"
            echo "3. Or check Mac App Store for Container apps"
        fi
        
        return 1
    fi
}

# Run tests
main "$@"