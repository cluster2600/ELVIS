#!/bin/bash

# ELVIS Trading Bot - Test Runner Script
# This script runs the comprehensive test suite with coverage reporting

set -e  # Exit on error

echo "🧪 Running ELVIS Trading Bot Test Suite..."
echo "========================================="

# Activate virtual environment
source venv310/bin/activate

# Install test dependencies if needed
echo "📦 Checking test dependencies..."
# Dependencies are installed via requirements.txt

# Clean up any previous coverage data
rm -f .coverage
rm -rf htmlcov/
rm -f coverage.xml

# Run tests with coverage
echo ""
echo "🔍 Running unit tests with coverage..."
echo "-------------------------------------"

echo "🐍 Python executable: $(which python)"
./venv310/bin/python -m pytest tests/ \
    --verbose \
    --cov=. \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=xml \
    --cov-config=.coveragerc \
    -W ignore::DeprecationWarning \
    --tb=short \
    --maxfail=5 \
    --disable-warnings \
    "$@"

# Check coverage threshold
echo ""
echo "📊 Coverage Summary:"
echo "-------------------"
coverage report --skip-covered --skip-empty --precision=2

# Generate coverage badge (if coverage-badge is installed)
if command -v coverage-badge &> /dev/null; then
    echo ""
    echo "🏷️  Generating coverage badge..."
    coverage-badge -o coverage.svg -f
fi

echo ""
echo "✅ Test suite completed!"
echo ""
echo "📋 Reports generated:"
echo "  - Terminal report: See above"
echo "  - HTML report: htmlcov/index.html"
echo "  - XML report: coverage.xml"
echo ""

# Open HTML coverage report if on macOS
if [[ "$OSTYPE" == "darwin"* ]] && [[ -f "htmlcov/index.html" ]]; then
    echo "📱 Opening HTML coverage report..."
    open htmlcov/index.html
fi

# Exit with pytest's exit code
exit ${PIPESTATUS[0]}
