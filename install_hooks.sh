#!/bin/bash

# Exit on error
set -e

echo "Installing franka-sim package with development dependencies..."

# Try to uninstall existing package first to avoid conflicts
pip uninstall -y franka-sim 2>/dev/null || true

# Clean up any build artifacts
rm -rf build/ *.egg-info/ dist/

# Install package in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit uninstall 2>/dev/null || true  # Remove any existing hooks first
pre-commit install

echo "Pre-commit hooks installed successfully!"
echo "The following checks will run on commit:"
echo "  - black (code formatting)"
echo "  - isort (import sorting)"
echo "  - flake8 (code linting)"
echo "  - pytest (unit tests)"
echo ""
echo "You can run the hooks manually using:"
echo "  - Run all hooks on all files:        pre-commit run --all-files"
echo "  - Run all hooks on staged files:     pre-commit run"
echo "  - Run specific hook (e.g., black):   pre-commit run black --all-files"
echo "  - Run on specific files:             pre-commit run --files path/to/file1 path/to/file2"
