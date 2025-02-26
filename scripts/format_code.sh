#!/bin/bash

# Exit on error
set -e

echo "Formatting Python code..."

# Run black formatter
echo "Running black..."
black franka_sim/ tests/

# Run isort
echo "Running isort..."
isort franka_sim/ tests/

echo "Code formatting complete!"
echo "Note: Some flake8 issues may still need manual fixing:"
echo "  - Unused imports (F401)"
echo "  - Unused variables (F841)"
echo "  - Bare except clauses (E722)"
echo ""
echo "Run 'pre-commit run --all-files' to check remaining issues."
