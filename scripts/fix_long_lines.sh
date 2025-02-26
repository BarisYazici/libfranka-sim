#!/bin/bash

# Exit on error
set -e

echo "Fixing long lines and formatting code..."

# Run black formatter with line length of 100
echo "Running black..."
black --line-length 100 franka_sim/ tests/

echo "Code formatting complete!"
echo "Note: If there are still flake8 errors about long lines, they may need manual fixing."
echo "These are typically lines that black cannot safely break, such as:"
echo "  - Long string literals"
echo "  - Comments"
echo "  - URLs"
echo ""
echo "Run 'pre-commit run flake8 --all-files' to check for remaining issues."
