#!/bin/bash
# Setup script for UV package manager

set -e  # Exit on error

echo "🚀 Setting up UV for bias-reasoning-LLM..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed!"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ UV is installed: $(uv --version)"

# Check Python version
echo "📍 Python version: $(python3 --version)"

# Check if .python-version exists
if [ -f ".python-version" ]; then
    echo "✅ .python-version file exists"
else
    echo "⚠️  Creating .python-version file..."
    echo "3.12.3" > .python-version
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    uv venv
else
    echo "✅ Virtual environment already exists"
fi

# Check for --full flag
INSTALL_FULL=false
if [ "$1" = "--full" ]; then
    INSTALL_FULL=true
fi

# Sync dependencies
if [ "$INSTALL_FULL" = true ]; then
    echo "📦 Installing FULL dependencies (includes GPU/model deps)..."
    echo "⚠️  This may take a while and requires GPU support..."
    if [ -f "requirements.txt" ]; then
        uv pip install -r requirements.txt
    else
        echo "❌ requirements.txt not found!"
        exit 1
    fi
else
    echo "📦 Installing test dependencies (lightweight, no GPU)..."
    if [ -f "requirements-test.txt" ]; then
        uv pip install -r requirements-test.txt
    else
        echo "❌ requirements-test.txt not found!"
        exit 1
    fi
    echo ""
    echo "💡 Tip: To install full dependencies (for running models):"
    echo "   ./setup_uv.sh --full"
fi

# Show virtual environment info
if [ -d ".venv" ]; then
    echo "✅ Virtual environment created at: .venv/"
    echo "   Python: $(ls -la .venv/bin/python* 2>/dev/null | head -1 || echo 'Not found')"
fi

echo ""
echo "✨ Setup complete! You can now:"
echo "   • Run tests:  make uv-test"
if [ "$INSTALL_FULL" = true ]; then
    echo "   • Run script: uv run python reasoning_eval/llm_judge_script.py --help"
else
    echo "   • Install full deps for running models: ./setup_uv.sh --full"
fi
echo "   • See guide:  cat UV_GUIDE.md"
echo ""

