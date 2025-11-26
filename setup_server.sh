#!/bin/bash
# Setup script for Linux GPU servers (handles ARM64 and x86_64)

set -e  # Exit on error

echo "🚀 Setting up for Linux GPU server..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed!"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ UV is installed: $(uv --version)"

# Check Python version
echo "📍 Python version: $(python3 --version)"

# Detect architecture
ARCH=$(uname -m)
echo "🖥️  Architecture: $ARCH"

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

# Install based on architecture
echo "📦 Installing server dependencies..."

if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo "⚠️  ARM64 detected - using flexible installation"
    echo "   (triton and bitsandbytes may be skipped)"
    
    # Install core dependencies first
    uv pip install torch transformers vllm dspy-ai tqdm numpy pandas accelerate
    
    echo ""
    echo "✅ Core dependencies installed!"
    echo "⚠️  Note: Some packages (triton, bitsandbytes) skipped on ARM64"
    
elif [ "$ARCH" = "x86_64" ]; then
    echo "✅ x86_64 detected - using full installation"
    
    if [ -f "requirements.txt" ]; then
        echo "   Using requirements.txt..."
        uv pip install -r requirements.txt
    elif [ -f "requirements-server.txt" ]; then
        echo "   Using requirements-server.txt..."
        uv pip install -r requirements-server.txt
    else
        echo "❌ No requirements file found!"
        exit 1
    fi
else
    echo "❌ Unsupported architecture: $ARCH"
    exit 1
fi

# Show virtual environment info
if [ -d ".venv" ]; then
    echo ""
    echo "✅ Virtual environment ready at: .venv/"
    echo "   Python: $(ls -la .venv/bin/python* 2>/dev/null | head -1 || echo 'Not found')"
fi

echo ""
echo "✨ Setup complete! You can now:"
echo "   • Test: .venv/bin/python tests/test_dry_run.py"
echo "   • Run: .venv/bin/python reasoning_eval/llm_judge_script.py --help"
echo ""
echo "📝 To run inference:"
echo "   .venv/bin/python reasoning_eval/llm_judge_script.py \\"
echo "     --model 'your-model' \\"
echo "     --prompt_path 'tests/judge_optimized_prompt.json' \\"
echo "     --data_path 'your-data.json' \\"
echo "     --device '0'"
echo ""

