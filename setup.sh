#!/bin/bash
echo "🔧 Starting Reasoning Pattern Extraction environment setup..."

# Load compatible Python module (for HPC-style systems)
module purge
module load profile/base
module load python/3.10.8--gcc--8.5.0

# Define virtual environment path
ENV_DIR=~/envs/reasoning_pattern_env

# Create virtual environment if it doesn't exist
if [ ! -d "$ENV_DIR" ]; then
    echo "📦 Creating virtual environment at $ENV_DIR"
    python -m venv "$ENV_DIR"
else
    echo "✅ Virtual environment already exists at $ENV_DIR"
fi

# Activate virtual environment
source "$ENV_DIR/bin/activate"

# Upgrade pip and install dependencies
pip install --upgrade pip

# Install dependencies (create requirements.txt if missing)
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "⚠️ No requirements.txt found! Installing defaults..."
    pip install sentence-transformers
fi

echo "✅ Reasoning Pattern Extraction environment is ready!"
echo "To activate it later: source $ENV_DIR/bin/activate"