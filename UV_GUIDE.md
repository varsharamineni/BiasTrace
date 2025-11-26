# UV Guide - Fast Python Package Management

`uv` is a blazingly fast Python package installer and resolver, written in Rust. It's 10-100x faster than `pip`.

## Quick Start with UV

### 1. Initial Setup (First Time Only)

```bash
# Sync all dependencies from requirements.txt or pyproject.toml
uv sync
```

This creates a virtual environment and installs all dependencies.

### 2. Run Tests with UV

```bash
# Option 1: Using Make
make uv-test

# Option 2: Direct UV command
uv run python tests/test_llm_judge_script.py
```

### 3. Run the LLM Judge Script with UV

```bash
# Run the main script
uv run python reasoning_eval/llm_judge_script.py \
  --model "your-model-path" \
  --prompt_path "path/to/optimized_prompt.json" \
  --data_path "reasoning_eval/data_to_label/sample_traces_inital.json" \
  --max_samples 5 \
  --device "0"
```

## Common UV Commands

### Installation & Setup

```bash
# Install dependencies from requirements.txt
uv pip install -r requirements.txt

# Install a specific package
uv pip install dspy-ai

# Install package in editable mode
uv pip install -e .

# Sync from pyproject.toml
uv sync
```

### Running Scripts

```bash
# Run any Python script with the virtual environment
uv run python script.py

# Run with additional arguments
uv run python script.py --arg1 value1 --arg2 value2

# Run a module
uv run -m pytest tests/
```

### Using Make Commands

```bash
# Sync dependencies
make uv-sync

# Run tests
make uv-test

# Run custom command
make uv-run CMD="python your_script.py --args"
```

## Why Use UV?

✅ **Speed**: 10-100x faster than pip  
✅ **Reliability**: Better dependency resolution  
✅ **Modern**: Written in Rust, actively maintained  
✅ **Compatible**: Works with pip, poetry, requirements.txt  
✅ **Simple**: Drop-in replacement for many pip commands

## UV vs Traditional Python

| Traditional | UV Equivalent |
|------------|---------------|
| `python script.py` | `uv run python script.py` |
| `pip install package` | `uv pip install package` |
| `pip install -r requirements.txt` | `uv pip install -r requirements.txt` |
| `source venv/bin/activate` | Not needed! Use `uv run` |

## Project Structure with UV

```
bias-reasoning-LLM/
├── .python-version         # Python version for uv (3.12.3)
├── pyproject.toml         # Project dependencies (poetry format)
├── requirements.txt       # Alternative dependency list
├── uv.lock               # Lock file (created after uv sync)
├── .venv/                # Virtual environment (created by uv)
└── tests/                # Test files
```

## Troubleshooting

### UV command not found
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Dependencies not syncing
```bash
# Force reinstall
uv sync --reinstall

# Or clear cache
uv cache clean
```

### Python version issues
```bash
# Check Python version
cat .python-version

# Use specific Python version
uv run --python 3.12 python script.py
```

## Advanced Usage

### Running Tests with Options

```bash
# Run tests with verbose output
uv run python tests/test_llm_judge_script.py -v

# Run specific test class
uv run python -m pytest tests/test_llm_judge_script.py::TestLoadReasoningData
```

### Running LLM Judge with UV

```bash
# Full example with all options
uv run python reasoning_eval/llm_judge_script.py \
  --model "Qwen/Qwen3-4B" \
  --prompt_path "tests/test_optimized_prompt.json" \
  --data_path "tests/test_data_sample.json" \
  --output_dir "reasoning_eval/llm_judge_samples/" \
  --device "0" \
  --max_samples 2 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --seed 42
```

### Development Workflow

```bash
# 1. Sync dependencies (first time or after updating requirements)
make uv-sync

# 2. Run tests to verify everything works
make uv-test

# 3. Run your script
make uv-run CMD="python reasoning_eval/llm_judge_script.py --help"

# 4. Clean up
make clean
```

## Integration with IDEs

### VS Code / Cursor
UV creates a virtual environment at `.venv/`. Point your IDE to:
```
.venv/bin/python
```

### PyCharm
1. Settings → Project → Python Interpreter
2. Add Interpreter → Existing Environment
3. Select `.venv/bin/python`

## Resources

- UV Documentation: https://github.com/astral-sh/uv
- UV vs Pip: https://astral.sh/blog/uv
- Python Packaging: https://packaging.python.org/

---

**Ready to use UV?** Run: `make uv-sync` then `make uv-test`

