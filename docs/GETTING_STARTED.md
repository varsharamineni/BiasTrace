# Getting Started with UV

UV is now set up for your project! Here's everything you need to know:

## 🎯 Quick Commands

### First Time Setup
```bash
# Run the setup script
./setup_uv.sh

# Or manually
make uv-sync
```

### Run Tests
```bash
# Run the LLM judge tests
make uv-test

# Or directly
uv run python tests/test_llm_judge_script.py
```

### Run the LLM Judge Script
```bash
# With UV (recommended)
uv run python reasoning_eval/llm_judge_script.py \
  --model "your-model-path" \
  --prompt_path "path/to/optimized_prompt.json" \
  --data_path "reasoning_eval/data_to_label/sample_traces_inital.json" \
  --max_samples 5 \
  --device "0"

# Or use Make
make uv-run CMD="python reasoning_eval/llm_judge_script.py --help"
```

## 📚 Documentation Files

- **`UV_GUIDE.md`** - Complete UV usage guide
- **`tests/QUICKSTART.md`** - Quick start for running tests
- **`tests/TEST_README.md`** - Detailed testing documentation
- **`Makefile`** - All available make commands

## 🛠️ Available Make Commands

### UV Commands (Fast!)
```bash
make uv-sync        # Sync dependencies
make uv-test        # Run tests
make uv-run         # Run custom command
```

### Traditional Commands
```bash
make test           # Run all tests
make test-judge     # Run LLM judge tests only
make lint           # Run linters
make lint-fix       # Auto-fix linting issues
make check          # Run linting + tests
```

## 📁 Project Structure

```
bias-reasoning-LLM/
├── .python-version              # Python 3.12.3
├── setup_uv.sh                  # UV setup script
├── UV_GUIDE.md                  # Complete UV guide
├── GETTING_STARTED.md           # This file
├── Makefile                     # Make commands
├── reasoning_eval/
│   ├── llm_judge_script.py     # Main script (refactored with DSPy)
│   └── ...
└── tests/
    ├── test_llm_judge_script.py # Test suite
    ├── test_data_sample.json    # Sample data
    ├── test_optimized_prompt.json # Sample prompt
    ├── QUICKSTART.md            # Quick test guide
    └── TEST_README.md           # Detailed test docs
```

## 🚀 Typical Workflow

1. **Setup** (first time only)
   ```bash
   ./setup_uv.sh
   ```

2. **Run Tests** (verify everything works)
   ```bash
   make uv-test
   ```

3. **Run Script** (with your data)
   ```bash
   uv run python reasoning_eval/llm_judge_script.py \
     --model "Qwen/Qwen3-4B" \
     --prompt_path "path/to/optimized_prompt.json" \
     --data_path "your_data.json" \
     --max_samples 10 \
     --device "0"
   ```

4. **Check Results**
   ```bash
   ls -lh reasoning_eval/llm_judge_samples/
   ```

## ✨ What's New with DSPy

The `llm_judge_script.py` has been refactored to use:

✅ **DSPy ChainOfThought** - Declarative prompting framework  
✅ **Optimized Signatures** - Load your DSPy-optimized prompts  
✅ **Structured Outputs** - Binary flags + reasoning extraction  
✅ **Type Safety** - Better error handling and validation

### Script Arguments

```bash
--model              # Model path (required)
--prompt_path        # DSPy optimized prompt JSON (required)
--data_path          # Input data JSON
--output_dir         # Output directory
--device             # CUDA device (e.g., "0" or "0,1")
--max_samples        # Limit samples for testing
--temperature        # Sampling temperature (default: 0.6)
--top_p              # Nucleus sampling (default: 0.95)
--top_k              # Top-k sampling (default: 20)
--seed               # Random seed (default: 42)
```

## 🔧 Troubleshooting

### UV not found?
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Dependencies not installing?
```bash
make uv-sync --reinstall
```

### Tests failing?
```bash
# Check imports
uv run python -c "import dspy; print('DSPy OK')"

# Run with verbose output
uv run python tests/test_llm_judge_script.py -v
```

## 📖 Learn More

- Run `make help` to see all commands
- Read `UV_GUIDE.md` for detailed UV usage
- Check `tests/TEST_README.md` for testing details

---

**Ready to start?** Run: `./setup_uv.sh && make uv-test` 🎉

