# macOS Dev Setup - Testing Without GPU Dependencies 🍎

## The Situation

You're developing/testing on macOS but will run the actual model inference on a Linux GPU server.

`bitsandbytes` and other GPU dependencies don't have wheels for macOS, so installing the full `requirements.txt` fails locally.

## The Solution ✅

**Two requirements files for two environments:**

- **`requirements-test.txt`** → macOS development/testing (you are here)
- **`requirements.txt`** → Linux GPU server (unchanged, kept for production)

I've created **two separate installation modes**:

### 1. Test Mode (Lightweight) - For Testing Only
- ✅ No GPU dependencies
- ✅ Fast installation
- ✅ Perfect for running tests
- ✅ Works on macOS

### 2. Full Mode - For Running Models
- Includes GPU/CUDA dependencies
- Only works on Linux with NVIDIA GPUs
- Not needed for testing

## Quick Fix - Run This Now! 🚀

```bash
# Remove the partially installed venv
rm -rf .venv

# Run setup in TEST mode (default)
./setup_uv.sh

# Run tests
make uv-test
```

That's it! The tests will now run without needing GPU dependencies.

## Commands Overview

### Test Mode (macOS-friendly)
```bash
# Setup for testing
./setup_uv.sh                # Installs requirements-test.txt

# Run tests
make uv-test

# Sync test dependencies
make uv-sync
```

### Full Mode (Linux/GPU only)
```bash
# Setup for running models (requires Linux + GPU)
./setup_uv.sh --full        # Installs requirements.txt

# Sync full dependencies
make uv-sync-full
```

## What Gets Installed in Test Mode

The `requirements-test.txt` includes only what you need for testing:
- ✅ `dspy-ai` - For the DSPy framework
- ✅ `tqdm` - For progress bars
- ✅ `pytest` - For running tests
- ✅ `ruff`, `black`, `mypy` - For linting

**Excluded** (not needed for tests):
- ❌ `bitsandbytes` - GPU quantization
- ❌ `vllm` - Model inference
- ❌ `torch` - Deep learning framework
- ❌ CUDA libraries

## When Do You Need Full Mode?

You only need full dependencies when:
1. Running on a **Linux machine with NVIDIA GPU**
2. Actually running inference with models
3. Not just testing

For development and testing on macOS, test mode is perfect!

## Expected Output

```bash
$ rm -rf .venv
$ ./setup_uv.sh
🚀 Setting up UV for bias-reasoning-LLM...
✅ UV is installed: uv 0.7.8 (Homebrew 2025-05-23)
📍 Python version: Python 3.12.3
✅ .python-version file exists
📦 Creating virtual environment...
📦 Installing test dependencies (lightweight, no GPU)...
[... quick installation ...]
✅ Virtual environment created at: .venv/

✨ Setup complete! You can now:
   • Run tests:  make uv-test
   • Install full deps for running models: ./setup_uv.sh --full
   • See guide:  cat UV_GUIDE.md

$ make uv-test
🧪 Running tests with uv...
test_load_reasoning_data ... ok
test_load_optimized_signature ... ok
test_parse_dspy_outputs ... ok
test_save_results ... ok
test_full_pipeline_mock ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.XXXs

OK
✅ Tests complete!
```

## Why This Approach?

**Development/Testing (macOS)**
- Fast setup
- No GPU needed
- Mock testing works perfectly

**Production (Linux + GPU)**
- Full model inference capabilities
- GPU acceleration
- All dependencies included

## Still Having Issues?

Try this clean reinstall:

```bash
# 1. Clean everything
rm -rf .venv
make clean

# 2. Fresh install (test mode)
./setup_uv.sh

# 3. Verify
uv run python -c "import dspy; print('DSPy OK!')"

# 4. Run tests
make uv-test
```

## Summary

| What You Want | Command |
|---------------|---------|
| **Test the script** | `./setup_uv.sh` (default) |
| **Run on GPU server** | `./setup_uv.sh --full` |
| **Run tests** | `make uv-test` |
| **Clean install** | `rm -rf .venv && ./setup_uv.sh` |

---

**Ready to test?** Run:
```bash
rm -rf .venv
./setup_uv.sh
make uv-test
```

🎉 This will work on macOS!

