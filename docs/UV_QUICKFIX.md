# UV Quick Fix - Issues Resolved! ✅

## What Was Wrong

1. ❌ Virtual environment didn't exist
2. ❌ `pyproject.toml` missing `[project]` table

## What Was Fixed

✅ Updated `setup_uv.sh` to create venv first  
✅ Added `[project]` table to `pyproject.toml`  
✅ Updated Makefile to auto-create venv if missing  

## Try Again Now! 🚀

### Clean Start (Recommended)

```bash
# Remove old venv if it exists (optional cleanup)
rm -rf .venv

# Run setup script again
./setup_uv.sh

# Run tests
make uv-test
```

### Or Just Run Tests Directly

```bash
# This will auto-create venv and install dependencies
make uv-test
```

### Manual Step-by-Step (If Issues Persist)

```bash
# 1. Create virtual environment
uv venv

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Run tests
uv run python tests/test_llm_judge_script.py
```

## Verify Installation

```bash
# Check venv exists
ls -la .venv/

# Check Python in venv
.venv/bin/python --version

# Check uv can see it
uv run python --version
```

## Quick Test Commands

```bash
# Test 1: Just run the test
uv run python tests/test_llm_judge_script.py

# Test 2: Test with verbose output
uv run python tests/test_llm_judge_script.py -v

# Test 3: Check imports work
uv run python -c "import sys; print('Python:', sys.version)"
```

## What to Expect Now

```bash
$ ./setup_uv.sh
🚀 Setting up UV for bias-reasoning-LLM...
✅ UV is installed: uv 0.7.8 (Homebrew 2025-05-23)
📍 Python version: Python 3.12.3
✅ .python-version file exists
📦 Creating virtual environment...
📦 Installing dependencies...
   Using requirements.txt...
[... installation progress ...]
✅ Virtual environment created at: .venv/

$ make uv-test
🧪 Running tests with uv...
test_load_reasoning_data ... ok
test_load_optimized_signature ... ok
[... test output ...]
✅ Tests complete!
```

## Still Having Issues?

Try this debug sequence:

```bash
# 1. Check UV version
uv --version

# 2. Check Python version
python3 --version

# 3. Create fresh venv
rm -rf .venv
uv venv

# 4. Activate manually to test
source .venv/bin/activate
pip install -r requirements.txt
python tests/test_llm_judge_script.py
deactivate

# 5. Try with uv
uv run python tests/test_llm_judge_script.py
```

---

**Ready?** Run: `./setup_uv.sh` 🎉

