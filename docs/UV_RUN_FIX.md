# UV Run Fix - Package Build Issue Resolved ✅

## What Was The Problem?

`uv run` tries to install the current directory as a Python package when it sees `pyproject.toml`. But this project isn't structured as an installable package - it's a collection of scripts.

Error was:
```
poetry.core.masonry.utils.module.ModuleOrPackageNotFoundError: 
No file/folder found for package bias-reasoning-llm
```

## The Solution ✅

Instead of using `uv run python ...`, we now directly use the virtual environment's Python:
```bash
.venv/bin/python script.py
```

This bypasses the package building step entirely.

## What Changed

### Makefile Commands Updated

**Before:**
```makefile
uv-test:
    uv run python tests/test_llm_judge_script.py  # ❌ Tries to build package
```

**After:**
```makefile
uv-test:
    .venv/bin/python tests/test_llm_judge_script.py  # ✅ Uses venv directly
```

## Try Again Now! 🚀

```bash
# Should work now!
make uv-test
```

## Alternative Ways to Run

### Option 1: Use Make (Easiest)
```bash
make uv-test
```

### Option 2: Direct venv Python
```bash
.venv/bin/python tests/test_llm_judge_script.py
```

### Option 3: Activate venv (Traditional)
```bash
source .venv/bin/activate
python tests/test_llm_judge_script.py
deactivate
```

### Option 4: Use uv with --no-project flag
```bash
uv run --no-project python tests/test_llm_judge_script.py
```

## Running Your Script

### On macOS (Testing)
```bash
# Run the script (won't actually do inference without GPU)
.venv/bin/python reasoning_eval/llm_judge_script.py --help

# Or
make uv-run CMD="python reasoning_eval/llm_judge_script.py --help"
```

### On Linux (Production)
```bash
# After ./setup_uv.sh --full
.venv/bin/python reasoning_eval/llm_judge_script.py \
  --model "Qwen/Qwen3-4B" \
  --prompt_path "optimized_prompt.json" \
  --data_path "data.json"
```

## Why This Approach?

1. **Simpler**: No package building needed
2. **Faster**: Direct execution
3. **Flexible**: Works for script-based projects
4. **Compatible**: Still uses UV for dependency management

## Updated Commands

| Command | What It Does | How It Works |
|---------|-------------|--------------|
| `make uv-test` | Run tests | Uses `.venv/bin/python` |
| `make uv-run CMD="..."` | Run command | Uses `.venv/bin/...` |
| `make uv-sync` | Install deps | Uses `uv pip install` |

## Summary

- ✅ UV still manages dependencies (fast!)
- ✅ Tests run directly with venv Python (no build step)
- ✅ Works with script-based projects
- ✅ No changes needed to your code

**Ready to test?** Run: `make uv-test` 🎉

