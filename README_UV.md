# UV + Dual Environment Setup

## 🎯 Quick Start

### On macOS (Testing)
```bash
./setup_uv.sh        # Uses requirements-test.txt
make uv-test         # Run tests
```

### On Linux GPU Server (Production)
```bash
./setup_uv.sh --full # Uses requirements.txt
.venv/bin/python reasoning_eval/llm_judge_script.py --model "..." --prompt_path "..."
```

## 📦 Two Requirements Files

| File | Environment | Includes GPU Deps? | Use Case |
|------|-------------|-------------------|----------|
| `requirements-test.txt` | macOS | ❌ No | Testing, linting, development |
| `requirements.txt` | Linux GPU | ✅ Yes | Model inference, production |

## ✅ What This Means

- Your **`requirements.txt` is UNCHANGED** - keeps bitsandbytes, vllm, torch, etc.
- New **`requirements-test.txt`** - lightweight subset for macOS testing
- Both files coexist peacefully in version control
- Choose which one to use based on environment

## 🚀 Workflow

1. **Develop on macOS** → Use test requirements
2. **Test locally** → Mock-based tests work perfectly
3. **Push to git** → Both requirement files committed
4. **Deploy on Linux** → Use full requirements
5. **Run inference** → All GPU dependencies available

## 📚 Documentation

- **`DEPLOYMENT_GUIDE.md`** - Complete macOS dev + Linux production workflow
- **`MACOS_FIX.md`** - Why this setup exists and how it works
- **`UV_GUIDE.md`** - How to use UV package manager
- **`GETTING_STARTED.md`** - General getting started guide

## 🎨 Make Commands

```bash
make uv-test         # Run tests (uses test requirements)
make uv-sync         # Install test dependencies
make uv-sync-full    # Install full dependencies (Linux)
make lint            # Run linters
make check           # Tests + linting
```

## 💡 Why Two Files?

**Problem:** GPU dependencies don't work on macOS  
**Solution:** Separate test/dev dependencies from production dependencies  
**Benefit:** Fast testing on macOS, full power on Linux  

## 🔧 Commands Reference

### macOS Development
```bash
# Setup
./setup_uv.sh

# Test
make uv-test

# Lint
make lint

# Run tests directly
.venv/bin/python tests/test_llm_judge_script.py
```

### Linux Production
```bash
# Setup
./setup_uv.sh --full

# Run inference
.venv/bin/python reasoning_eval/llm_judge_script.py \
  --model "Qwen/Qwen3-4B" \
  --prompt_path "optimized_prompt.json" \
  --data_path "data.json" \
  --device "0,1"
```

## 📖 Read More

See `DEPLOYMENT_GUIDE.md` for the complete workflow!

