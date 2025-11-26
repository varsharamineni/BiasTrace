# Deployment Guide - macOS Dev + Linux Production

This guide covers the typical workflow: develop/test on macOS, deploy on Linux GPU server.

## 📋 Two Requirements Files

### `requirements-test.txt` - For macOS Development
- Lightweight dependencies
- No GPU/CUDA libraries
- Perfect for testing and linting
- **Use this on your macOS machine**

### `requirements.txt` - For Linux GPU Server
- Complete dependencies
- Includes bitsandbytes, vllm, torch, CUDA
- **Use this on your Linux server for production**

## 🍎 Local Development (macOS)

### Initial Setup
```bash
# On your macOS machine
cd /Users/ksevegnani/Desktop/exps/bias-reasoning-LLM

# Setup test environment (fast, no GPU deps)
./setup_uv.sh

# Or manually
uv venv
uv pip install -r requirements-test.txt
```

### Run Tests Locally
```bash
# Run tests
make uv-test

# Run linting
make lint

# Both
make check
```

### What Works on macOS
✅ All unit tests  
✅ Script validation (imports, syntax)  
✅ Code linting and formatting  
✅ Data loading and output parsing tests  

### What Doesn't Work on macOS
❌ Actual model inference (needs GPU)  
❌ vLLM loading  
❌ Bitsandbytes quantization  

**But that's OK!** Tests use mocks, so you can verify everything works before deploying.

## 🐧 Production Deployment (Linux GPU Server)

### On Your Linux Server

```bash
# SSH into your Linux server
ssh user@your-gpu-server

# Clone or sync your code
cd /path/to/bias-reasoning-LLM

# Setup with FULL dependencies
./setup_uv.sh --full

# Or manually
uv venv
uv pip install -r requirements.txt  # Full deps with GPU support
```

### Run the Script on Linux
```bash
# With venv directly (recommended)
.venv/bin/python reasoning_eval/llm_judge_script.py \
  --model "Qwen/Qwen3-4B" \
  --prompt_path "path/to/optimized_prompt.json" \
  --data_path "reasoning_eval/data_to_label/sample_traces_inital.json" \
  --output_dir "reasoning_eval/llm_judge_samples/" \
  --device "0,1" \
  --temperature 0.6 \
  --top_p 0.95 \
  --seed 42

# Or activate venv and run
source .venv/bin/activate
python reasoning_eval/llm_judge_script.py --model "..." --prompt_path "..."
deactivate
```

## 🔄 Typical Workflow

### 1. Develop on macOS
```bash
# Edit code locally
vim reasoning_eval/llm_judge_script.py

# Test locally
make uv-test

# Lint code
make lint
```

### 2. Commit and Push
```bash
git add .
git commit -m "Update llm judge script"
git push
```

### 3. Deploy to Linux Server
```bash
# On Linux server
git pull

# If first time or dependencies changed
./setup_uv.sh --full

# Run the script
uv run python reasoning_eval/llm_judge_script.py \
  --model "your-model" \
  --prompt_path "your-prompt.json" \
  --data_path "your-data.json"
```

## 🎯 Quick Reference

| Environment | Requirements File | Setup Command | Use Case |
|-------------|------------------|---------------|----------|
| **macOS (dev)** | `requirements-test.txt` | `./setup_uv.sh` | Testing, linting |
| **Linux (prod)** | `requirements.txt` | `./setup_uv.sh --full` | Model inference |

## 📝 Both Files Are Kept

- ✅ `requirements.txt` - **UNCHANGED** - Full deps for Linux
- ✅ `requirements-test.txt` - **NEW** - Subset for macOS testing

## 💡 Pro Tips

### On macOS
```bash
# Fast test cycle
make uv-test && make lint

# Test specific function
.venv/bin/python -c "from reasoning_eval.llm_judge_script import load_reasoning_data; print(load_reasoning_data('tests/test_data_sample.json'))"
```

### On Linux Server
```bash
# Check GPU availability
nvidia-smi

# Run with GPU monitoring
watch -n 1 nvidia-smi  # In one terminal
.venv/bin/python reasoning_eval/llm_judge_script.py ...  # In another

# Run with specific GPUs
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python reasoning_eval/llm_judge_script.py --device "0,1" ...
```

## 🔧 Troubleshooting

### On macOS: "No module named X"
```bash
# Make sure you're using test requirements
cat .venv/pyvenv.cfg
uv pip list | grep dspy

# Reinstall if needed
rm -rf .venv
./setup_uv.sh
```

### On Linux: "CUDA not available"
```bash
# Check CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with full dependencies
rm -rf .venv
./setup_uv.sh --full
```

### Dependency Conflicts
If you update dependencies:

**macOS:**
```bash
# Update test requirements as needed
vim requirements-test.txt
rm -rf .venv && ./setup_uv.sh
```

**Linux:**
```bash
# Update full requirements
vim requirements.txt
rm -rf .venv && ./setup_uv.sh --full
```

## 📦 CI/CD Integration

If you have CI/CD:

```yaml
# .github/workflows/test.yml (example)
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install UV
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install test dependencies
        run: |
          uv venv
          uv pip install -r requirements-test.txt
      - name: Run tests
        run: uv run python tests/test_llm_judge_script.py
```

## 🎉 Summary

- **Develop on macOS**: Use `requirements-test.txt`, fast testing
- **Deploy on Linux**: Use `requirements.txt`, full GPU support
- **Both files coexist**: No conflicts, clear separation
- **Version control**: Both files committed to git

Your `requirements.txt` stays exactly as it is for your Linux server! 🚀

