# Final Deployment Guide 🚀

## ✅ What We Accomplished

### 1. **Refactored Script to DSPy**
- ✅ Script now uses `dspy.ChainOfThought`
- ✅ Loads DSPy-optimized prompts from JSON
- ✅ Structured output with 8 binary flags + reasoning
- ✅ Location: `reasoning_eval/llm_judge_script.py`

### 2. **Set Up UV Package Manager**
- ✅ Fast Python dependency management
- ✅ Separate environments for macOS dev and Linux prod
- ✅ No Poetry dependency - uses requirements.txt

### 3. **Created Test Suite**
- ✅ Unit tests: `tests/test_llm_judge_script.py`
- ✅ Dry-run test: `tests/test_dry_run.py` (no LLM needed)
- ✅ All passing ✓

### 4. **Added DSPy to Dependencies**
- ✅ Added to requirements.txt
- ✅ Your optimized prompt: `tests/judge_optimized_prompt.json`

### 5. **Cleaned Up Configuration**
- ✅ Removed Poetry sections from pyproject.toml
- ✅ Removed author attribution
- ✅ Minimal config for UV

## 📦 Files Structure

```
bias-reasoning-LLM/
├── reasoning_eval/
│   └── llm_judge_script.py          # ✨ Refactored with DSPy
├── tests/
│   ├── judge_optimized_prompt.json  # 🎯 Your optimized prompt
│   ├── test_llm_judge_script.py     # Unit tests
│   ├── test_dry_run.py              # Pre-deployment verification
│   ├── test_data_sample.json        # Sample data
│   └── QUICKSTART.md                # Quick test guide
├── requirements.txt                 # Full deps (with dspy-ai)
├── requirements-test.txt            # macOS testing only
├── requirements-server.txt          # Flexible server install
├── setup_uv.sh                      # macOS setup
├── setup_server.sh                  # Server setup (ARM64/x86_64)
├── pyproject.toml                   # Minimal (cleaned up)
└── Makefile                         # Quick commands
```

## 🍎 On Your macOS (Local Development)

### 1. Run Dry-Run Test
```bash
cd /Users/ksevegnani/Desktop/exps/bias-reasoning-LLM

# Run comprehensive verification (no LLM needed)
make dry-run
```

**Expected output:**
```
🎉 ALL TESTS PASSED!
✅ Ready to deploy to Linux server!
```

### 2. Commit and Push
```bash
git add .
git commit -m "Refactor to DSPy with optimized prompt"
git push
```

## 🐧 On Your Linux Server (ARM64 - Production)

### 1. Get the Code
```bash
# SSH to server
ssh user@your-server

# Clone or pull
cd /path/to/project
git pull
```

### 2. Setup with New Script
```bash
# Make setup script executable
chmod +x setup_server.sh

# Run setup (handles ARM64 automatically)
./setup_server.sh
```

**This will:**
- ✅ Detect ARM64 architecture
- ✅ Install core dependencies (torch, vllm, dspy-ai, transformers)
- ✅ Skip triton and bitsandbytes (ARM64 incompatible)
- ✅ Create .venv with all needed packages

### 3. Test on Server (Dry Run)
```bash
# Verify everything loads
.venv/bin/python tests/test_dry_run.py
```

### 4. Configure LM Settings

Edit `reasoning_eval/lm_config.py` to match your setup:

```python
# Model configuration
DEFAULT_MODEL = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"

# API configuration (adjust to your vLLM server)
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_API_KEY = "dummy"
```

See `reasoning_eval/LM_CONFIG_README.md` for detailed configuration guide.

### 5. Run Small Test (5 samples)

**Simple (uses defaults from lm_config.py):**
```bash
chmod +x run_judge_example.sh
./run_judge_example.sh
```

**Or run directly:**
```bash
.venv/bin/python reasoning_eval/llm_judge_script.py \
  --prompt_path "tests/judge_optimized_prompt.json" \
  --data_path "reasoning_eval/data_to_label/sample_traces_inital.json" \
  --max_samples 5
```

**With custom settings (override defaults):**
```bash
.venv/bin/python reasoning_eval/llm_judge_script.py \
  --prompt_path "tests/judge_optimized_prompt.json" \
  --data_path "reasoning_eval/data_to_label/sample_traces_inital.json" \
  --model "different-model" \
  --api_base "http://localhost:8001/v1" \
  --max_samples 5
```

### 6. Run Full Dataset

```bash
# Remove --max_samples to run on all data (uses settings from lm_config.py)
.venv/bin/python reasoning_eval/llm_judge_script.py \
  --prompt_path "tests/judge_optimized_prompt.json" \
  --data_path "reasoning_eval/data_to_label/sample_traces_inital.json"
```

### 7. Check Results
```bash
# List output files
ls -lh reasoning_eval/llm_judge_samples/

# View results (if you have jq)
cat reasoning_eval/llm_judge_samples/llm_eval_dspy_*.json | jq '.results[0]'

# Or view directly
cat reasoning_eval/llm_judge_samples/llm_eval_dspy_*.json
```

## 🎯 Expected Output Structure

```json
{
  "metadata": {
    "judge_model": "Qwen/Qwen3-4B",
    "framework": "dspy",
    "module": "ChainOfThought",
    "sampling_params": {...}
  },
  "results": [
    {
      "sample_id": "...",
      "category": "Age",
      "judge_reasoning": "The model's reasoning...",
      "judge_output": {
        "reasoning": "Detailed analysis...",
        "group_assumption": 0,
        "bias_acknowledgement": 1,
        "meta_reflection": 0,
        "outside_demo_knowledge": 0,
        "outside_topical_knowledge": 0,
        "unresolved": 0,
        "overthinking": 0,
        "missing_logic": 0
      }
    },
    ...
  ]
}
```

## 🔧 Troubleshooting

### On macOS: "Module not found"
```bash
rm -rf .venv
./setup_uv.sh
make dry-run
```

### On Server: "Package not compatible"
```bash
# ARM64 server - this is expected
# setup_server.sh automatically handles it
./setup_server.sh
```

### CUDA Issues
```bash
# Check GPU
nvidia-smi

# Check CUDA in Python
.venv/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🔧 Configuration with lm_config.py

All LM settings are now centralized in `reasoning_eval/lm_config.py`:

```python
# Edit these defaults to match your setup
DEFAULT_MODEL = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_API_KEY = "dummy"
DEFAULT_TIMEOUT = 600.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.7
```

**Benefits:**
- ✅ **Set once, use everywhere** - No need to specify API settings every time
- ✅ **Easy to maintain** - One file to update
- ✅ **Can still override** - Use CLI args when needed
- ✅ **Version controlled** - Team shares same defaults

See `reasoning_eval/LM_CONFIG_README.md` for full documentation.

## 📊 Quick Reference Card

| Task | Command |
|------|---------|
| **macOS: Run tests** | `make dry-run` |
| **Server: Setup** | `./setup_server.sh` |
| **Server: Configure** | Edit `reasoning_eval/lm_config.py` |
| **Server: Test (5 samples)** | `./run_judge_example.sh` |
| **Server: Run full** | `.venv/bin/python reasoning_eval/llm_judge_script.py --prompt_path "tests/judge_optimized_prompt.json"` |
| **Override settings** | Add `--model "..." --temperature 0.5` etc. to any command |

## ✨ Key Features

1. **DSPy Integration**: Uses your optimized prompt directly
2. **ARM64 Compatible**: Works on your aarch64 server
3. **Structured Outputs**: 8 binary flags + detailed reasoning
4. **Tested & Ready**: All tests passing locally
5. **Clean Config**: No Poetry baggage

## 🎉 You're Ready!

Everything is set up and tested. Just:
1. Run `make dry-run` on macOS
2. Push to git
3. Pull on server
4. Run `./setup_server.sh`
5. Execute the script!

Good luck with your evaluation! 🚀

