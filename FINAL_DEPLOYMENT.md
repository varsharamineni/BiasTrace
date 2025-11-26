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

### 4. Run Small Test (5 samples)
```bash
.venv/bin/python reasoning_eval/llm_judge_script.py \
  --model "Qwen/Qwen3-4B" \
  --prompt_path "tests/judge_optimized_prompt.json" \
  --data_path "reasoning_eval/data_to_label/sample_traces_inital.json" \
  --output_dir "reasoning_eval/llm_judge_samples/" \
  --device "0" \
  --max_samples 5 \
  --temperature 0.6 \
  --seed 42
```

### 5. Run Full Dataset
```bash
# Remove --max_samples to run on all data
.venv/bin/python reasoning_eval/llm_judge_script.py \
  --model "Qwen/Qwen3-4B" \
  --prompt_path "tests/judge_optimized_prompt.json" \
  --data_path "reasoning_eval/data_to_label/sample_traces_inital.json" \
  --output_dir "reasoning_eval/llm_judge_samples/" \
  --device "0" \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --seed 42
```

### 6. Check Results
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

## 📊 Quick Reference Card

| Task | Command |
|------|---------|
| **macOS: Run tests** | `make dry-run` |
| **macOS: Verify prompt** | `.venv/bin/python -c "from reasoning_eval.llm_judge_script import load_optimized_signature; load_optimized_signature('tests/judge_optimized_prompt.json')"` |
| **Server: Setup** | `./setup_server.sh` |
| **Server: Test small** | `.venv/bin/python reasoning_eval/llm_judge_script.py --model "..." --prompt_path "tests/judge_optimized_prompt.json" --max_samples 5` |
| **Server: Run full** | Same as above, remove `--max_samples` |

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

