# Quick Start - Run Tests Now! 🚀

All test files are ready in the `tests/` directory. Here's how to run them:

## 🚀 Using UV (Recommended - Fast!)

```bash
cd /Users/ksevegnani/Desktop/exps/bias-reasoning-LLM

# First time setup
./setup_uv.sh

# Run tests
make uv-test
```

## 🐍 Traditional Python

```bash
cd /Users/ksevegnani/Desktop/exps/bias-reasoning-LLM
python3 tests/test_llm_judge_script.py
```

This will run all unit tests with verbose output.

## 2. Using Make Commands

```bash
# With UV (fast)
make uv-test

# Traditional (after installing dependencies)
make test-judge

# Run tests + linting
make check
```

## 3. Test Files Location

```
tests/
├── __init__.py                    # Makes tests a Python package
├── test_llm_judge_script.py       # Main test suite
├── test_data_sample.json          # Sample test data
├── test_optimized_prompt.json     # Sample optimized prompt
├── TEST_README.md                 # Detailed testing guide
└── QUICKSTART.md                  # This file
```

## What Gets Tested

✅ **Data Loading**: Verifies JSON data loads correctly  
✅ **Signature Loading**: Tests DSPy optimized prompt parsing  
✅ **Output Parsing**: Validates correct extraction of binary flags  
✅ **Result Saving**: Ensures output format is correct  
✅ **Integration**: Mocks full pipeline (no model needed)

## Expected Output

```
test_load_reasoning_data (__main__.TestLoadReasoningData) ... ok
test_load_optimized_signature (__main__.TestLoadOptimizedSignature) ... ok
test_parse_dspy_outputs (__main__.TestParseDspyOutputs) ... ok
test_save_results (__main__.TestSaveResults) ... ok
test_full_pipeline_mock (__main__.TestIntegration) ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.XXXs

OK
```

## If Tests Pass ✅

You're ready to run the actual script:

```bash
python reasoning_eval/llm_judge_script.py \
  --model "your-model-path" \
  --prompt_path "path/to/optimized_prompt.json" \
  --data_path "reasoning_eval/data_to_label/sample_traces_inital.json" \
  --max_samples 5 \
  --device "0"
```

## If Tests Fail ❌

1. Check Python version (requires >= 3.11)
2. Verify dependencies are installed
3. Check error messages for missing imports
4. See `TEST_README.md` for troubleshooting

## Quick Lint Check (Optional)

```bash
# Install if not already installed
pip install ruff black mypy

# Run quick lint
make lint
```

Ready to test? Run: `python tests/test_llm_judge_script.py`

