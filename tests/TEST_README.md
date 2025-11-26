# Testing Guide for LLM Judge Script

This guide explains how to test and lint the LLM judge script before running it with actual models.

**Note:** All test files are now in the `tests/` directory at the project root.

## Quick Start

### 1. Install Linting Tools

```bash
make install-lint
```

This installs:
- **ruff**: Fast Python linter
- **black**: Code formatter
- **mypy**: Static type checker

### 2. Run Tests

```bash
# Run just the LLM judge tests
make test-judge

# Or run all tests in the project
make test
```

### 3. Run Linting

```bash
# Check code quality (no changes)
make lint

# Auto-fix issues where possible
make lint-fix

# Format code with black
make format
```

### 4. Run Everything

```bash
# Run both linting and tests
make check
```

## Test Files

All test files are located in `tests/`:

### Unit Tests
- **`tests/test_llm_judge_script.py`**: Comprehensive unit tests for all functions
  - Tests data loading
  - Tests signature loading
  - Tests output parsing
  - Tests result saving
  - Mocks DSPy components (no actual model inference needed)

### Test Data
- **`tests/test_data_sample.json`**: Sample reasoning traces for testing
- **`tests/test_optimized_prompt.json`**: Sample optimized DSPy prompt

## Testing Without Model Inference

The unit tests use Python's `unittest.mock` to simulate DSPy behavior without loading actual models. This allows you to:

1. Verify function logic
2. Test data flow through the pipeline
3. Catch errors before running expensive inference
4. Ensure output format correctness

## Example: Dry Run Test

To test if your script can load real data without running inference:

```python
from reasoning_eval.llm_judge_script import load_reasoning_data, load_optimized_signature

# Test loading your actual data
data = load_reasoning_data("reasoning_eval/data_to_label/sample_traces_inital.json")
print(f"✅ Loaded {len(data)} samples")

# Test loading your optimized prompt
signature = load_optimized_signature("path/to/your/optimized_prompt.json")
print(f"✅ Loaded signature with instructions")
```

## Makefile Targets

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install-lint` | Install linting tools |
| `make lint` | Run all linters (check only) |
| `make lint-fix` | Auto-fix linting issues |
| `make format` | Format code with black |
| `make test` | Run all tests |
| `make test-judge` | Run LLM judge tests only |
| `make check` | Run linting + tests |
| `make clean` | Remove cache files |

## Before Running with Real Models

1. **Run tests**: `make test-judge`
2. **Run linters**: `make lint`
3. **Verify data loads**: Check that your JSON files are valid
4. **Check imports**: Ensure `dspy` and `vllm` are installed

## Common Issues

### Import Errors
If you see `Import "dspy" could not be resolved`:
- This is a warning from the linter
- Install dspy: `pip install dspy-ai`
- Or ignore if dspy is installed but not in the linter's path

### Test Failures
If tests fail:
1. Check that all test data files exist
2. Verify JSON format is correct
3. Ensure Python version >= 3.11

### Linting Warnings
Linting warnings are informational and won't prevent execution. Fix critical errors first.

## CI/CD Integration

You can add these commands to your CI pipeline:

```bash
# In your CI script
make install-lint
make check  # Runs linting + tests
```

## Next Steps

After tests pass:
1. Prepare your optimized prompt JSON
2. Set up your model path/API
3. Run with a small sample: `--max_samples 5`
4. Verify outputs before running full evaluation

