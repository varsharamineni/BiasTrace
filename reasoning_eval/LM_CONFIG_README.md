# LM Configuration Guide

## Overview

The `lm_config.py` file contains all the default settings for your language model API. This keeps your configuration centralized and makes it easy to manage different model setups.

## Quick Start

### 1. Edit Default Settings

Open `reasoning_eval/lm_config.py` and modify the defaults:

```python
# Model configuration
DEFAULT_MODEL = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"

# API configuration
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_API_KEY = "dummy"
DEFAULT_CUSTOM_LLM_PROVIDER = "openai"

# Timeout and retry settings
DEFAULT_TIMEOUT = 600.0  # 10 minutes
DEFAULT_MAX_RETRIES = 3

# Sampling parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
```

### 2. Run with Defaults

Once configured, you can run with minimal arguments:

```bash
.venv/bin/python reasoning_eval/llm_judge_script.py \
  --prompt_path "tests/judge_optimized_prompt.json" \
  --data_path "your_data.json"
```

All API settings will be loaded from `lm_config.py`!

### 3. Override Defaults (Optional)

You can still override any setting via command-line:

```bash
.venv/bin/python reasoning_eval/llm_judge_script.py \
  --prompt_path "tests/judge_optimized_prompt.json" \
  --data_path "your_data.json" \
  --model "different-model" \
  --temperature 0.5 \
  --api_base "http://localhost:8001/v1"
```

## Configuration Options

### Model Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `DEFAULT_MODEL` | nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 | Model name or path |

### API Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `DEFAULT_API_BASE` | http://localhost:8000/v1 | Base URL for your vLLM/API server |
| `DEFAULT_API_KEY` | dummy | API key (use "dummy" for local servers) |
| `DEFAULT_CUSTOM_LLM_PROVIDER` | openai | Provider type (openai, anthropic, etc.) |

### Reliability Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `DEFAULT_TIMEOUT` | 600.0 | Timeout per request (seconds) |
| `DEFAULT_MAX_RETRIES` | 3 | Number of retry attempts on failure |

### Sampling Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `DEFAULT_TEMPERATURE` | 0.7 | Sampling temperature (0.0-1.0) |
| `DEFAULT_MAX_TOKENS` | 2048 | Maximum tokens to generate |

## Common Scenarios

### Scenario 1: Multiple vLLM Servers

If you have different models on different ports:

```python
# In lm_config.py, add pre-configured instances:

nemotron_server = dspy.LM(
    model="nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
    api_base="http://localhost:8000/v1",
    api_key="dummy",
    custom_llm_provider="openai",
    timeout=600.0,
    max_retries=3,
    temperature=0.7,
    max_tokens=2048,
)

qwen_server = dspy.LM(
    model="Qwen/Qwen3-4B",
    api_base="http://localhost:8001/v1",
    api_key="dummy",
    custom_llm_provider="openai",
    timeout=600.0,
    max_retries=3,
    temperature=0.7,
    max_tokens=2048,
)
```

Then in your script, you can:
```python
from lm_config import nemotron_server, qwen_server
dspy.configure(lm=nemotron_server)
```

### Scenario 2: Remote API Server

For a remote server with authentication:

```python
# In lm_config.py
DEFAULT_API_BASE = "https://your-server.com/v1"
DEFAULT_API_KEY = "your-actual-api-key"
DEFAULT_TIMEOUT = 1200.0  # Longer timeout for remote
```

### Scenario 3: Different Temperature for Evaluation

```python
# In lm_config.py
DEFAULT_TEMPERATURE = 0.1  # Lower for more deterministic outputs
```

Or override via CLI:
```bash
.venv/bin/python reasoning_eval/llm_judge_script.py \
  --prompt_path "..." \
  --temperature 0.1
```

## Using the create_lm() Function

The `create_lm()` function makes it easy to create LM instances programmatically:

```python
from lm_config import create_lm

# Use all defaults
lm = create_lm()

# Override specific settings
lm = create_lm(
    model="different-model",
    temperature=0.5
)

# Use in DSPy
dspy.configure(lm=lm)
```

## Testing Your Configuration

Test that your config works:

```bash
# Quick test with 1 sample
.venv/bin/python reasoning_eval/llm_judge_script.py \
  --prompt_path "tests/judge_optimized_prompt.json" \
  --data_path "tests/test_data_sample.json" \
  --max_samples 1
```

## Troubleshooting

### Connection Errors

If you get connection errors:
1. Check `DEFAULT_API_BASE` matches your server URL
2. Verify your vLLM server is running: `curl http://localhost:8000/v1/models`
3. Increase `DEFAULT_TIMEOUT` for slower connections

### Model Not Found

If model isn't found:
1. Check `DEFAULT_MODEL` matches the model name in your server
2. Use `curl http://localhost:8000/v1/models` to see available models

### Timeout Issues

If requests timeout:
1. Increase `DEFAULT_TIMEOUT` (especially for large models)
2. Reduce `DEFAULT_MAX_TOKENS` to speed up generation
3. Check server logs for issues

## Best Practices

1. **Keep defaults for your primary setup** - Set `lm_config.py` for your main use case
2. **Use CLI overrides for experiments** - Quick tests with different settings
3. **Document custom configs** - Add comments in `lm_config.py` explaining your setup
4. **Version control** - Commit `lm_config.py` so team uses same defaults
5. **Environment-specific configs** - Consider separate configs for dev/prod

## Examples

### Example 1: Basic Usage (All Defaults)
```bash
./run_judge_example.sh
```

### Example 2: Override Model Only
```bash
.venv/bin/python reasoning_eval/llm_judge_script.py \
  --model "Qwen/Qwen3-4B" \
  --prompt_path "tests/judge_optimized_prompt.json"
```

### Example 3: Override API Settings
```bash
.venv/bin/python reasoning_eval/llm_judge_script.py \
  --prompt_path "tests/judge_optimized_prompt.json" \
  --api_base "http://192.168.1.100:8000/v1" \
  --timeout 1200.0
```

## Summary

- ✅ **Centralized config** - One place for all LM settings
- ✅ **Sensible defaults** - Works out of the box for local vLLM
- ✅ **Easy overrides** - Command-line args override defaults
- ✅ **Reusable** - Import configs in other scripts

Edit `reasoning_eval/lm_config.py` to match your setup! 🎯

