# DSPy Testing Guide

## What I Fixed

### Issue
DSPy was not including the actual input values (context, question, etc.) in the prompt sent to the model. The model only saw the task instructions.

### Changes Made

1. **Fixed `load_optimized_signature()`** 
   - Changed how fields are loaded from the JSON
   - Now properly uses `prefix` parameter for DSPy fields
   - Ignores the `${template}` placeholders in field descriptions

2. **Enhanced Debug Output**
   - Shows the actual prompt sent to the model
   - Displays messages format
   - Shows model responses

## How to Test

Run this command on your server:

```bash
uv run reasoning_eval/llm_judge_script.py \
  --model nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \
  --prompt_path ../dspy_judge/optimized_reasoning_analyzer.json \
  --data_path reasoning_eval/data_to_label/sample_traces_inital.json \
  --max_samples 2 \
  --temperature 0.7 \
  --seed 42
```

## What to Look For

### ✅ Success Indicators

In the debug output, check:

1. **"ACTUAL PROMPT SENT TO MODEL"** section should show:
   ```
   Context: The doctor treated two patients...
   Question: Who has COVID-19?
   Answer Options: ['Not answerable', 'The Hispanic woman', 'The Asian woman']
   Model Reasoning: Okay, let's try to figure this out...
   ```

2. **Model response** should:
   - NOT say "input lacks necessary components"
   - Actually analyze the reasoning
   - Generate binary flags (0 or 1) for each metric

3. **Fallback parsing result** should show:
   - Non-zero values for relevant metrics
   - Proper evaluation of the reasoning

### ❌ Still Broken Indicators

If you see:
- Model says "The provided input lacks the necessary components"
- Prompt doesn't contain the actual context/question values
- All binary flags are still 0

Then DSPy is still not formatting the prompt correctly.

## If Still Broken

Two alternatives:

### Option A: Manual Prompt Construction
I can create a version that constructs the prompt manually instead of relying on DSPy's automatic formatting.

### Option B: Use Working Multi-Prompt Script
Fall back to the vLLM script which is proven to work:

```bash
uv run reasoning_eval/llm_judge_script_vllm.py \
  --model nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \
  --data_path reasoning_eval/data_to_label/sample_traces_inital.json \
  --prompts reasoning_eval/prompts.json \
  --mode multi_prompt \
  --temperature 0.7 \
  --seed 42
```

## Files Modified

- ✅ `reasoning_eval/llm_judge_script.py` - Fixed signature loading + enhanced debug
- ✅ `reasoning_eval/DSPY_TESTING_GUIDE.md` - This file

