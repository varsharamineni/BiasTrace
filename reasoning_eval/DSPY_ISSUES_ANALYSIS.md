# DSPy Judge Script Issues Analysis

## Problem Summary

The DSPy-based LLM judge script (`llm_judge_script.py`) using Nemotron model produced broken outputs:

1. **Identical Reasoning**: All 12 samples have the exact same reasoning text about "rural areas and technology"
2. **All Zeros**: All binary classification flags are 0, despite the reasoning text mentioning values of 1
3. **Mismatch**: The reasoning text doesn't match the sample categories (Race, Physical_appearance, Disability, etc.)

## Root Cause

DSPy's `ChainOfThought` module is not properly parsing the model's structured output. The `getattr()` calls in `parse_dspy_outputs()` are all falling back to default values (empty string for reasoning, 0 for all flags).

## Evidence

From debug analysis (`debug_dspy_output.py`):
```
Number of unique reasoning texts: 1  ⚠️

The reasoning mentions:
  - Group Assumption: 1
  - Bias Acknowledgement: 1
  - Outside Demographic Knowledge: 1
  - Unresolved: 1
  - Missing Logic: 1

But the judge_output has:
  - group_assumption: 0
  - bias_acknowledgement: 0
  - outside_demo_knowledge: 0
  - unresolved: 0
  - missing_logic: 0
```

## Potential Causes

1. **DSPy Parsing Failure**: The model generates text with structured fields, but DSPy's parser doesn't extract them into attributes
2. **Model Output Format Mismatch**: The model may not be following the expected output format from the DSPy signature
3. **Caching/Repetition Issue**: The same response is being returned for all inputs (possibly a vLLM caching issue or model problem)
4. **Prompt Engineering**: The DSPy prompt construction might not be clear enough for the model

## Comparison with Working Scripts

### DSPy Approach (BROKEN)
- **File**: `llm_judge_script.py`
- **Framework**: DSPy with ChainOfThought
- **Strategy**: Single prompt with all 8 metrics + structured output
- **Status**: ❌ Not parsing correctly

### vLLM Approach (WORKING)
- **File**: `llm_judge_script_vllm.py`
- **Framework**: OpenAI client (no DSPy)
- **Strategy**: Separate prompt per metric (8 prompts total)
- **Status**: ✅ Working (used for other models)

### Claude Approach (WORKING)
- **File**: `llm_eval_claude-sonnet-4-5-20250929_detailed_example_clarification.json`
- **Framework**: Anthropic API
- **Strategy**: Single prompt with JSON structured output
- **Status**: ✅ Working correctly

## Proposed Solutions

### Option 1: Fix DSPy Script (Recommended for DSPy Users)

I've already added improvements to `llm_judge_script.py`:

1. **Better Debug Logging**: Shows what DSPy is actually returning
2. **Fallback Text Parser**: `parse_text_output()` that extracts values from raw text using regex
3. **Automatic Fallback**: Tries text parsing when DSPy attributes are missing

**Next Steps**:
- Run the fixed script with debug output
- Check if the text parser successfully extracts values
- If the model is generating correct text but DSPy isn't parsing it, the fallback will work
- If the model is generating wrong text (same response for all), need to fix prompt or model configuration

### Option 2: Use vLLM Script Instead

The vLLM script is already working. You can:
```bash
uv run reasoning_eval/llm_judge_script_vllm.py \
  --model nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \
  --data_path reasoning_eval/data_to_label/sample_traces_inital.json \
  --prompts reasoning_eval/prompts.json \
  --mode multi_prompt \
  --temperature 0.7 \
  --seed 42
```

**Pros**: 
- Already tested and working
- More modular (separate prompts per metric)
- Better for debugging individual metrics

**Cons**:
- 8x more API calls (one per metric)
- Slower overall

### Option 3: Create Simple OpenAI-based Single-Prompt Script

Write a new script similar to Claude's approach but using OpenAI client:
- Single prompt with all 8 metrics
- Parse JSON or structured text output
- No DSPy dependency
- Similar to how Claude script works

## Recommended Action

1. **Test the fixed DSPy script** with the new fallback parsing:
   ```bash
   uv run reasoning_eval/llm_judge_script.py \
     --model nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \
     --prompt_path tests/judge_optimized_prompt.json \
     --data_path reasoning_eval/data_to_label/sample_traces_inital.json \
     --max_samples 3 \
     --temperature 0.7 \
     --seed 42
   ```

2. **Check the debug output** to see:
   - What DSPy is returning
   - Whether the text parser successfully extracts values
   - If the model generates different responses per sample

3. **If still broken**, investigate:
   - vLLM caching settings (might be caching first response)
   - Model temperature/seed (might need more diversity)
   - Prompt clarity (model might not understand the format)

## Files Modified

- ✅ `reasoning_eval/llm_judge_script.py` - Added debug logging and fallback parser
- ✅ `reasoning_eval/debug_dspy_output.py` - Created analysis script
- ✅ `reasoning_eval/DSPY_ISSUES_ANALYSIS.md` - This file

## Next Steps

Run the test command above and share the debug output to see if:
1. The text parser extracts correct values
2. Each sample gets unique responses
3. The model is following the expected format

