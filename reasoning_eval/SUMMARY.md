# DSPy Judge Script - Issue Analysis & Fixes

## Issues Found

Looking at your DSPy Nemotron output file, I identified two critical problems:

### 1. **Identical Reasoning for All Samples**
All 12 samples have the exact same reasoning text about "rural areas and technology" - which doesn't make sense for different categories (Race, Physical_appearance, Disability, SES, Age, etc.).

### 2. **Mismatched Binary Flags**
- The `judge_reasoning` text mentions: `Group Assumption: 1`, `Bias Acknowledgement: 1`, `Unresolved: 1`, etc.
- But the `judge_output` object shows all fields as `0`
- This means DSPy's parser isn't extracting the structured values from the model's text response

## Root Cause

DSPy's `ChainOfThought` module generated text output but failed to parse it into structured attributes. The `getattr()` calls in `parse_dspy_outputs()` all fell back to default values (0 for all flags).

## What I've Done

### ✅ Created Diagnostic Tools

**File: `reasoning_eval/debug_dspy_output.py`**
- Analyzes the broken output file
- Confirms all reasoning is identical
- Confirms all binary flags are 0
- Shows the mismatch between reasoning text and structured output

Run it with:
```bash
uv run reasoning_eval/debug_dspy_output.py
```

### ✅ Fixed the DSPy Script

**File: `reasoning_eval/llm_judge_script.py`**

Added three improvements:

1. **Enhanced Debug Logging**
   - Shows what DSPy returns (object type, attributes, values)
   - Displays first sample's input and output
   - Checks DSPy's internal history

2. **Text-Based Fallback Parser**
   - `parse_text_output()` function that extracts values using regex
   - Handles cases where DSPy doesn't parse correctly
   - Looks for patterns like "Group Assumption: 1" in the text

3. **Automatic Fallback Logic**
   - Detects when all binary flags are 0 (parsing failure)
   - Tries to get raw text from `output.completions` or string conversion
   - Applies text parser to extract actual values
   - Reports success/failure

### ✅ Documentation

**File: `reasoning_eval/DSPY_ISSUES_ANALYSIS.md`**
- Detailed analysis of the problem
- Comparison with working scripts (vLLM, Claude)
- Three proposed solutions with pros/cons
- Recommended next steps

## Latest Fix (After Your Test Run)

### New Issue Discovered
Your test showed that **DSPy wasn't including input values in the prompt**! The model said:
> "The provided input lacks the necessary components"

But the inputs WERE being passed - DSPy just wasn't formatting them into the prompt correctly.

### Fix Applied
1. **Rewrote `load_optimized_signature()`** to properly use DSPy field `prefix` parameters
2. **Added detailed prompt debugging** to show exactly what's sent to the model
3. **Enhanced DSPy history inspection** to diagnose formatting issues

## Next Steps: Test the New Fix

```bash
uv run reasoning_eval/llm_judge_script.py \
  --model nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \
  --prompt_path ../dspy_judge/optimized_reasoning_analyzer.json \
  --data_path reasoning_eval/data_to_label/sample_traces_inital.json \
  --max_samples 2 \
  --temperature 0.7 \
  --seed 42
```

**Look for**: The "ACTUAL PROMPT SENT TO MODEL" section should now show the context, question, and reasoning values!

### What to Look For

1. **Debug output shows**:
   - What DSPy actually returns
   - Whether samples get unique responses (not all the same)
   - If fallback parser successfully extracts values

2. **Possible Outcomes**:
   
   **Best Case**: Text parser extracts correct values
   - Output shows: "✅ Fallback parsing result: group_assumption: 1, bias_acknowledgement: 1, ..."
   - Binary flags now have correct non-zero values
   - ✅ Problem solved!
   
   **Medium Case**: Model generates same text for all samples
   - This suggests a caching or prompt issue
   - Need to check vLLM caching settings
   - May need to increase temperature or vary prompts
   
   **Worst Case**: Model outputs don't contain the expected format
   - Text doesn't have "Group Assumption: X" format
   - Model isn't following the DSPy signature instructions
   - Need to revise the prompt or use a different approach

## Alternative: Use the Working vLLM Script

If DSPy continues to have issues, you already have a working alternative:

```bash
uv run reasoning_eval/llm_judge_script_vllm.py \
  --model nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \
  --data_path reasoning_eval/data_to_label/sample_traces_inital.json \
  --prompts reasoning_eval/prompts.json \
  --mode multi_prompt \
  --temperature 0.7 \
  --seed 42
```

This uses 8 separate prompts (one per metric) and already works with other models.

## Files Modified/Created

- ✅ `reasoning_eval/llm_judge_script.py` - Fixed with fallback parser + debug logging
- ✅ `reasoning_eval/debug_dspy_output.py` - Diagnostic script
- ✅ `reasoning_eval/DSPY_ISSUES_ANALYSIS.md` - Detailed analysis
- ✅ `reasoning_eval/SUMMARY.md` - This file

All linting errors have been fixed.

