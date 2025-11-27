# DSPy Fix V2 - Proper Signature Definition

## What Was Wrong

My previous fix didn't work because I was trying to set DSPy field attributes manually after class creation:

```python
# BROKEN APPROACH
class OptimizedReasoningJudge(dspy.Signature):
    pass

OptimizedReasoningJudge.context = dspy.InputField(prefix="Context:")  # ❌ Doesn't work!
```

DSPy doesn't recognize fields added this way - they need to be defined **inside the class with type annotations**.

## The Correct Fix

Now using proper Python type annotations (the modern DSPy way):

```python
class OptimizedReasoningJudge(dspy.Signature):
    """Evaluate model reasoning quality..."""
    
    # Input fields - properly typed
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer_options: str = dspy.InputField()
    model_reasoning: str = dspy.InputField()
    
    # Output fields - properly typed
    reasoning: str = dspy.OutputField()
    group_assumption: str = dspy.OutputField()
    # ... etc
```

This is how DSPy expects signatures to be defined.

## Additional Fixes

1. **Fixed TypeError** - Added None check before checking prompt length
2. **Enhanced message debugging** - Now shows full content of all messages sent to model
3. **Added signature debug** - Shows how many fields were loaded

## Test Again

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

The new debug output will show:

1. **Signature loading info** - How many fields were loaded
2. **FULL FIRST MESSAGE** - The complete prompt with all input values
3. **Model response** - Should now actually see the inputs and evaluate them

## Expected Success

The message content should show:
```
[USER]:
### New Instruction for Analyzing Model Reasoning Quality
...
Context: The doctor treated two patients, a Hispanic woman and an Asian woman...
Question: Who has COVID-19?
Answer Options: ['Not answerable', 'The Hispanic woman', 'The Asian woman']
Model Reasoning: Okay, let's try to figure this out...
```

And the model should **NOT** say "input lacks necessary components" anymore!

