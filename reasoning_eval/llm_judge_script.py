import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import dspy
from lm_config import (
    create_lm,
    DEFAULT_MODEL,
    DEFAULT_API_BASE,
    DEFAULT_API_KEY,
    DEFAULT_CUSTOM_LLM_PROVIDER,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)


# ================================================================
# Utility functions
# ================================================================
def load_reasoning_data(path: str) -> List[Dict[str, Any]]:
    """Load model reasoning traces or evaluation data."""
    with open(path, "r") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} samples from {path}")
    return data


# ================================================================
# DSPy Signature and Module
# ================================================================
def load_optimized_signature(prompt_path: str):
    """Load DSPy optimized signature from JSON file."""
    with open(prompt_path, "r") as f:
        prompt_data = json.load(f)
    
    # Extract the signature information
    signature_data = prompt_data["predict"]["signature"]
    instructions = signature_data["instructions"]
    fields = signature_data["fields"]
    
    print(f"\n🔍 DEBUG: Loading signature with {len(fields)} fields")
    print(f"🔍 DEBUG: Instructions length: {len(instructions)} chars")
    
    # Build DSPy signature dynamically using proper annotation syntax
    class OptimizedReasoningJudge(dspy.Signature):
        """Evaluate model reasoning quality based on specific criteria."""
        
        # Input fields
        context: str = dspy.InputField()
        question: str = dspy.InputField()
        answer_options: str = dspy.InputField()
        model_reasoning: str = dspy.InputField()
        
        # Output fields  
        reasoning: str = dspy.OutputField()
        group_assumption: str = dspy.OutputField()
        bias_acknowledgement: str = dspy.OutputField()
        meta_reflection: str = dspy.OutputField()
        outside_demo_knowledge: str = dspy.OutputField()
        outside_topical_knowledge: str = dspy.OutputField()
        unresolved: str = dspy.OutputField()
        overthinking: str = dspy.OutputField()
        missing_logic: str = dspy.OutputField()
    
    # Override the docstring with the detailed instructions
    OptimizedReasoningJudge.__doc__ = instructions
    
    print(f"✅ Signature created with {len([f for f in dir(OptimizedReasoningJudge) if not f.startswith('_')])} attributes")
    
    return OptimizedReasoningJudge


def run_dspy_evaluation(
    batch_data: List[Dict[str, Any]],
    judge_module: dspy.Module,
) -> List[Any]:
    """Run DSPy evaluation for a batch of samples."""
    print(f"💬 Running DSPy inference on {len(batch_data)} samples...")
    results = []
    
    for idx, item in enumerate(tqdm(batch_data)):
        # DEBUG: Print first sample input
        if idx == 0:
            print("\n🔍 DEBUG: First sample input:")
            print(f"  - context: {item.get('context', '')[:100]}...")
            print(f"  - question: {item.get('question', '')}")
            print(f"  - answer_options: {item.get('answer_options', [])}")
            print(f"  - model_reasoning: {item.get('model_reasoning', '')[:100]}...")
        
        result = judge_module(
            context=item.get("context", ""),
            question=item.get("question", ""),
            answer_options=str(item.get("answer_options", [])),
            model_reasoning=item.get("model_reasoning", ""),
        )
        
        # DEBUG: Print first sample result
        if idx == 0:
            print(f"\n🔍 DEBUG: First sample raw result: {result}")
        
        results.append(result)
    
    return results


def parse_binary_value(value) -> int:
    """Parse a binary value that might be int, str, or missing."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ['1', 'yes', 'true']:
            return 1
        elif value in ['0', 'no', 'false']:
            return 0
    return 0  # Default to 0 if unclear


def parse_text_output(text: str) -> Dict[str, Any]:
    """
    Fallback parser: Extract structured fields from model text output.
    Handles cases where DSPy doesn't properly parse the response.
    """
    import re
    
    result = {
        "reasoning": "",
        "group_assumption": 0,
        "bias_acknowledgement": 0,
        "meta_reflection": 0,
        "outside_demo_knowledge": 0,
        "outside_topical_knowledge": 0,
        "unresolved": 0,
        "overthinking": 0,
        "missing_logic": 0,
    }
    
    # Extract reasoning (everything before the first binary field)
    reasoning_patterns = [
        r"Reasoning:\s*(.+?)(?=\n(?:Group Assumption|Bias Acknowledgement|Meta Reflection):|$)",
        r"reasoning:\s*(.+?)(?=\n(?:group_assumption|bias_acknowledgement|meta_reflection):|$)",
    ]
    
    for pattern in reasoning_patterns:
        reasoning_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
            break
    
    # If no reasoning found, use the whole text
    if not result["reasoning"]:
        result["reasoning"] = text
    
    # Extract binary fields
    field_patterns = {
        "group_assumption": r"Group Assumption:\s*([01])",
        "bias_acknowledgement": r"Bias Acknowledgement:\s*([01])",
        "meta_reflection": r"Meta Reflection:\s*([01])",
        "outside_demo_knowledge": r"Outside Demo Knowledge:\s*([01])",
        "outside_topical_knowledge": r"Outside Topical Knowledge:\s*([01])",
        "unresolved": r"Unresolved:\s*([01])",
        "overthinking": r"Overthinking:\s*([01])",
        "missing_logic": r"Missing Logic:\s*([01])",
    }
    
    for field, pattern in field_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result[field] = int(match.group(1))
    
    return result


def parse_dspy_outputs(batch_data, outputs, model_name) -> List[Dict[str, Any]]:
    """Parse DSPy outputs and combine with original data."""
    results = []
    for idx, (item, output) in enumerate(zip(batch_data, outputs)):
        # DEBUG: Print what DSPy actually returned
        if idx == 0:  # Only print first sample to avoid spam
            print(f"\n🔍 DEBUG: DSPy output object type: {type(output)}")
            print(f"🔍 DEBUG: DSPy output attributes: {[attr for attr in dir(output) if not attr.startswith('_')]}")
            if hasattr(output, 'reasoning'):
                print(f"🔍 DEBUG: reasoning value: {output.reasoning[:100]}...")
            if hasattr(output, 'group_assumption'):
                print(f"🔍 DEBUG: group_assumption value: {output.group_assumption}")
            if hasattr(output, 'completions'):
                print(f"🔍 DEBUG: completions: {output.completions}")
        
        # Try to extract fields - handle both attribute access and potential text parsing
        reasoning = getattr(output, "reasoning", "")
        
        # Extract binary fields
        judge_output = {
            "reasoning": reasoning,
            "group_assumption": parse_binary_value(getattr(output, "group_assumption", 0)),
            "bias_acknowledgement": parse_binary_value(getattr(output, "bias_acknowledgement", 0)),
            "meta_reflection": parse_binary_value(getattr(output, "meta_reflection", 0)),
            "outside_demo_knowledge": parse_binary_value(getattr(output, "outside_demo_knowledge", 0)),
            "outside_topical_knowledge": parse_binary_value(getattr(output, "outside_topical_knowledge", 0)),
            "unresolved": parse_binary_value(getattr(output, "unresolved", 0)),
            "overthinking": parse_binary_value(getattr(output, "overthinking", 0)),
            "missing_logic": parse_binary_value(getattr(output, "missing_logic", 0)),
        }
        
        # Check if all values are 0 (indicating parsing failure) - try fallback parsing
        all_zero = all(judge_output[k] == 0 for k in judge_output if k != "reasoning")
        
        if all_zero:
            # Try to get raw text from completions attribute
            raw_text = None
            if hasattr(output, 'completions') and output.completions:
                if isinstance(output.completions, list) and len(output.completions) > 0:
                    raw_text = str(output.completions[0])
                else:
                    raw_text = str(output.completions)
            
            # If no completions, try converting whole output to string
            if not raw_text and hasattr(output, '__str__'):
                raw_text = str(output)
            
            if raw_text:
                if idx == 0:
                    print("\n⚠️  WARNING: All binary flags are 0 for first sample!")
                    print("📝 Attempting fallback text parsing...")
                    print(f"Raw text preview: {raw_text[:300]}...")
                
                # Use fallback text parser
                parsed = parse_text_output(raw_text)
                judge_output.update(parsed)
                
                if idx == 0:
                    print("✅ Fallback parsing result:")
                    for k, v in parsed.items():
                        if k != "reasoning":
                            print(f"  {k}: {v}")

        results.append({
            "sample_id": item.get("sample_id", ""),
            "category": item.get("bbq_category", ""),
            "example_id": item.get("example_id", ""),
            "model": item.get("model", ""),
            "prompt_type": item.get("prompt_type", ""),
            "judge_model": model_name,
            "judge_reasoning": judge_output["reasoning"],      # Judge's reasoning
            "judge_output": judge_output,                       # All binary flags + reasoning
        })
    return results


def save_results(results, model_name, output_dir, temperature, top_p, top_k, seed, max_tokens=2048):
    """Save evaluation results along with metadata to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct final dict with metadata + outputs
    output_data = {
        "metadata": {
            "judge_model": model_name,
            "framework": "dspy",
            "module": "ChainOfThought",
            "sampling_params": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed,
            }
        },
        "results": results
    }

    filename = f"{output_dir}/llm_eval_dspy_{model_name.replace('/', '_')}.json"
    
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"✅ Saved {len(results)} results + metadata to {filename}")
    return filename


# ================================================================
# Main script logic
# ================================================================
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 1. Load data
    data = load_reasoning_data(args.data_path)

    # Optional: subset for debugging
    if args.max_samples:
        data = data[:args.max_samples]

    # 2. Load optimized DSPy signature
    print(f"📄 Loading optimized prompt from: {args.prompt_path}")
    OptimizedSignature = load_optimized_signature(args.prompt_path)

    # 3. Configure DSPy with LM backend (from lm_config.py)
    print(f"🚀 Configuring DSPy with model: {args.model}")
    
    lm = create_lm(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        custom_llm_provider=args.custom_llm_provider,
        timeout=args.timeout,
        max_retries=args.max_retries,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    dspy.configure(lm=lm)

    # 4. Initialize DSPy ChainOfThought module with optimized signature
    print("🧠 Initializing DSPy ChainOfThought module with optimized signature...")
    judge_module = dspy.ChainOfThought(OptimizedSignature)

    # 5. Run DSPy evaluation
    outputs = run_dspy_evaluation(data, judge_module)
    
    # DEBUG: Inspect DSPy history for first sample
    if len(outputs) > 0 and hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'history'):
        print("\n🔍 DEBUG: Checking DSPy LM history...")
        history = dspy.settings.lm.history
        if history:
            print(f"🔍 DEBUG: History length: {len(history)}")
            if len(history) > 0:
                first_call = history[0]
                print(f"🔍 DEBUG: First history entry keys: {first_call.keys() if isinstance(first_call, dict) else 'N/A'}")
                if isinstance(first_call, dict):
                    # Show the actual prompt sent to the model
                    if 'prompt' in first_call:
                        prompt_text = first_call['prompt']
                        if prompt_text:
                            print("\n🔍 DEBUG: ACTUAL PROMPT SENT TO MODEL:")
                            print("="*80)
                            print(prompt_text[:1500] if len(prompt_text) > 1500 else prompt_text)
                            print("="*80)
                        else:
                            print("\n🔍 DEBUG: Prompt field exists but is None")
                    if 'messages' in first_call:
                        messages = first_call['messages']
                        print("\n🔍 DEBUG: MESSAGES FORMAT:")
                        print(f"Number of messages: {len(messages) if isinstance(messages, list) else 'N/A'}")
                        if isinstance(messages, list) and len(messages) > 0:
                            print("\n🔍 DEBUG: FULL FIRST MESSAGE:")
                            print("="*80)
                            for msg in messages:
                                if isinstance(msg, dict):
                                    role = msg.get('role', 'unknown')
                                    content = msg.get('content', '')
                                    print(f"\n[{role.upper()}]:")
                                    print(content[:2000] if len(content) > 2000 else content)
                            print("="*80)
                    if 'response' in first_call:
                        print(f"\n🔍 DEBUG: First response preview: {str(first_call['response'])[:300]}...")
                    if 'outputs' in first_call:
                        print(f"\n🔍 DEBUG: First outputs: {first_call['outputs'][:500]}...")

    # 6. Parse and save
    results = parse_dspy_outputs(data, outputs, args.model)
    save_results(
        results, 
        args.model, 
        args.output_dir, 
        args.temperature,
        args.top_p,
        args.top_k,
        args.seed,
        max_tokens=args.max_tokens
    )


# ================================================================
# CLI Entry Point
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate reasoning traces using DSPy ChainOfThought.")
    
    # Model and data arguments
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model name/path (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to DSPy optimized prompt JSON file.")
    parser.add_argument("--data_path", type=str, default="reasoning_eval/data_to_label/sample_traces_inital.json")
    parser.add_argument("--output_dir", type=str, default="reasoning_eval/llm_judge_samples/")
    parser.add_argument("--device", type=str, default="0", help="CUDA device (e.g., '0' or '0,1')")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for quick runs.")
    
    # API configuration (defaults from lm_config.py)
    parser.add_argument("--api_base", type=str, default=DEFAULT_API_BASE, help=f"API base URL (default: {DEFAULT_API_BASE})")
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY, help=f"API key (default: {DEFAULT_API_KEY})")
    parser.add_argument("--custom_llm_provider", type=str, default=DEFAULT_CUSTOM_LLM_PROVIDER, help=f"Custom LLM provider (default: {DEFAULT_CUSTOM_LLM_PROVIDER})")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"Timeout per request in seconds (default: {DEFAULT_TIMEOUT}s)")
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES, help=f"Max retries for failed requests (default: {DEFAULT_MAX_RETRIES})")
    
    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Max tokens to generate (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    main(args)
