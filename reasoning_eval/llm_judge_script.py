import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import dspy
from dspy import LM


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
    
    # Build DSPy signature dynamically
    class OptimizedReasoningJudge(dspy.Signature):
        pass
    
    # Set instructions
    OptimizedReasoningJudge.__doc__ = instructions
    
    # Add input fields (first 4 fields are inputs based on the format)
    input_fields = {
        "context": dspy.InputField(desc=fields[0]["description"]),
        "question": dspy.InputField(desc=fields[1]["description"]),
        "answer_options": dspy.InputField(desc=fields[2]["description"]),
        "model_reasoning": dspy.InputField(desc=fields[3]["description"]),
    }
    
    # Add output fields (remaining fields are outputs)
    output_fields = {
        "reasoning": dspy.OutputField(desc=fields[4]["description"], prefix=fields[4]["prefix"]),
        "group_assumption": dspy.OutputField(desc=fields[5]["description"], prefix=fields[5]["prefix"]),
        "bias_acknowledgement": dspy.OutputField(desc=fields[6]["description"], prefix=fields[6]["prefix"]),
        "meta_reflection": dspy.OutputField(desc=fields[7]["description"], prefix=fields[7]["prefix"]),
        "outside_demo_knowledge": dspy.OutputField(desc=fields[8]["description"], prefix=fields[8]["prefix"]),
        "outside_topical_knowledge": dspy.OutputField(desc=fields[9]["description"], prefix=fields[9]["prefix"]),
        "unresolved": dspy.OutputField(desc=fields[10]["description"], prefix=fields[10]["prefix"]),
        "overthinking": dspy.OutputField(desc=fields[11]["description"], prefix=fields[11]["prefix"]),
        "missing_logic": dspy.OutputField(desc=fields[12]["description"], prefix=fields[12]["prefix"]),
    }
    
    # Merge all fields
    all_fields = {**input_fields, **output_fields}
    for field_name, field_obj in all_fields.items():
        setattr(OptimizedReasoningJudge, field_name, field_obj)
    
    return OptimizedReasoningJudge


def run_dspy_evaluation(
    batch_data: List[Dict[str, Any]],
    judge_module: dspy.Module,
) -> List[Any]:
    """Run DSPy evaluation for a batch of samples."""
    print(f"💬 Running DSPy inference on {len(batch_data)} samples...")
    results = []
    
    for item in tqdm(batch_data):
        result = judge_module(
            context=item.get("context", ""),
            question=item.get("question", ""),
            answer_options=str(item.get("answer_options", [])),
            model_reasoning=item.get("model_reasoning", ""),
        )
        results.append(result)
    
    return results


def parse_dspy_outputs(batch_data, outputs, model_name) -> List[Dict[str, Any]]:
    """Parse DSPy outputs and combine with original data."""
    results = []
    for item, output in zip(batch_data, outputs):
        # Extract all output fields from the optimized signature
        judge_output = {
            "reasoning": getattr(output, "reasoning", ""),
            "group_assumption": int(getattr(output, "group_assumption", 0)),
            "bias_acknowledgement": int(getattr(output, "bias_acknowledgement", 0)),
            "meta_reflection": int(getattr(output, "meta_reflection", 0)),
            "outside_demo_knowledge": int(getattr(output, "outside_demo_knowledge", 0)),
            "outside_topical_knowledge": int(getattr(output, "outside_topical_knowledge", 0)),
            "unresolved": int(getattr(output, "unresolved", 0)),
            "overthinking": int(getattr(output, "overthinking", 0)),
            "missing_logic": int(getattr(output, "missing_logic", 0)),
        }

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

    # 3. Configure DSPy with vLLM backend
    print(f"🚀 Configuring DSPy with model: {args.model}")
    
    lm = LM(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=2048,
        seed=args.seed,
        # Additional vLLM-specific params can be passed here
    )
    
    dspy.configure(lm=lm)

    # 4. Initialize DSPy ChainOfThought module with optimized signature
    print("🧠 Initializing DSPy ChainOfThought module with optimized signature...")
    judge_module = dspy.ChainOfThought(OptimizedSignature)

    # 5. Run DSPy evaluation
    outputs = run_dspy_evaluation(data, judge_module)

    # 6. Parse and save
    results = parse_dspy_outputs(data, outputs, args.model)
    save_results(
        results, 
        args.model, 
        args.output_dir, 
        args.temperature,
        args.top_p,
        args.top_k,
        args.seed
    )


# ================================================================
# CLI Entry Point
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate reasoning traces using DSPy ChainOfThought.")
    parser.add_argument("--model", type=str, required=True, help="Path or name of model to use (e.g., Qwen/Qwen3-4B).")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to DSPy optimized prompt JSON file.")
    parser.add_argument("--data_path", type=str, default="reasoning_eval/data_to_label/sample_traces_inital.json")
    parser.add_argument("--output_dir", type=str, default="reasoning_eval/llm_judge_samples/")
    parser.add_argument("--device", type=str, default="0", help="CUDA device (e.g., '0' or '0,1')")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for quick runs.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
