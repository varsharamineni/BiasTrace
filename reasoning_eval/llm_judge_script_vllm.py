#!/usr/bin/env python
import os
import json
import argparse
import re
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI
from prompt_manager import PromptManager

# ================================================================
# Utility functions
# ================================================================
def load_reasoning_data(path: str) -> List[Dict[str, Any]]:
    """Load model reasoning traces or evaluation data."""
    with open(path, "r") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} samples from {path}")
    return data


def create_messages(prompt: str) -> List[Dict[str, str]]:
    """Format prompt into OpenAI/Anthropic chat structure."""
    return [{"role": "user", "content": prompt}]


def build_batch_messages(batch_data, pm, prompt_key):
    """Generate batch of chat messages with reasoning context."""
    messages_batch = []
    for item in batch_data:
        prompt = pm.get_prompt(
            prompt_key,
            reasoning_trace=item.get("model_reasoning", ""),
            final_answer=item.get("model_answer", ""),
            category=item.get("bbq_category", ""),
            context=item.get("context", ""),
            question=item.get("question", ""),
            answer_options=item.get("answer_options", []),
            sample_id=item.get("sample_id", ""),
            example_id=item.get("example_id", ""),
            model=item.get("model", ""),
            prompt_type=item.get("prompt_type", "")
        )
        messages_batch.append(create_messages(prompt))
    return messages_batch


# ================================================================
# Flexible vLLM Evaluation
# ================================================================
def run_vllm_evaluation(
    model_name: str,
    messages_batch: List[List[Dict[str, str]]],
    temperature: float = 0.6,
    max_tokens: int = 2048,
    seed: int = 42,
    reasoning_level: str = "medium",
    top_p: float = 1.0 
):
    """
    Calls the OpenAI Responses API with reasoning enabled.
    Falls back to chat completions if needed.
    """
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL")
    )

    outputs = []
    print(f"🚀 Calling model via flexible vLLM client: {model_name} with reasoning={reasoning_level}")

    for messages in tqdm(messages_batch, desc="Inference batches"):
        # Try Responses API --------------------------
        try:
            response = client.responses.create(
                model=model_name,
                input=messages,
                max_output_tokens=max_tokens,
                temperature=temperature,
                reasoning={"effort": reasoning_level},
                top_p=top_p  
            )
            outputs.append(response)
            continue
        except Exception:
            pass

        # Fallback: Chat Completions API --------------------
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            outputs.append(response)
        except Exception as e:
            print(f"⚠️ Error on sample: {e}")
            outputs.append(None)

    return outputs


# ================================================================
# Parse Outputs
# ================================================================
def parse_outputs(batch_data, outputs, model_name, prompt_key, reasoning_level="medium"):
    """Extract full reasoning + last JSON block."""
    results = []
    json_block_pattern = re.compile(r"\{[\s\S]*?\}", re.MULTILINE)

    for item, output in zip(batch_data, outputs):
        if not output:
            results.append(_empty_parse(item, model_name, prompt_key))
            continue

        # Try Responses API format
        text = getattr(output, "output_text", None)
        if not text:
            # Fallback: Chat completions
            try:
                text = output.choices[0].message["content"].strip()
            except Exception:
                text = None

        reasoning_text = text

        # extract JSON block if present
        metric_value = None
        if text:
            matches = json_block_pattern.findall(text)
            if matches:
                try:
                    parsed_json = json.loads(matches[-1])
                    metric_value = list(parsed_json.values())[0]
                except Exception:
                    pass

        metric_name = prompt_key.replace("judge_", "")

        results.append({
            "sample_id": item.get("sample_id", ""),
            "category": item.get("bbq_category", ""),
            "example_id": item.get("example_id", ""),
            "model": item.get("model", ""),
            "prompt_type": item.get("prompt_type", ""),
            "judge_model": model_name,
            "judge_prompt": prompt_key,
            "reasoning_on": True,
            "reasoning_level": reasoning_level,
            "judge_output": {metric_name: metric_value},
            "judge_explanations": {metric_name: reasoning_text},
        })

    return results


def _empty_parse(item, model_name, prompt_key):
    metric_name = prompt_key.replace("judge_", "")
    return {
        "sample_id": item.get("sample_id", ""),
        "category": item.get("bbq_category", ""),
        "example_id": item.get("example_id", ""),
        "model": item.get("model", ""),
        "prompt_type": item.get("prompt_type", ""),
        "judge_model": model_name,
        "judge_prompt": prompt_key,
        "reasoning_on": True,
        "reasoning_level": "medium",
        "judge_output": {metric_name: None},
        "judge_explanations": {metric_name: None},
    }


# ================================================================
# Save Results
# ================================================================
def save_results(results, model_name, output_dir, prompt_key, params, extra_metadata=None):
    os.makedirs(output_dir, exist_ok=True)
    output_data = {
        "metadata": {
            "judge_model": model_name,
            "judge_prompt": prompt_key,
            "reasoning_on": True,
            "reasoning_level": params.get("reasoning_level", "medium"),
            "sampling_params": params,
        },
        "results": results
    }
    if extra_metadata:
        output_data["metadata"].update(extra_metadata)

    filename = f"{output_dir}/llm_eval_{model_name.replace('/', '_')}_{prompt_key}.json"
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"✅ Saved {len(results)} results → {filename}")
    return filename


# ================================================================
# Multi-prompt Mode
# ================================================================
def run_multi_prompt_evaluation(args):
    metric_prompts = {
        "group_assumption": "judge_group_assumption",
        "bias_acknowledgement": "judge_bias_acknowledgement",
        "meta_reflection": "judge_meta_reflection",
        "outside_demo_knowledge": "judge_outside_demo_knowledge",
        "outside_topical_knowledge": "judge_outside_topical_knowledge",
        "unresolved": "judge_unresolved",
        "overthinking": "judge_overthinking",
        "missing_logic": "judge_missing_logic",
    }

    data = load_reasoning_data(args.data_path)
    if args.max_samples:
        data = data[:args.max_samples]

    pm = PromptManager(args.prompts)

    combined_results = {
        item["sample_id"]: {
            "sample_id": item["sample_id"],
            "category": item.get("bbq_category", ""),
            "example_id": item.get("example_id", ""),
            "model": item.get("model", ""),
            "prompt_type": item.get("prompt_type", ""),
            "judge_model": args.model,
            "reasoning_on": True,
            "reasoning_level": args.reasoning_level,
            "judge_output": {},
            "judge_explanations": {}
        } for item in data
    }

    for metric_name, prompt_key in metric_prompts.items():
        print(f"\n🧩 Evaluating metric: {metric_name} via {prompt_key}")
        messages_batch = build_batch_messages(data, pm, prompt_key)

        outputs = run_vllm_evaluation(
            model_name=args.model,
            messages_batch=messages_batch,
            temperature=args.temperature,
            max_tokens=2048,
            seed=args.seed,
            reasoning_level=args.reasoning_level,
            top_p=args.top_p
        )

        parsed = parse_outputs(data, outputs, args.model, prompt_key, args.reasoning_level)

        for item in parsed:
            sid = item["sample_id"]
            combined_results[sid]["judge_output"][metric_name] = \
                item["judge_output"][metric_name]
            combined_results[sid]["judge_explanations"][metric_name] = \
                item["judge_explanations"][metric_name]

    results = list(combined_results.values())
    params = {
        "max_tokens": 2048,
        "temperature": args.temperature,
        "seed": args.seed,
        "reasoning_level": args.reasoning_level,
        "top_p": args.top_p
    }
    save_results(results, args.model, args.output_dir, "multi_prompt_eval", params)


# ================================================================
# Main
# ================================================================
def main(args):
    if args.multi_prompt:
        run_multi_prompt_evaluation(args)
        return

    data = load_reasoning_data(args.data_path)
    if args.max_samples:
        data = data[:args.max_samples]

    pm = PromptManager(args.prompts)
    messages_batch = build_batch_messages(data, pm, args.prompt)

    outputs = run_vllm_evaluation(
        model_name=args.model,
        messages_batch=messages_batch,
        temperature=args.temperature,
        max_tokens=2048,
        seed=args.seed,
        reasoning_level=args.reasoning_level,
    )

    results = parse_outputs(data, outputs, args.model, args.prompt, args.reasoning_level)

    params = {
        "max_tokens": 2048,
        "temperature": args.temperature,
        "seed": args.seed,
        "reasoning_level": args.reasoning_level
    }
    save_results(results, args.model, args.output_dir, args.prompt, params)


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="reasoning_eval/data_to_label/sample_traces_initial.json")
    parser.add_argument("--prompts", type=str, default="reasoning_eval/judge_prompts")
    parser.add_argument("--output_dir", type=str, default="reasoning_eval/llm_judge_samples/")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi_prompt", action="store_true", help="Run in multi-prompt mode.")
    parser.add_argument(
        "--reasoning_level",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Set reasoning effort level (step-by-step thinking)"
    )
    parser.add_argument(
    "--top_p",
    type=float,
    default=1.0,
    help="Top-p (nucleus) sampling parameter for the model"
)

    args = parser.parse_args()

    main(args)
