import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from anthropic import Anthropic
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
    """Format prompt into Anthropic message structure."""
    return [{"role": "user", "content": prompt}]


def build_batch_messages(batch_data, pm, prompt_key):
    """Generate batch of chat-style messages for Claude."""
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
# Claude API inference
# ================================================================
def run_claude_evaluation(
    model_name: str,
    messages_batch: List[List[Dict[str, str]]],
    temperature: float = 0.6,
    max_tokens: int = 2048,
    seed: int = 42,
):
    """Run evaluation using Claude API."""
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    print(f"🚀 Calling Claude model: {model_name}")
    outputs = []

    for messages in tqdm(messages_batch, desc="Running Claude batches"):
        try:
            response = client.messages.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            outputs.append(response)
        except Exception as e:
            print(f"⚠️ Error on sample: {e}")
            outputs.append(None)
    return outputs


# ================================================================
# Parse and save
# ================================================================
import re

def parse_outputs(batch_data, outputs, model_name, prompt_key):
    """Convert Claude outputs to structured JSON format, handling code fences."""
    results = []
    code_fence_pattern = re.compile(r"```(?:json)?\n(.*?)```", re.DOTALL)

    for i, (item, output) in enumerate(zip(batch_data, outputs)):
        if not output:
            parsed = {"raw_text": None}
        else:
            text = output.content[0].text  # Claude output text

            # Strip code fences if present
            match = code_fence_pattern.search(text)
            if match:
                text = match.group(1).strip()

            # Try to parse as JSON
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = {"raw_text": text}

        results.append({
            "sample_id": item.get("sample_id", ""),
            "category": item.get("bbq_category", ""),
            "example_id": item.get("example_id", ""),
            "model": item.get("model", ""),
            "prompt_type": item.get("prompt_type", ""),
            "judge_model": model_name,
            "judge_prompt": prompt_key,
            "judge_output": parsed,
        })
    return results


def save_results(results, model_name, output_dir, prompt_key, params):
    """Save evaluation results + metadata."""
    os.makedirs(output_dir, exist_ok=True)
    output_data = {
        "metadata": {
            "judge_model": model_name,
            "judge_prompt": prompt_key,
            "sampling_params": params,
        },
        "results": results
    }

    filename = f"{output_dir}/llm_eval_{model_name.replace('/', '_')}_{prompt_key}.json"
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"✅ Saved {len(results)} results to {filename}")
    return filename


# ================================================================
# Main
# ================================================================
def main(args):
    data = load_reasoning_data(args.data_path)

    if args.max_samples:
        data = data[:args.max_samples]

    pm = PromptManager(args.prompts)
    messages_batch = build_batch_messages(data, pm, args.prompt)

    outputs = run_claude_evaluation(
        model_name=args.model,
        messages_batch=messages_batch,
        temperature=args.temperature,
        max_tokens=2048,
        seed=args.seed,
    )

    results = parse_outputs(data, outputs, args.model, args.prompt)
    params = {
        "max_tokens": 2048,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    save_results(results, args.model, args.output_dir, args.prompt, params)


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate reasoning traces with Claude API.")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-latest")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="reasoning_eval/data_to_label/sample_traces_inital.json")
    parser.add_argument("--prompts", type=str, default="reasoning_eval/judge_prompts")
    parser.add_argument("--output_dir", type=str, default="reasoning_eval/llm_judge_samples/")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
