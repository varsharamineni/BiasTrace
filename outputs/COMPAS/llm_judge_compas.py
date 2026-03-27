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
# Load COMPAS data
# ================================================================
def load_reasoning_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)

    for i, item in enumerate(data):
        if "sample_id" not in item:
            item["sample_id"] = item.get("id", f"compas-{i}")

        item["category"] = item.get("race", "")
        item["prompt_type"] = "compas"

    print(f"✅ Loaded {len(data)} samples from {path}")
    return data


# ================================================================
# Build messages (COMPAS-specific)
# ================================================================
def build_batch_messages(batch_data, pm, prompt_key, reasoning_prompt_text=None):
    messages_batch = []

    EXCLUDE_KEYS = {
        "is_recid", "is_violent_recid", "event",
        "decile_score", "decile_score.1",
        "score_text", "v_decile_score", "v_score_text",
        "r_charge_degree", "r_days_from_arrest",
        "r_offense_date", "r_charge_desc", "r_jail_in",
        "name", "first", "last", "dob"
    }

    for item in batch_data:
        features = item.get("features", {})
        features = {k: v for k, v in features.items() if k not in EXCLUDE_KEYS}

        feature_str = "\n".join([f"{k}: {v}" for k, v in features.items()])

        prompt = pm.get_prompt(
            prompt_key,
            reasoning_trace=item.get("model_reasoning", ""),
            final_answer=item.get("model_answer", ""),
            category=item.get("race", ""),
            context=feature_str,
            sample_id=item.get("sample_id", ""),
            example_id=item.get("sample_id", ""),
            model=item.get("model", ""),
            prompt_type="compas"
        )

        if reasoning_prompt_text:
            prompt = f"{prompt}\n{reasoning_prompt_text}"

        messages_batch.append([{"role": "user", "content": prompt}])

    return messages_batch


# ================================================================
# Model inference
# ================================================================
def run_vllm_evaluation(
    model_name: str,
    messages_batch: List[List[Dict[str, str]]],
    temperature: float = 0.6,
    max_tokens: int = 2048,
    seed: int = 42,
    top_p: float = 1.0
):
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL")
    )

    outputs = []
    print(f"🚀 Running judge model: {model_name}")

    for messages in tqdm(messages_batch, desc="Inference"):
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
            print(f"⚠️ Error: {e}")
            outputs.append(None)

    return outputs


# ================================================================
# Parse outputs
# ================================================================
def parse_outputs(batch_data, outputs, model_name, prompt_key):
    results = []

    for item, output in zip(batch_data, outputs):
        text = None

        if output and hasattr(output, "choices"):
            try:
                text = output.choices[0].message.content.strip()
            except Exception:
                pass

        if not text:
            results.append({
                "sample_id": item.get("sample_id"),
                "judge_model": model_name,
                "judge_prompt": prompt_key,
                "judge_output": None,
                "judge_explanations": None,
                "raw_output": None
            })
            continue

        # Try parse JSON
        judge_output = None
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != -1:
                judge_output = json.loads(text[start:end])
        except Exception:
            pass

        results.append({
            "sample_id": item.get("sample_id"),
            "race": item.get("race"),
            "sex": item.get("sex"),
            "model_answer": item.get("model_answer"),
            "judge_model": model_name,
            "judge_prompt": prompt_key,
            "judge_output": judge_output,
            "judge_explanations": text,
            "raw_output": text
        })

    return results


# ================================================================
# Save results (WITH METADATA)
# ================================================================
def save_results(results, model_name, output_dir, prompt_key, params):
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "judge_model": model_name,
        "judge_prompt": prompt_key,
        "sampling_params": params,
        "num_samples": len(results)
    }

    output_data = {
        "metadata": metadata,
        "results": results
    }

    filename = os.path.join(
        output_dir,
        f"compas_judge_{model_name.replace('/', '_')}_{prompt_key}.json"
    )

    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Saved {len(results)} results → {filename}")


# ================================================================
# Main
# ================================================================
def main(args):
    data = load_reasoning_data(args.data_path)

    if args.max_samples:
        data = data[:args.max_samples]

    pm = PromptManager(args.prompts)

    messages_batch = build_batch_messages(
        data,
        pm,
        args.prompt,
        reasoning_prompt_text=args.reasoning_prompt_text
    )

    outputs = run_vllm_evaluation(
        model_name=args.model,
        messages_batch=messages_batch,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        top_p=args.top_p
    )

    results = parse_outputs(
        data,
        outputs,
        args.model,
        args.prompt
    )

    params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "reasoning_prompt_text": args.reasoning_prompt_text
    }

    save_results(
        results,
        args.model,
        args.output_dir,
        args.prompt,
        params
    )


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)

    # 👇 SAME AS YOUR ORIGINAL PIPELINE
    parser.add_argument("--prompts", type=str, default="reasoning_eval/judge_prompts")
    parser.add_argument("--prompt", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default="compas_judge_results/")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--reasoning_prompt_text",
        type=str,
        default=None,
        help="Optional reasoning prefix"
    )

    args = parser.parse_args()
    main(args)