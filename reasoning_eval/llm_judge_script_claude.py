import os
import json
import argparse
import re
from typing import List, Dict, Any
from tqdm import tqdm
from anthropic import Anthropic
from prompt_manager import PromptManager

# ================================================================
# Utility Functions
# ================================================================
def load_reasoning_data(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} samples from {path}")
    return data


def create_messages(prompt: str):
    return [{"role": "user", "content": prompt}]


def build_batch_messages(batch_data, pm, prompt_key, reasoning_prompt_text=None):
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

        # Prepend optional reasoning text (aligned with OpenAI script)
        if reasoning_prompt_text:
            prompt = f"{prompt}\n{reasoning_prompt_text}"

        messages_batch.append(create_messages(prompt))

    return messages_batch


# ================================================================
# Claude Inference
# ================================================================
def run_claude_evaluation(
    model_name,
    messages_batch,
    temperature=0.6,
    max_tokens=2048,
    seed=42
):
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    outputs = []

    print(f"🚀 Calling Claude model: {model_name}")

    for messages in tqdm(messages_batch, desc="Claude batches"):
        try:
            response = client.messages.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            outputs.append(response)
        except Exception as e:
            print(f"⚠️ Claude API error: {e}")
            outputs.append(None)

    return outputs


# ================================================================
# Parse Outputs (aligned with OpenAI parser)
# ================================================================
def parse_outputs(batch_data, outputs, model_name, prompt_key):
    results = []
    score_pattern = re.compile(r"Score\s*:\s*(\d+)", re.IGNORECASE)

    for item, output in zip(batch_data, outputs):
        if not output:
            text = None
        else:
            try:
                text = output.content[0].text.strip()
            except Exception:
                text = None

        if not text:
            print(f"⚠️ Could not parse Claude output for sample_id {item.get('sample_id')}")
            results.append(_empty_parse(item, model_name, prompt_key))
            continue

        raw_output = reasoning_text = text

        # -----------------------------------------------------------
        # Extract JSON block anywhere in string (same logic as vLLM)
        # -----------------------------------------------------------
        judge_output = None
        start_idx = text.find("{")
        if start_idx != -1:
            brace_count = 0
            for i, c in enumerate(text[start_idx:], start=start_idx):
                if c == "{":
                    brace_count += 1
                elif c == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[start_idx:i+1]
                        try:
                            judge_output = json.loads(json_str)
                        except Exception as e:
                            print(f"⚠️ JSON parse error sample_id {item.get('sample_id')}: {e}")
                        break

        # Fallback: Score: N
        if judge_output is None:
            match = score_pattern.search(text)
            if match:
                judge_output = {"score": int(match.group(1))}

        results.append({
            "sample_id": item.get("sample_id", ""),
            "category": item.get("bbq_category", ""),
            "example_id": item.get("example_id", ""),
            "model": item.get("model", ""),
            "prompt_type": item.get("prompt_type", ""),
            "judge_model": model_name,
            "judge_prompt": prompt_key,
            "raw_output": raw_output,
            "judge_output": judge_output,
            "judge_explanations": reasoning_text,
        })

    return results


def _empty_parse(item, model_name, prompt_key):
    return {
        "sample_id": item.get("sample_id", ""),
        "category": item.get("bbq_category", ""),
        "example_id": item.get("example_id", ""),
        "model": item.get("model", ""),
        "prompt_type": item.get("prompt_type", ""),
        "judge_model": model_name,
        "judge_prompt": prompt_key,
        "raw_output": None,
        "judge_output": None,
        "judge_explanations": None,
    }


# ================================================================
# Save Results (aligned with OpenAI script)
# ================================================================
def save_results(results, model_name, output_dir, prompt_key, params, extra_metadata=None):
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "judge_model": model_name,
        "judge_prompt": prompt_key,
        "sampling_params": params,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    output_data = {
        "metadata": metadata,
        "results": results
    }


    # Encode key sampling params in filename
    filename_parts = [
        "llm_eval",
        model_name.replace("/", "_"),
        prompt_key,
        f"temp{params.get('temperature', 0.6)}",
        f"seed{params.get('seed', 42)}",
        f"max_tokens{params.get('max_tokens', 2048)}"
    ]

    if params.get("reasoning_prompt_text"):
        filename_parts.append("reasoning")

    filename = os.path.join(output_dir, "_".join(filename_parts) + ".json")

    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Saved {len(results)} results → {filename}")
    return filename


# ================================================================
# Multi-Prompt Mode (aligned with OpenAI script)
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

    combined = {
        d["sample_id"]: {
            "sample_id": d["sample_id"],
            "category": d.get("bbq_category", ""),
            "example_id": d.get("example_id", ""),
            "model": d.get("model", ""),
            "prompt_type": d.get("prompt_type", ""),
            "judge_model": args.model,
            "judge_output": {},
            "judge_explanations": {}
        }
        for d in data
    }

    for metric_name, prompt_key in metric_prompts.items():
        print(f"\n🧩 Evaluating metric: {metric_name}")

        messages_batch = build_batch_messages(
            data, pm, prompt_key, args.reasoning_prompt_text
        )

        outputs = run_claude_evaluation(
            args.model, messages_batch,
            temperature=args.temperature,
            max_tokens=2048,
            seed=args.seed
        )

        parsed = parse_outputs(data, outputs, args.model, prompt_key)

        for item in parsed:
            sid = item["sample_id"]

            combined[sid]["raw_output"] = item["raw_output"]
            combined[sid]["judge_output"][metric_name] = (
                item["judge_output"].get(metric_name)
                if isinstance(item["judge_output"], dict)
                else item["judge_output"]
            )
            combined[sid]["judge_explanations"][metric_name] = item["judge_explanations"]

    results = list(combined.values())

    params = {
        "max_tokens": 2048,
        "temperature": args.temperature,
        "seed": args.seed,
        "reasoning_prompt_text": args.reasoning_prompt_text
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

    messages_batch = build_batch_messages(
        data, pm, args.prompt, args.reasoning_prompt_text
    )

    outputs = run_claude_evaluation(
        args.model, messages_batch,
        temperature=args.temperature,
        max_tokens=2048,
        seed=args.seed
    )

    results = parse_outputs(data, outputs, args.model, args.prompt)

    params = {
        "max_tokens": 2048,
        "temperature": args.temperature,
        "seed": args.seed,
        "reasoning_prompt_text": args.reasoning_prompt_text
    }

    save_results(results, args.model, args.output_dir, args.prompt, params)


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-latest")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--prompts", type=str, default="reasoning_eval/judge_prompts")
    parser.add_argument("--data_path", type=str, default="reasoning_eval/data_to_label/sample_traces_initial.json")
    parser.add_argument("--output_dir", type=str, default="reasoning_eval/llm_judge_samples/")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--multi_prompt", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reasoning_prompt_text", type=str, default=None)

    args = parser.parse_args()
    main(args)
