#!/usr/bin/env python
import os
import json
import argparse
import re
import glob
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI
from prompt_manager import PromptManager

# ================================================================
# Utility functions
# ================================================================
def resolve_input_files(data_paths):
    files = []
    for p in data_paths:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "*.json")))
        else:
            files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        raise ValueError("❌ No input JSON files found.")
    return files


def load_reasoning_data(path: str):
    with open(path, "r") as f:
        raw = json.load(f)

    # Handle merged BBQ-style files
    if isinstance(raw, dict) and "results" in raw:
        data = raw["results"]
        metadata = raw.get("metadata", {})
    else:
        data = raw
        metadata = {}

    print(f"✅ Loaded {len(data)} samples from {path}")
    return data, metadata

def infer_prompt_type_from_path(path: str) -> str:
    """
    Infer prompt_type from a path by taking the part immediately before 'prompt' 
    and appending '_prompt'.
    
    Examples:
        'qwen_full_8B_full_prompt.json' -> 'full_prompt'
        'qwen_simple_prompt_v2.json' -> 'simple_prompt'
    """
    filename = os.path.basename(path)
    stem = os.path.splitext(filename)[0]

    parts = stem.split("_")

    for i, part in enumerate(parts):
        if "prompt" in part.lower():
            if i > 0:
                return f"{parts[i-1]}_prompt"
            else:
                return "prompt"  # fallback if it's the first part
    return "unknown"


def normalize_sample(item, metadata, idx, path=None):
    sentences = item.get("sentences", [])

    letter_to_idx = {"A": 0, "B": 1, "C": 2}
    chosen_idx = letter_to_idx.get(item.get("model_answer_letter"))

    chosen_sentence = (
        sentences[chosen_idx] if chosen_idx is not None and chosen_idx < len(sentences) else None
    )

    return {
        # IDs
        "sample_id": item.get("id", f"stereoset_{idx}"),
        "example_id": item.get("id", idx),

        # StereoSet metadata
        "task_type": item.get("task_type", ""),
        "bias_type": item.get("bias_type", ""),
        "target": item.get("target", ""),

        # Prompt content
        "context": item.get("context", ""),
        "question": "Choose the most appropriate completion.",

        # Options
        "answer_options": sentences,
        "option_labels": item.get("gold_labels"),  # list aligned with sentences

        # Model outputs
        "model_reasoning": item.get("model_reasoning", ""),
        "model_answer": chosen_sentence,
        "model_answer_letter": item.get("model_answer_letter"),

        # StereoSet labels
        "gold_labels": item.get("gold_labels"),
        "chosen_gold_label": item.get("chosen_gold_label"),
        "chosen_label_name": item.get("chosen_label_name"),
        "bias_category": item.get("bias_category"),

        # Metadata
        "model": metadata.get("run_metadata", {}).get("model", ""),
        "prompt_type": infer_prompt_type_from_path(path) if path else "unknown",
    }


def create_messages(prompt: str) -> List[Dict[str, str]]:
    """Format prompt into OpenAI/Anthropic chat structure."""
    return [{"role": "user", "content": prompt}]


def build_batch_messages(batch_data, pm, prompt_key, reasoning_prompt_text=None):
    """Generate batch of chat messages with reasoning context."""
    messages_batch = []
    for item in batch_data:
        prompt = pm.get_prompt(
            prompt_key,
            reasoning_trace=item.get("model_reasoning", ""),
            final_answer=item.get("model_answer", ""),
            category=item.get("bias_type", ""),
            context=item.get("context", ""),
            question=item.get("question", ""),
            answer_options=item.get("answer_options", []),
            sample_id=item.get("sample_id", ""),
            example_id=item.get("example_id", ""),
            model=item.get("model", ""),
            prompt_type=item.get("prompt_type", "")
        )

         # Prepend reasoning text if provided
        if reasoning_prompt_text:
            prompt = f"{prompt}\n{reasoning_prompt_text}"

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
    Calls the OpenAI Responses API for GPT models.
    For other models, uses Chat Completions API.
    """
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL")
    )

    outputs = []
    print(f"🚀 Calling model via flexible vLLM client: {model_name}")

    for messages in tqdm(messages_batch, desc="Inference batches"):
        if "gpt" in model_name.lower():
            try:
                response = client.responses.create(
                    model=model_name,
                    input=[{
                    "role": "user",
                    "content": [{"type": "text", "text": messages[0]["content"]}]
                }],
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    reasoning={"effort": reasoning_level},
                    top_p=top_p  
                )
                outputs.append(response)
            except Exception as e:
                print(f"⚠️ GPT Responses API failed: {e}")
                outputs.append(None)
        else:
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
                print(f"⚠️ Chat Completions API failed: {e}")
                outputs.append(None)

    return outputs


# ================================================================
# Parse Outputs
# ================================================================
def parse_outputs(batch_data, outputs, model_name, prompt_key, reasoning_level="medium"):
    results = []
    score_pattern = re.compile(r"Score\s*:\s*(\d+)", re.IGNORECASE)

    for item, output in zip(batch_data, outputs):
        text = None

        # Responses API
        if hasattr(output, "output") and output.output:
            chunks = []
            for seg in output.output:
                for c in getattr(seg, "content", []):
                    if hasattr(c, "text"):
                        chunks.append(c.text)
            text = "\n".join(chunks)

        # ChatCompletion API
        elif hasattr(output, "choices"):
            try:
                text = output.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️ Could not extract text for sample_id {item.get('sample_id')}: {e}")

        # vLLM dict output
        elif isinstance(output, dict) and "text" in output:
            text = output["text"].strip()

        # vLLM string output
        elif isinstance(output, str):
            text = output.strip()

        # fallback: if no text, try reasoning segment
        if not text and hasattr(output, "output") and output.output:
            reasoning_segments = []
            for seg in output.output:
                if getattr(seg, "type", None) == "reasoning":
                    for c in getattr(seg, "content", []):
                        if hasattr(c, "text") and c.text:
                            reasoning_segments.append(c.text)
            if reasoning_segments:
                text = "\n".join(reasoning_segments)

        if not text:
            print(f"⚠️ Could not parse output for sample_id {item.get('sample_id')}")
            results.append(_empty_parse(item, model_name, prompt_key, reasoning_level))
            continue



        raw_output = text
        reasoning_text = text
        

        # Extract first JSON block at the start
        judge_output = None
        # Find the first opening brace
        start_idx = text.find("{")
        if start_idx != -1:
            brace_count = 0
            for i, c in enumerate(text[start_idx:], start=start_idx):
                if c == "{":
                    brace_count += 1
                elif c == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        # Candidate JSON block
                        json_str = text[start_idx:i+1]
                        try:
                            judge_output = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            print(f"⚠️ JSON parse error for sample_id {item.get('sample_id')}: {e}")
                        break

        # Fallback: extract Score: N if no JSON
        if judge_output is None:
            score_match = score_pattern.search(text)
            if score_match:
                try:
                    judge_output = {"score": int(score_match.group(1))}
                except Exception:
                    pass

        result = {
            "sample_id": item.get("sample_id", ""),
            "category": item.get("bias_type", ""),
            "example_id": item.get("example_id", ""),
            "model": item.get("model", ""),
            "prompt_type": item.get("prompt_type", ""),
            "judge_model": model_name,
            "judge_prompt": prompt_key,
            "raw_output": raw_output,
            "judge_output": judge_output,
            "judge_explanations": reasoning_text,


            "stereoset_choice": item.get("chosen_label_name"),
            "stereoset_choice_id": item.get("chosen_gold_label"),
            "bias_category": item.get("bias_category"),
        }

        if "gpt" in model_name.lower():
            result["reasoning_level"] = reasoning_level

        results.append(result)

    return results


def _empty_parse(item, model_name, prompt_key, reasoning_level=None):
    """Return default structure for empty output."""
    result = {
        "sample_id": item.get("sample_id", ""),
        "category": item.get("bias_type", ""),
        "example_id": item.get("example_id", ""),
        "model": item.get("model", ""),
        "prompt_type": item.get("prompt_type", ""),
        "judge_model": model_name,
        "judge_prompt": prompt_key,
        "raw_output": None,
        "judge_output": None,
        "judge_explanations": None,


        # carry through original input evaluation flags
        "stereoset_choice": item.get("chosen_label_name"),
        "stereoset_choice_id": item.get("chosen_gold_label"),
        "bias_category": item.get("bias_category"),
    }

    if "gpt" in model_name.lower():
        result["reasoning_level"] = reasoning_level

    return result




# ================================================================
# Save Results
# ================================================================
def save_results(results, model_name, output_dir, prompt_key, params, extra_metadata=None):
    os.makedirs(output_dir, exist_ok=True)

    is_gpt_model = "gpt" in model_name.lower()

    # Make a copy of params for sampling_params
    sampling_params = params.copy()

    # Remove reasoning_level for non-GPT models
    if not is_gpt_model and "reasoning_level" in sampling_params:
        sampling_params.pop("reasoning_level")

    metadata = {
        "judge_model": model_name,
        "judge_prompt": prompt_key,
        "sampling_params": sampling_params
    }

    if params.get("reasoning_prompt_text"):
        metadata["reasoning_prompt_used"] = True
        metadata["reasoning_prompt_text"] = params["reasoning_prompt_text"]

    # Only include reasoning_level at top-level metadata if GPT model
    if is_gpt_model and "reasoning_level" in params:
        metadata["reasoning_level"] = params["reasoning_level"]

    if extra_metadata:
        metadata.update(extra_metadata)

    output_data = {
        "metadata": metadata,
        "results": results
    }

    # Encode key sampling params in filename
    filename_parts = ["llm_eval"]

    # Then model, prompt, and sampling params
    filename_parts.extend([
        model_name.replace("/", "_"),
        prompt_key,
        f"temp{params.get('temperature', 0.6)}",
        f"top_p{params.get('top_p', 1.0)}",
        f"seed{params.get('seed', 42)}",
        f"max_tokens{params.get('max_tokens', 2048)}"
    ])

    # Only append reasoning_level to filename if GPT model
    if is_gpt_model and "reasoning_level" in params:
        filename_parts.append(params["reasoning_level"])

    if params.get("reasoning_prompt_text"):
        filename_parts.append("reasoning")


    filename = os.path.join(output_dir, "_".join(filename_parts) + ".json")

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

    data, metadata = load_reasoning_data(args.data_path)
    if args.max_samples:
        data = data[:args.max_samples]

    pm = PromptManager(args.prompts)

    combined_results = {
        item["sample_id"]: {
            "sample_id": item["sample_id"],
            "category": item.get("bias_type", ""),
            "example_id": item.get("example_id", ""),
            "model": item.get("model", ""),
            "prompt_type": item.get("prompt_type", ""),
            "judge_model": args.model,
            "reasoning_level": args.reasoning_level,
            "judge_output": {},
            "judge_explanations": {}
        } for item in data
    }

    for metric_name, prompt_key in metric_prompts.items():
        print(f"\n🧩 Evaluating metric: {metric_name} via {prompt_key}")

        messages_batch = build_batch_messages(
            data, 
            pm, 
            prompt_key, 
            reasoning_prompt_text=args.reasoning_prompt_text
        )

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
            
            combined_results[sid]["raw_output"] = combined_results[sid].get("raw_output", {})
            combined_results[sid]["raw_output"][metric_name] = item["raw_output"]

            combined_results[sid]["judge_output"][metric_name] = item["judge_output"]
            combined_results[sid]["judge_explanations"][metric_name] = item["judge_explanations"]

    results = list(combined_results.values())

    params = {
        "max_tokens": 2048,
        "temperature": args.temperature,
        "seed": args.seed,
        "top_p": args.top_p,
        "reasoning_prompt_text": args.reasoning_prompt_text
    }


    # Only add reasoning_level if GPT model
    if "gpt" in args.model.lower():
        params["reasoning_level"] = args.reasoning_level

    save_results(results, args.model, args.output_dir, "multi_prompt_eval", params)

# ================================================================
# Main
# ================================================================
def main(args):

    pm = PromptManager(args.prompts)

    if args.data_paths:
        input_files = resolve_input_files(args.data_paths)
    else:
        input_files = [args.data_path]

    for data_path in input_files:
        print(f"\n📂 Processing: {data_path}")

        raw_data, metadata = load_reasoning_data(data_path)

        data = [
        normalize_sample(item, metadata, idx, path=data_path)
            for idx, item in enumerate(raw_data)
        ]

        if args.max_samples:
            data = data[:args.max_samples]

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
            max_tokens=2048,
            seed=args.seed,
            reasoning_level=args.reasoning_level,
            top_p=args.top_p
        )

        results = parse_outputs(
            data, outputs, args.model, args.prompt, args.reasoning_level
        )

        # 🔑 derive output name from input file
        input_stem = os.path.splitext(os.path.basename(data_path))[0]

        # extract everything after "bbq_" and before "_results"
        if input_stem.startswith("bbq_") and "_results" in input_stem:
            category_from_filename = input_stem[len("bbq_"):input_stem.find("_results")]
        else:
            category_from_filename = "unknown"



        params = {
            "max_tokens": 2048,
            "temperature": args.temperature,
            "seed": args.seed,
            "top_p": args.top_p,
            "reasoning_prompt_text": args.reasoning_prompt_text
        }

        if "gpt" in args.model.lower():
            params["reasoning_level"] = args.reasoning_level

        save_results(
            results,
            args.model,
            args.output_dir,
            prompt_key=f"{args.prompt}",
            params=params,
            extra_metadata={
                "input_file": data_path,
                "input_stem": input_stem,
                "bias_type": category_from_filename
            }
        )


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompts", type=str, default="reasoning_eval/judge_prompts")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--multi_prompt", action="store_true", help="Run in multi-prompt mode.")
    parser.add_argument("--data_path", type=str, default="reasoning_eval/data_to_label/sample_traces_initial.json")
    parser.add_argument("--output_dir", type=str, default="reasoning_eval/llm_judge_samples/")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling parameter for the model")
    parser.add_argument(
        "--reasoning_level",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Set reasoning effort level (step-by-step thinking) - only used for gpt models otherwise ignore "
    )
    parser.add_argument(
    "--reasoning_prompt_text",
    type=str,
    default=None,
    help="Text to prepend to the prompt for step-by-step reasoning"
    )

    parser.add_argument(
    "--multi_prompt_keys",
    type=str,
    default=None,
    help="Comma-separated list of prompt keys for multi-prompt evaluation"
    )

    parser.add_argument(
    "--data_paths",
    type=str,
    nargs="+",
    help="One or more input JSON files or glob patterns"
)


    args = parser.parse_args()

    main(args)
