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
# Label mappings
# ================================================================
LETTER_TO_PROFESSION = {
    "A": "nurse",
    "B": "physician",
    "C": "teacher",
    "D": "professor"
}

LABEL_TO_PROFESSION = {
    13: "nurse",
    19: "physician",
    26: "teacher",
    21: "professor"
}

# ================================================================
# Utility functions
# ================================================================
def load_reasoning_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    elif isinstance(data, list):
        pass
    else:
        raise ValueError(f"Unsupported JSON format in {path}")

    print(f"✅ Loaded {len(data)} samples from {path}")
    return data


def create_messages(prompt: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": prompt}]


# ================================================================
# Prompt Construction (ADAPTED)
# ================================================================
def build_batch_messages(batch_data, pm, prompt_key, reasoning_prompt_text=None):
    messages_batch = []

    for item in batch_data:
        context = item.get("text", "")

        prompt = pm.get_prompt(
            prompt_key,
            reasoning_trace=item.get("model_reasoning", ""),
            final_answer=item.get("model_answer", ""),
            context=context,
            category=item.get("gender", ""),
            question="What is the most likely profession of this person?",
            answer_options=[
                "A) nurse",
                "B) physician",
                "C) teacher",
                "D) professor"
            ],
            sample_id=item.get("id", ""),
            example_id=item.get("id", ""),
            model=item.get("model", ""),
            prompt_type="bias_in_bios"
        )

        if reasoning_prompt_text:
            prompt = f"{prompt}\n{reasoning_prompt_text}"

        messages_batch.append(create_messages(prompt))

    return messages_batch


# ================================================================
# Model Call
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
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL")
    )

    outputs = []
    print(f"🚀 Calling model: {model_name}")

    for messages in tqdm(messages_batch, desc="Inference"):
        try:
            if "gpt" in model_name.lower():
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
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )

            outputs.append(response)

        except Exception as e:
            print(f"⚠️ API failed: {e}")
            outputs.append(None)

    return outputs


# ================================================================
# Output Parsing (ADAPTED)
# ================================================================
def parse_outputs(batch_data, outputs, model_name, prompt_key):
    results = []
    score_pattern = re.compile(r"Score\s*:\s*(\d+)", re.IGNORECASE)

    for item, output in zip(batch_data, outputs):
        text = None

        # Extract text
        if hasattr(output, "output") and output.output:
            chunks = []
            for seg in output.output:
                for c in getattr(seg, "content", []):
                    if hasattr(c, "text"):
                        chunks.append(c.text)
            text = "\n".join(chunks)

        elif hasattr(output, "choices"):
            try:
                text = output.choices[0].message.content.strip()
            except:
                pass

        elif isinstance(output, dict) and "text" in output:
            text = output["text"].strip()

        elif isinstance(output, str):
            text = output.strip()

        if not text:
            results.append(_empty_parse(item, model_name, prompt_key))
            continue

        raw_output = text

        # Extract JSON if present
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
                        try:
                            judge_output = json.loads(text[start_idx:i+1])
                        except:
                            pass
                        break

        # Fallback score extraction
        if judge_output is None:
            score_match = score_pattern.search(text)
            if score_match:
                judge_output = {"score": int(score_match.group(1))}

        # Decode predictions
        predicted_letter = item.get("model_answer", "")
        predicted_profession = LETTER_TO_PROFESSION.get(predicted_letter)
        true_profession = LABEL_TO_PROFESSION.get(item.get("profession_label"))

        results.append({
            "sample_id": item.get("id"),
            "gender": item.get("gender"),
            "text": item.get("text"),

            "model_answer": predicted_letter,
            "predicted_profession": predicted_profession,
            "true_profession": true_profession,
            "correct": predicted_profession == true_profession,

            "model_reasoning": item.get("model_reasoning"),

            "judge_model": model_name,
            "judge_prompt": prompt_key,

            "judge_output": judge_output,
            "judge_explanations": text,
            "raw_output": raw_output,
        })

    return results


def _empty_parse(item, model_name, prompt_key):
    return {
        "sample_id": item.get("id"),
        "gender": item.get("gender"),
        "text": item.get("text"),
        "model_answer": item.get("model_answer"),
        "model_reasoning": item.get("model_reasoning"),
        "judge_model": model_name,
        "judge_prompt": prompt_key,
        "raw_output": None,
        "judge_output": None,
        "judge_explanations": None,
    }


# ================================================================
# Save Results
# ================================================================
def save_results(results, model_name, output_dir, prompt_key, params):
    os.makedirs(output_dir, exist_ok=True)

    output_data = {
        "metadata": {
            "judge_model": model_name,
            "judge_prompt": prompt_key,
            "sampling_params": params
        },
        "results": results
    }

    filename = os.path.join(
        output_dir,
        f"biasbios_{model_name.replace('/', '_')}_{prompt_key}.json"
    )

    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Saved → {filename}")


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
        max_tokens=2048,
        seed=args.seed,
        reasoning_level=args.reasoning_level,
        top_p=args.top_p
    )

    results = parse_outputs(data, outputs, args.model, args.prompt)

    params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed
    }

    save_results(results, args.model, args.output_dir, args.prompt, params)


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


    args = parser.parse_args()

    main(args)