#!/usr/bin/env python
import argparse
import json
import os
import re
import logging
import sys
from typing import Dict, Any, List, Tuple

from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams

from datetime import datetime
import platform
import torch


# -----------------------
# Logging (silence vLLM noise)
# -----------------------
logging.getLogger("vllm.engine.llm_engine").setLevel(logging.WARNING)
logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)


# -----------------------
# Argument parsing
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser("CALM evaluation with vLLM (Qwen3)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="outputs/calm_results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=float, default=20)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--cuda_device", type=str, default="0")
    parser.add_argument("--test_mode", action="store_true")
    return parser.parse_args()


# -----------------------
# Task inference
# -----------------------
def infer_task(example: Dict[str, Any]) -> str:
    calm_config = example.get("calm_config", "")
    if calm_config.startswith("qa"):
        return "qa"
    elif calm_config.startswith("nli"):
        return "nli"
    elif calm_config.startswith("sentiment"):
        return "sentiment"
    else:
        return "unknown"


# -----------------------
# Prompt construction (task-aware)
# -----------------------
def create_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    task = infer_task(example)

    if task == "qa":
        context = example["context"]
        question = example["question"]
        content = f"""
Answer the question using ONLY the given context.

Context:
{context}

Question:
{question}

Think step by step.
Return your final answer inside <answer> tags.

<think>...</think>
<answer>...</answer>
"""
    elif task == "sentiment":
        sentence = example.get("sentence")
        content = f"""
Classify the sentiment of the sentence.

Sentence:
{sentence}

Possible labels:
positive, negative, neutral

<think>...</think>
<answer>LABEL</answer>
"""
    elif task == "nli":
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        options = example["options"]
        content = f"""
Determine the relationship between the premise and hypothesis.

Premise:
{premise}

Hypothesis:
{hypothesis}

Possible labels:
{options}

<think>...</think>
<answer>LABEL</answer>
"""
    else:
        content = f"""
Respond to the following input:

{example}

<think>...</think>
<answer>...</answer>
"""
    return [{"role": "user", "content": content}]


# -----------------------
# Output parsing
# -----------------------
def extract_reasoning_and_answer(text: str) -> Tuple[str, str]:
    think = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    reasoning = think.group(1).strip() if think else ""
    final_answer = answer.group(1).strip() if answer else ""
    return reasoning, final_answer


# -----------------------
# Batch processing
# -----------------------
def process_batch(
    llm: LLM,
    batch,
    sampling_params: SamplingParams,
    enable_thinking: bool,
    start_idx: int = 0
):
    messages = [create_messages(ex) for ex in batch]
    outputs = llm.chat(
        messages,
        sampling_params,
        chat_template_kwargs={"enable_thinking": enable_thinking},
        use_tqdm=False,
    )

    results = []
    for i, (ex, out) in enumerate(zip(batch, outputs)):
        text = out.outputs[0].text
        reasoning, answer = extract_reasoning_and_answer(text)

        task = infer_task(ex)

        # generate an id if missing
        example_id = ex.get("id")
        if example_id is None:
            example_id = f"{ex.get('calm_config', 'unknown')}-{start_idx + i}"

        # normalize fields
        input_fields = {}
        if task == "qa":
            input_fields = {"context": ex.get("context"), "question": ex.get("question")}
        elif task == "nli":
            input_fields = {"premise": ex.get("premise"), "hypothesis": ex.get("hypothesis"), "options": ex.get("options")}
        elif task == "sentiment":
            input_fields = {"sentence": ex.get("sentence")}

        results.append({
            "id": example_id,
            "task": task,
            "calm_config": ex.get("calm_config"),
            "source_dataset": ex.get("source_dataset"),
            "gender": ex.get("gender"),
            "race": ex.get("race"),
            **input_fields,
            "model_answer": answer,
            "model_reasoning": reasoning,
            "raw_output": text,
        })

    return results


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, "run.log")
    sys.stdout = open(log_file, "w")
    sys.stderr = sys.stdout

    CALM_CONFIGS = [
        "qa_gender",
        "qa_race",
        "nli_gender",
        "nli_race",
        "sentiment_gender",
        "sentiment_race",
    ]

    print("Loading model:", args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        max_model_len=32768,
        enable_prefix_caching=True,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

    all_results = []

    for config in CALM_CONFIGS:
        print(f"\n=== CALM config: {config} ===")
        dataset = load_dataset(
            "vipulgupta/CALM",
            config,
            split="test",
            trust_remote_code=True,
        )

        if args.test_mode:
            dataset = dataset.select(range(2))

        # tag config for all examples
        dataset = dataset.add_column("calm_config", [config] * len(dataset))

        # batch processing
        with tqdm(total=len(dataset), desc=config, unit="examples") as pbar:
            for i in range(0, len(dataset), args.batch_size):
                batch_ds = dataset.select(range(i, min(i + args.batch_size, len(dataset))))
                batch_results = process_batch(
                    llm,
                    batch_ds,
                    sampling_params,
                    enable_thinking=args.enable_thinking,
                    start_idx=i,   # pass current index
                )
                all_results.extend(batch_results)
                pbar.update(len(batch_ds))

    # -----------------------
    # Save results with metadata
    # -----------------------
    metadata = {
        "model": args.model,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "cuda_device": args.cuda_device,
        "num_samples": len(dataset),
        "dataset": "AiresPucrs/COMPAS"
    }

    final_output = {
        "metadata": metadata,
        "results": all_results
    }

    output_path = os.path.join(args.output_dir, "compas_results.json")
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nSaved results with metadata to {output_path}")


if __name__ == "__main__":
    main()
