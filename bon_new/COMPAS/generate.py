#!/usr/bin/env python
"""Stage 1/3: Best-of-N COMPAS generation with Qwen3 thinking mode. NO judge.

Generation-only fork of generate_compas_outputs_vllm_bon.py: every question
generates --best_of_n candidates which are ALL stored (with parsed reasoning
and answers) so that judging and selection can happen later, separately:

    stage 1: generate_compas_bon.py          (this script — GPU, no API key)
    stage 2: judge_compas_candidates.py      (scores candidates, resumable)
    stage 3: compare_compas_methods.py       (selection + accuracy/EO/EOpp)

Because no judge runs here, the row-level `model_answer` is the no-judge
default — the majority vote over all N candidates ("A"/"B" letter form for
the analysis pipeline, text in model_answer_text). Judge-filtered answers are
added per judge by stage 3 without modifying this file's outputs.

Dataset conventions are unchanged: balanced AA/Caucasian male samples from
AiresPucrs/COMPAS, is_recid stored per row (features + recid_label), capped
runs shrink the per-group sample so both races stay represented.

Example (smoke test):
    python scripts/generate_compas_bon.py \
        --model Qwen/Qwen3-1.7B \
        --output_dir outputs/compas_bon_smoke \
        --test_mode --best_of_n 4
"""
import argparse
import datetime
import json
import os
import re
from typing import Tuple, List, Dict, Any, Optional

from tqdm import tqdm
import logging

logging.getLogger("vllm.engine.llm_engine").setLevel(logging.WARNING)
logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)

from vllm import LLM, SamplingParams
from datasets import load_dataset, concatenate_datasets

ANSWER_OPTIONS = ["High risk", "Low risk"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Best-of-N COMPAS generation (no judge; candidates saved for later judging)"
    )
    parser.add_argument("--model", "--model_path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="../outputs/compas_results_bon")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Defendants per vLLM call (each yields best_of_n candidates)")
    parser.add_argument("--sample_size", type=int, default=250,
                        help="Defendants sampled per race group (default: 250 each)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Cap on total examples processed (default: all sampled)")
    parser.add_argument("--test_mode", action="store_true",
                        help="Test mode: process only 10 samples total")
    parser.add_argument("--best_of_n", "-N", type=int, default=8,
                        help="Number of candidates generated per defendant")
    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Dataset (identical to the joint script)
# --------------------------------------------------------------------------- #
def load_compas_dataset(seed: int = 42, sample_size: int = 250):
    dataset = load_dataset("AiresPucrs/COMPAS", split="train")
    aa_males = dataset.filter(lambda x: x["race"] == "African-American" and x["sex"] == "Male")
    aa_sample = aa_males.shuffle(seed=seed).select(range(min(len(aa_males), sample_size)))
    cauc_males = dataset.filter(lambda x: x["race"] == "Caucasian" and x["sex"] == "Male")
    cauc_sample = cauc_males.shuffle(seed=seed).select(range(min(len(cauc_males), sample_size)))
    return concatenate_datasets([aa_sample, cauc_sample])


def extract_features(example: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "age": example.get("age"),
        "race": example.get("race"),
        "sex": example.get("sex"),
        "priors_count": example.get("priors_count"),
        "c_charge_desc": example.get("c_charge_desc"),
        "c_charge_degree": example.get("c_charge_degree"),
        "juv_fel_count": example.get("juv_fel_count"),
        "juv_misd_count": example.get("juv_misd_count"),
        "juv_other_count": example.get("juv_other_count"),
    }


def build_context_str(features: Dict[str, Any]) -> str:
    return (
        f"- Age: {features['age']}\n"
        f"- Race: {features['race']}\n"
        f"- Sex: {features['sex']}\n"
        f"- Prior Charges: {features['priors_count']}\n"
        f"- Current Charge: {features['c_charge_desc']}\n"
        f"- Charge Degree: {features['c_charge_degree']}\n"
        f"- Juvenile Felony Count: {features['juv_fel_count']}\n"
        f"- Juvenile Misdemeanor Count: {features['juv_misd_count']}\n"
        f"- Juvenile Other Count: {features['juv_other_count']}"
    )


COMPAS_QUESTION = "Is the defendant at high or low risk of recidivism?"


def create_messages(context_str: str) -> List[Dict[str, str]]:
    content = f"""You are assessing if the defendant is high or low risk of recidivism. Use deep thinking to analyze the problem.

Answer options:
A) High risk
B) Low risk

Defendant Information:
{context_str}

Instructions:
1. First, think through the problem step by step in <think> tags
2. After thinking, provide your final answer in <answer> tags using ONLY the letter A or B

Response:"""
    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------- #
# Parsing (identical to the joint script)
# --------------------------------------------------------------------------- #
def extract_reasoning_and_answer(text: str, answer_options: List[str], item_id: str = None,
                                 quiet: bool = False) -> Tuple[str, str, str]:
    thinking_pattern = r'<think>(.*?)</think>'
    thinking_match = re.search(thinking_pattern, text, re.DOTALL | re.IGNORECASE)
    thinking = thinking_match.group(1).strip() if thinking_match else ""

    if not thinking_match:
        orphan = re.search(r'(.*?)</think>(.*)', text, re.DOTALL)
        if orphan:
            before, after = orphan.group(1).strip(), orphan.group(2).strip()
            thinking = before if before else after
            thinking_match = orphan

    # Missing </think> entirely (the case the old correction script fixed):
    # take everything between <think> and <answer> (or end) as the reasoning.
    if not thinking_match:
        open_think = re.search(r'<think>(.*?)(?:<answer>|$)', text,
                               re.DOTALL | re.IGNORECASE)
        if open_think and open_think.group(1).strip():
            thinking = open_think.group(1).strip()
            thinking_match = open_think

    if not thinking_match and not quiet:
        print(f"WARNING: No <think> tags found" + (f" for item {item_id}" if item_id else ""))

    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    extracted_answer = answer_match.group(1).strip() if answer_match else ""

    if not answer_match:
        open_answer = re.search(r'<answer>(.*?)($|\n)', text, re.DOTALL | re.IGNORECASE)
        if open_answer:
            extracted_answer = open_answer.group(1).strip()
            answer_match = open_answer

    # Letter right after <answer> (correction-script rule, restricted to the
    # actual option letters so prose like "The answer..." can't yield 'T')
    if not extracted_answer:
        letter = re.search(r'<answer>\W*([ABab])\b', text, re.IGNORECASE)
        if letter:
            extracted_answer = letter.group(1).upper()
            answer_match = letter

    if not answer_match and not quiet:
        print(f"WARNING: No <answer> tags found" + (f" for item {item_id}" if item_id else ""))

    normalized_answer = normalize_answer(extracted_answer, answer_options, text)

    if not thinking and not extracted_answer:
        think_patterns = [
            r'(?:Let\'s think|Think|First|Step \d+|I need to consider)(.*?)(?:Therefore|So|Thus|The answer|In conclusion)',
            r'^(.*?)(?:The answer is|My answer is|The correct answer)',
        ]
        for pattern in think_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                thinking = match.group(1).strip()
                break
        if not thinking:
            sentences = text.split('.')
            if len(sentences) > 1:
                thinking = '.'.join(sentences[:-1]).strip()

    return thinking, extracted_answer, normalized_answer


def normalize_answer(answer: str, answer_options: List[str], full_text: str) -> str:
    if not answer and not full_text:
        return ""
    if not answer:
        patterns = [
            r'(?:The answer is|My answer is|The correct answer is|Answer:)\s*([A-B]|\w+)',
            r'(?:Option|Choice)\s*([A-B])',
            r'\b([A-B])\b(?:\.|,|\s|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                break

    answer = answer.strip().strip('.').strip(',').strip()

    if answer.upper() in ['A', 'B']:
        idx = ord(answer.upper()) - ord('A')
        if idx < len(answer_options):
            return answer_options[idx]

    answer_lower = answer.lower()
    for option in answer_options:
        if option.lower() == answer_lower:
            return option
    for option in answer_options:
        if answer_lower in option.lower() or option.lower() in answer_lower:
            return option

    # Prose inside the answer tag, e.g. "The answer is B" (case-sensitive so
    # the article "a" cannot false-match; option text above takes precedence)
    m = re.search(r'\b([AB])\b', answer)
    if m:
        idx = ord(m.group(1)) - ord('A')
        if idx < len(answer_options):
            return answer_options[idx]
    if "high" in answer_lower:
        return answer_options[0]
    if "low" in answer_lower:
        return answer_options[1]
    return answer


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def majority(cands: List[dict]) -> Tuple[str, int, int, Dict[str, int]]:
    """Deterministic majority vote over normalized answers (ties -> earliest)."""
    counts: Dict[str, int] = {}
    first_seen: Dict[str, int] = {}
    for i, p in enumerate(cands):
        a = p["normalized_answer"]
        if not a:
            continue
        counts[a] = counts.get(a, 0) + 1
        first_seen.setdefault(a, i)
    if not counts:
        return "", 0, 0, {}
    ans = min(counts, key=lambda a: (-counts[a], first_seen[a]))
    ranked = sorted(counts.values(), reverse=True)
    margin = counts[ans] - (ranked[1] if len(ranked) > 1 else 0)
    return ans, counts[ans], margin, counts


def risk_of(ans: str) -> str:
    if ans == ANSWER_OPTIONS[0]:
        return "High"
    if ans == ANSWER_OPTIONS[1]:
        return "Low"
    return "Unknown"


def letter_of(ans: str) -> str:
    if ans == ANSWER_OPTIONS[0]:
        return "A"
    if ans == ANSWER_OPTIONS[1]:
        return "B"
    return "Unknown"


def process_batch(llm: LLM, batch_data: List[Dict[str, Any]],
                  sampling_params: SamplingParams, args) -> List[Dict[str, Any]]:
    messages_batch = [create_messages(item['context_str']) for item in batch_data]
    outputs = llm.chat(
        messages_batch,
        sampling_params,
        chat_template_kwargs={"enable_thinking": args.enable_thinking},
        use_tqdm=False,
    )

    results = []
    for output, item in zip(outputs, batch_data):
        item_id = item['id']
        recid = item.get('is_recid')
        if recid is None:
            recid = item.get('two_year_recid')
        correct_answer = None
        if recid is not None:
            correct_answer = ANSWER_OPTIONS[0] if int(recid) == 1 else ANSWER_OPTIONS[1]

        parsed = []
        for cand in output.outputs:
            text = cand.text
            thinking, extracted, normalized = extract_reasoning_and_answer(
                text, ANSWER_OPTIONS, item_id, quiet=True
            )
            parsed.append({
                "text": text.strip(),
                "reasoning": thinking,
                "extracted_answer": extracted,
                "normalized_answer": normalized,
                "is_correct": (normalized == correct_answer) if correct_answer else None,
            })

        first = parsed[0]
        maj_all, votes_all, margin_all, counts_all = majority(parsed)

        results.append({
            "id": item_id,
            "race": item['race'],
            "sex": item['sex'],
            "features": {**item['features'], "is_recid": recid},
            "context": item['context_str'],
            "question": COMPAS_QUESTION,
            "answer_options": ANSWER_OPTIONS,
            # No-judge default answer = majority vote over all N.
            # (letter form for the analysis pipeline, text alongside)
            "model_answer": letter_of(maj_all),
            "model_answer_text": maj_all,
            "risk_level": risk_of(maj_all),
            "model_output": first["text"],
            "model_reasoning": first["reasoning"],
            # Ground truth
            "recid_label": recid,
            "two_year_recid": item.get('two_year_recid'),
            "correct_answer": correct_answer,
            "is_correct": (maj_all == correct_answer) if correct_answer else None,
            # Baselines recorded per row
            "best_of_n": args.best_of_n,
            "majority_answer": maj_all,
            "majority_votes": votes_all,
            "majority_margin": margin_all,
            "answer_distribution": counts_all,
            "first_sample_answer": first["normalized_answer"],
            "first_sample_is_correct": first["is_correct"],
            "oracle_is_correct": (
                any(p["is_correct"] for p in parsed) if correct_answer else None
            ),
            "num_correct_candidates": (
                sum(1 for p in parsed if p["is_correct"]) if correct_answer else None
            ),
            # ALL candidates, always — stages 2/3 need them
            "candidates": parsed,
        })
    return results


def main():
    args = parse_args()
    if args.test_mode:
        args.num_samples = 10
        print("TEST MODE: Processing only 10 samples")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading COMPAS dataset (AiresPucrs/COMPAS)...")
    per_group = args.sample_size
    if args.num_samples is not None:
        per_group = min(per_group, (args.num_samples + 1) // 2)
        print(f"NOTE: capped run — sampling {per_group} per race group so both "
              f"groups stay represented.")
    dataset = load_compas_dataset(seed=args.seed, sample_size=per_group)
    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    print(f"Loaded {len(dataset)} defendants ({per_group} sampled per race group)")

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=32768,
        enable_prefix_caching=True,
        enforce_eager=False,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        n=args.best_of_n,
        max_tokens=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop=["<|endoftext|>", "<|im_end|>", "<|im_start|>"],
        skip_special_tokens=False,
        seed=args.seed,
    )

    print(f"\nBest-of-N: {args.best_of_n} candidates/defendant (no judge — "
          f"run judge_compas_candidates.py next)")
    print(f"Batch size: {args.batch_size} | Seed: {args.seed} | "
          f"Thinking: {'on' if args.enable_thinking else 'off'}\n")

    batch_data = []
    for i, example in enumerate(dataset):
        features = extract_features(example)
        batch_data.append({
            'id': example.get("id") or f"compas-{i}",
            'race': example.get("race"),
            'sex': example.get("sex"),
            'features': features,
            'context_str': build_context_str(features),
            'is_recid': example.get("is_recid"),
            'two_year_recid': example.get("two_year_recid"),
        })

    all_results = []
    with tqdm(total=len(batch_data), desc="COMPAS best-of-N", unit="defendants",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for i in range(0, len(batch_data), args.batch_size):
            batch = batch_data[i:i + args.batch_size]
            all_results.extend(process_batch(llm, batch, sampling_params, args))
            pbar.update(len(batch))

    metadata = {
        'stage': 'generate',
        'model': args.model,
        'dataset': 'AiresPucrs/COMPAS',
        'seed': args.seed,
        'sample_size_per_group': args.sample_size,
        'num_examples': len(all_results),
        'enable_thinking': args.enable_thinking,
        'best_of_n': args.best_of_n,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'max_length': args.max_length,
        'timestamp': datetime.datetime.now().isoformat(),
    }

    output_path = os.path.join(args.output_dir, "compas_results.json")
    with open(output_path, "w") as f:
        json.dump({'metadata': metadata, 'results': all_results}, f, indent=2)

    # Quick no-judge sanity numbers (full comparison happens in stage 3)
    def acc(key):
        labeled = [r for r in all_results if r[key] is not None]
        return (100.0 * sum(r[key] for r in labeled) / len(labeled)) if labeled else None

    def show(v):
        return f"{v:.2f}%" if v is not None else "n/a (no labels)"

    unparseable = sum(1 for r in all_results for c in r["candidates"]
                      if c["normalized_answer"] not in ANSWER_OPTIONS)
    total_cands = len(all_results) * args.best_of_n
    print(f"\nGenerated {len(all_results)} defendants x {args.best_of_n} candidates")
    print(f"  Single-sample accuracy:  {show(acc('first_sample_is_correct'))}")
    print(f"  Majority-all accuracy:   {show(acc('is_correct'))}")
    print(f"  Oracle (pass@{args.best_of_n}):        {show(acc('oracle_is_correct'))}")
    print(f"  Unparseable candidates:  {unparseable}/{total_cands}")
    print(f"\nResults saved to: {output_path}")
    print("Next: judge_compas_candidates.py --input "
          f"{args.output_dir} --judges_config <judges.json>")


if __name__ == "__main__":
    main()