#!/usr/bin/env python
"""
gpt_oss_reasoning_length_analysis.py

Focus:
- GPT-OSS models only
- Extract reasoning tokens directly from bbq_*_results_merged.json
- Test whether reasoning length predicts:
    1. incorrect_and_stereotype
    2. correlation diagnostics

Key idea:
Separates reasoning effort vs reasoning verbosity.
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from scipy import stats

# ======================
# CONFIG
# ======================

GPT_OSS_BASE_DIRS = [
    "outputs/gpt-oss-120b_simple_prompt_low_reasoning/20251216_114545",
    "outputs/gpt-oss-120b_simple_prompt_medium_reasoning/20251217_110543",
    "outputs/gpt-oss-120b_full_prompt_low_reasoning/20251218_140849",
    "outputs/gpt-oss-120b_full_prompt_low_reasoning/20251225_204037",
    "outputs/gpt-oss-120b_full_prompt_medium_reasoning/20251218_113157",
    "outputs/gpt-oss-120b_full_prompt_medium_reasoning/20251225_224835",
    "outputs/gpt-oss-120b_full_prompt_medium_reasoning/20251226_123752",
]

OUTCOME = "incorrect_and_stereotype"

tokenizer = AutoTokenizer.from_pretrained(
    "openai/gpt-oss-120b",
    use_fast=True
)

# ======================
# TOKEN COUNTER
# ======================

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


# ======================
# PARSING HELPERS (FIXED)
# ======================

def parse_model_dir(base_dir):
    """
    Extract:
    - prompt_type: simple_prompt / full_prompt
    - reasoning_level: low / medium / high
    """
    folder = base_dir.rstrip("/").split("/")[-2]

    prompt_type = "unknown"
    if "simple_prompt" in folder:
        prompt_type = "simple_prompt"
    elif "full_prompt" in folder:
        prompt_type = "full_prompt"

    reasoning_level = "unknown"
    for lvl in ["low", "medium", "high"]:
        if f"{lvl}_reasoning" in folder:
            reasoning_level = lvl
            break

    return prompt_type, reasoning_level


# ======================
# LOAD DATA (FIXED)
# ======================

def load_dataset(base_dirs):
    rows = []

    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Skipping missing: {base_dir}")
            continue

        prompt_type, reasoning_level = parse_model_dir(base_dir)

        merged_files = glob.glob(
            os.path.join(base_dir, "**", "bbq_*_results_merged.json"),
            recursive=True
        )

        if not merged_files:
            print(f"No merged files found in {base_dir}")
            continue

        for f in merged_files:
            with open(f) as fp:
                data = json.load(fp)

            for r in data.get("results", []):

                reasoning_text = (
                    r.get("model_reasoning")
                    or r.get("reasoning")
                    or ""
                )

                token_count = count_tokens(reasoning_text)

                rows.append({
                    "sample_id": f"{r.get('category','unk')}_{r.get('example_id','unk')}",

                    "prompt_type": prompt_type,
                    "bbq_category": r.get("category", "unknown"),
                    "reasoning_level": reasoning_level,

                    "reasoning_text": reasoning_text,
                    "reasoning_tokens": token_count,
                    "log_reasoning_tokens": np.log1p(token_count),

                    "is_correct": int(r.get("is_correct", False)),
                    "incorrect_and_stereotype": int(r.get("incorrect_and_stereotype", False)),
                })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No data loaded. Check merged files exist.")

    return df


# ======================
# CORRELATIONS
# ======================

def correlation_checks(df):
    print("\n--- Correlations ---")

    pb = stats.pointbiserialr(
        df["incorrect_and_stereotype"],
        df["reasoning_tokens"]
    )

    sp = stats.spearmanr(
        df["incorrect_and_stereotype"],
        df["reasoning_tokens"]
    )

    print(f"Point-biserial r: {pb.statistic:.3f}, p={pb.pvalue:.3e}")
    print(f"Spearman r:       {sp.statistic:.3f}, p={sp.pvalue:.3e}")


# ======================
# LENGTH BINNING
# ======================

def bin_analysis(df):
    df = df.copy()

    df["length_bin"] = pd.qcut(
        df["reasoning_tokens"],
        10,
        duplicates="drop"
    )

    summary = df.groupby("length_bin").agg(
        stereo_rate=("incorrect_and_stereotype", "mean"),
        mean_tokens=("reasoning_tokens", "mean"),
        n=("reasoning_tokens", "count")
    )

    print("\n--- Length bin analysis ---")
    print(summary)

    return summary


# ======================
# MAIN
# ======================

def main(output_dir="./reasoning_length_analysis"):

    os.makedirs(output_dir, exist_ok=True)

    df = load_dataset(GPT_OSS_BASE_DIRS)

    print(f"\nLoaded rows: {len(df)}")
    print(f"Mean tokens: {df['reasoning_tokens'].mean():.2f}")

    correlation_checks(df)
    bin_analysis(df)

    # save dataset
    df.to_csv(os.path.join(output_dir, "gpt_oss_reasoning_length.csv"), index=False)

    print(f"\nSaved dataset to {output_dir}")

    # ======================
    # MEAN TOKENS BY REASONING LEVEL
    # ======================

    print("\n--- Mean reasoning tokens by reasoning level ---")

    level_summary = df.groupby("reasoning_level").agg(
        n=("reasoning_tokens", "count"),
        mean_tokens=("reasoning_tokens", "mean"),
        std_tokens=("reasoning_tokens", "std"),
        median_tokens=("reasoning_tokens", "median"),
        mean_log_tokens=("log_reasoning_tokens", "mean"),
        stereo_rate=("incorrect_and_stereotype", "mean"),
    ).reset_index()

    level_summary["stereo_rate_pct"] = level_summary["stereo_rate"] * 100

    print(level_summary.sort_values("mean_tokens", ascending=False))

    level_summary.to_csv(
        os.path.join(output_dir, "reasoning_level_summary.csv"),
        index=False
    )

    # ======================
    # MEAN TOKENS BY PROMPT TYPE
    # ======================

    print("\n--- Mean reasoning tokens by prompt type ---")

    prompt_summary = df.groupby("prompt_type").agg(
        n=("reasoning_tokens", "count"),
        mean_tokens=("reasoning_tokens", "mean"),
        std_tokens=("reasoning_tokens", "std"),
        median_tokens=("reasoning_tokens", "median"),
        stereo_rate=("incorrect_and_stereotype", "mean"),
    ).reset_index()

    prompt_summary["stereo_rate_pct"] = prompt_summary["stereo_rate"] * 100

    print(prompt_summary)

    prompt_summary.to_csv(
        os.path.join(output_dir, "prompt_type_summary.csv"),
        index=False
    )


# ======================
# CLI
# ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./reasoning_length_analysis")
    args = parser.parse_args()

    main(args.output_dir)