#!/usr/bin/env python3
"""
Compare all LLM judge outputs in a folder to human-labeled ground truth.

Outputs:
- Merged CSV with all samples + metadata extracted from filename
- Metrics CSV (long format) per model × prompt × error_label
- Leaderboard CSV (mean accuracy per model × prompt)
"""

import json
import re
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score
)
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np


# -------------------------------------------------------------
# Parse metadata embedded in filenames
# -------------------------------------------------------------
def parse_llm_filename(fname: str):
    """
    Robust filename parser.
    Supports:
    - llm_eval_model_prompt_temp1.0_top_p1.0_seed42_max_tokens2048_reasoning.json
    - llm_eval_model_prompt.json
    - llm_eval_model_prompt_something_else.json   (no sampling)
    """

    name = Path(fname).stem  # strip .json

    # Remove prefix
    if name.startswith("llm_eval_"):
        name = name[len("llm_eval_"):]

    parts = name.split("_")

    # Try to find sampling parameters (temp…)
    temp_idx = next((i for i, p in enumerate(parts) if p.startswith("temp")), None)

    # ------------------------------------------------------
    # CASE 1 — full structured filename with sampling params
    # ------------------------------------------------------
    if temp_idx is not None:
        model_prompt_parts = parts[:temp_idx]
        sampling_parts = parts[temp_idx:]

        # Optional reasoning at the end
        reasoning = None
        if not sampling_parts[-1].startswith(("temp", "top", "seed", "max")):
            reasoning = sampling_parts[-1]
            sampling_parts = sampling_parts[:-1]

        model = model_prompt_parts[0]
        prompt = "_".join(model_prompt_parts[1:]) if len(model_prompt_parts) > 1 else ""
        sampling_str = "_".join(sampling_parts)

        # Extract numeric params
        def extract(pattern, text, cast=float):
            m = re.search(pattern, text)
            return cast(m.group(1)) if m else None

        temperature = extract(r"temp([0-9.]+)", sampling_str)
        top_p = extract(r"top_p([0-9.]+)", sampling_str)
        seed = extract(r"seed([0-9]+)", sampling_str, int)
        max_tokens = extract(r"max_tokens([0-9]+)", sampling_str, int)

        return {
            "model": model,
            "prompt": prompt,
            "sampling_string": sampling_str,
            "reasoning_style": reasoning,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "max_tokens": max_tokens,
        }

    # ------------------------------------------------------
    # CASE 2 — no sampling params → fallback mode
    # ------------------------------------------------------
    model = parts[0]
    prompt = "_".join(parts[1:]) if len(parts) > 1 else ""

    return {
        "model": model,
        "prompt": prompt,
        "sampling_string": None,
        "reasoning_style": None,
        "temperature": None,
        "top_p": None,
        "seed": None,
        "max_tokens": None,
    }

# ----------------------------
# CLI arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Compare all LLM outputs to human labels")
parser.add_argument("--human_file", type=str, required=True, help="JSON file with human labels")
parser.add_argument("--llm_folder", type=str, required=True, help="Folder with LLM output JSONs")
parser.add_argument("--output_prefix", type=str, default="results/comparison", help="Prefix for output CSVs")
args = parser.parse_args()

# ----------------------------
# Load human labels
# ----------------------------
with open(args.human_file, "r") as f:
    human_data = json.load(f)
human_df = pd.DataFrame(human_data)

# ----------------------------
# Define the error/label columns explicitly
# ----------------------------
label_cols = [
    "group_assumption",
    "bias_acknowledgement",
    "meta_reflection",
    "outside_demo_knowledge",
    "outside_topical_knowledge",
    "unresolved",
    "overthinking",
    "missing_logic"
]

# ----------------------------
# Collect metrics + merged rows
# ----------------------------
all_metrics = []
all_merged_rows = []

for llm_file in tqdm(list(Path(args.llm_folder).glob("*.json")), desc="Processing LLM outputs"):

    # -------------------------------------------------------------
    # Load JSON + detect format
    # -------------------------------------------------------------
    with open(llm_file, "r") as f:
        llm_data_raw = json.load(f)

    if isinstance(llm_data_raw, dict):
        llm_results = llm_data_raw.get("results", [])
        metadata = llm_data_raw.get("metadata", {})
    else:
        llm_results = llm_data_raw
        metadata = {}

    # -------------------------------------------------------------
    # Parse metadata from filename (used if JSON metadata missing)
    # -------------------------------------------------------------
    file_meta = parse_llm_filename(llm_file.name)

    judge_model = metadata.get("judge_model", file_meta["model"])
    judge_prompt = metadata.get("judge_prompt", file_meta["prompt"])
    enable_thinking = metadata.get("enable_thinking", False)

    reasoning_style = metadata.get("reasoning_style", file_meta["reasoning_style"])
    temperature = metadata.get("temperature", file_meta["temperature"])
    top_p = metadata.get("top_p", file_meta["top_p"])
    seed = metadata.get("seed", file_meta["seed"])
    max_tokens = metadata.get("max_tokens", file_meta["max_tokens"])

    # -------------------------------------------------------------
    # Build DataFrame
    # -------------------------------------------------------------
    llm_df = pd.DataFrame(llm_results)

    # Flatten nested judge_output dict
    if "judge_output" in llm_df.columns:
        judge_expanded = pd.json_normalize(llm_df["judge_output"])
        llm_df = pd.concat([llm_df.drop(columns=["judge_output"]), judge_expanded], axis=1)

    # Align sample IDs
    llm_df["sample_id"] = human_df["sample_id"].values[:len(llm_df)]

    # Inject metadata
    llm_df["judge_model"] = judge_model
    llm_df["judge_prompt"] = judge_prompt
    llm_df["thinking_mode"] = enable_thinking

    llm_df["reasoning_style"] = reasoning_style
    llm_df["temperature"] = temperature
    llm_df["top_p"] = top_p
    llm_df["seed"] = seed
    llm_df["max_tokens"] = max_tokens

    # -------------------------------------------------------------
    # Merge with human labels
    # -------------------------------------------------------------
    merged_df = human_df.merge(llm_df, on="sample_id", suffixes=("_human", "_llm"))
    merged_df["source_file"] = llm_file.name
    all_merged_rows.append(merged_df)

    # -------------------------------------------------------------
    # Compute metrics per label
    # -------------------------------------------------------------
    for col in label_cols:
        human_col = col + "_human"
        llm_col = col + "_llm"

        if human_col not in merged_df or llm_col not in merged_df:
            print(f"⚠️ Missing column: {col}")
            continue

        merged_df[human_col] = pd.to_numeric(merged_df[human_col], errors="coerce")
        merged_df[llm_col] = pd.to_numeric(merged_df[llm_col], errors="coerce")

        df_valid = merged_df.dropna(subset=[human_col, llm_col])
        if df_valid.empty:
            continue

        y_true = df_valid[human_col]
        y_pred = df_valid[llm_col]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=np.nan)
        recall = recall_score(y_true, y_pred, zero_division=np.nan)
        f1 = f1_score(y_true, y_pred, zero_division=np.nan)
        kappa = cohen_kappa_score(y_true, y_pred)
        pearson = df_valid[[human_col, llm_col]].corr().iloc[0, 1]
        spearman = df_valid[[human_col, llm_col]].corr(method="spearman").iloc[0, 1]

        all_metrics.append({
            "judge_model": judge_model,
            "judge_prompt": judge_prompt,
            "thinking_mode": enable_thinking,
            "reasoning_style": reasoning_style,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "max_tokens": max_tokens,
            "error_label": col,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cohens_kappa": kappa,
            "pearson": pearson,
            "spearman": spearman,
            "human_positive_ratio": y_true.mean(),
        })

# ----------------------------
# Save merged CSV
# ----------------------------
final_merged = pd.concat(all_merged_rows, ignore_index=True)
merged_file = f"{args.output_prefix}_merged.csv"
final_merged.to_csv(merged_file, index=False)
print(f"✅ Merged CSV saved to {merged_file}")

# ----------------------------
# Save long-format metrics CSV
# ----------------------------
metrics_long_df = pd.DataFrame(all_metrics)
metrics_long_file = f"{args.output_prefix}_metrics_long.csv"
metrics_long_df.to_csv(metrics_long_file, index=False)
print(f"✅ Metrics CSV saved to {metrics_long_file}")

# ----------------------------
# Leaderboard
# ----------------------------
leaderboard = metrics_long_df.groupby(
    ["judge_model", "judge_prompt", "reasoning_style", "temperature", "thinking_mode"]
)[["cohens_kappa", "accuracy", "precision", "recall", "f1_score"]].mean().reset_index()

leaderboard_file = f"{args.output_prefix}_leaderboard.csv"
leaderboard.to_csv(leaderboard_file, index=False)
print(f"✅ Leaderboard CSV saved to {leaderboard_file}")

print("\n🏆 Leaderboard (sorted by Cohen's κ) 🏆")
print(leaderboard.sort_values("cohens_kappa", ascending=False).head(100))
