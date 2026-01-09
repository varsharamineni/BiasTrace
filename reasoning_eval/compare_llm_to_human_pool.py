#!/usr/bin/env python3
"""
Compute metrics for two human-labeled datasets and two corresponding LLM output folders,
then combine metrics for matching model/prompt/parameters across datasets.
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
import numpy as np
from tqdm import tqdm

# ----------------------------
# Parse LLM filename for metadata
# ----------------------------
def parse_llm_filename(fname: str):
    name = Path(fname).stem
    if name.startswith("llm_eval_"):
        name = name[len("llm_eval_"):]
    parts = name.split("_")
    temp_idx = next((i for i, p in enumerate(parts) if p.startswith("temp")), None)

    if temp_idx is not None:
        model_prompt_parts = parts[:temp_idx]
        sampling_parts = parts[temp_idx:]
        reasoning = None
        if not sampling_parts[-1].startswith(("temp", "top", "seed", "max")):
            reasoning = sampling_parts[-1]
            sampling_parts = sampling_parts[:-1]

        model = model_prompt_parts[0]
        prompt = "_".join(model_prompt_parts[1:]) if len(model_prompt_parts) > 1 else ""
        sampling_str = "_".join(sampling_parts)

        def extract(pattern, text, cast=float):
            import re
            m = re.search(pattern, text)
            return cast(m.group(1)) if m else None

        temperature = extract(r"temp([0-9.]+)", sampling_str)
        top_p = extract(r"top_p([0-9.]+)", sampling_str)
        seed = extract(r"seed([0-9]+)", sampling_str, int)
        max_tokens = extract(r"max_tokens([0-9]+)", sampling_str, int)

        return {
            "model": model,
            "prompt": prompt,
            "reasoning_style": reasoning,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "max_tokens": max_tokens,
        }

    # fallback
    model = parts[0]
    prompt = "_".join(parts[1:]) if len(parts) > 1 else ""
    return {
        "model": model,
        "prompt": prompt,
        "reasoning_style": None,
        "temperature": None,
        "top_p": None,
        "seed": None,
        "max_tokens": None,
    }

# ----------------------------
# Compute metrics for one dataset
# ----------------------------
def compute_metrics(human_file, llm_folder, dataset_id):
    with open(human_file, "r") as f:
        human_data = json.load(f)
    human_df = pd.DataFrame(human_data)

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

    all_metrics = []

    for llm_file in tqdm(list(Path(llm_folder).glob("*.json")), desc=f"Processing {dataset_id}"):
        with open(llm_file, "r") as f:
            llm_data_raw = json.load(f)

        if isinstance(llm_data_raw, dict):
            llm_results = llm_data_raw.get("results", [])
            metadata = llm_data_raw.get("metadata", {})
        else:
            llm_results = llm_data_raw
            metadata = {}

        file_meta = parse_llm_filename(llm_file.name)

        judge_model = metadata.get("judge_model", file_meta["model"])
        judge_prompt = metadata.get("judge_prompt", file_meta["prompt"])
        enable_thinking = metadata.get("enable_thinking", False)

        reasoning_style = metadata.get("reasoning_style", file_meta["reasoning_style"])
        temperature = metadata.get("temperature", file_meta["temperature"])
        top_p = metadata.get("top_p", file_meta["top_p"])
        seed = metadata.get("seed", file_meta["seed"])
        max_tokens = metadata.get("max_tokens", file_meta["max_tokens"])

        llm_df = pd.DataFrame(llm_results)
        if "judge_output" in llm_df.columns:
            judge_expanded = pd.json_normalize(llm_df["judge_output"])
            llm_df = pd.concat([llm_df.drop(columns=["judge_output"]), judge_expanded], axis=1)

        llm_df["sample_id"] = human_df["sample_id"].values[:len(llm_df)]
        merged_df = human_df.merge(llm_df, on="sample_id", suffixes=("_human", "_llm"))

        # Compute metrics per label
        for col in label_cols:
            human_col = col + "_human"
            llm_col = col + "_llm"

            if human_col not in merged_df or llm_col not in merged_df:
                continue

            merged_df[human_col] = pd.to_numeric(merged_df[human_col], errors="coerce")
            merged_df[llm_col] = pd.to_numeric(merged_df[llm_col], errors="coerce")

            df_valid = merged_df.dropna(subset=[human_col, llm_col])
            if df_valid.empty:
                continue

            y_true = df_valid[human_col]
            y_pred = df_valid[llm_col]

            all_metrics.append({
                "dataset_id": dataset_id,
                "judge_model": judge_model,
                "judge_prompt": judge_prompt,
                "reasoning_style": reasoning_style,
                "thinking_mode": enable_thinking,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "max_tokens": max_tokens,
                "error_label": col,
                "n_samples": len(df_valid),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=np.nan),
                "recall": recall_score(y_true, y_pred, zero_division=np.nan),
                "f1_score": f1_score(y_true, y_pred, zero_division=np.nan),
                "cohens_kappa": cohen_kappa_score(y_true, y_pred),
            })
    return pd.DataFrame(all_metrics)

# ----------------------------
# Combine metrics across datasets
# ----------------------------
def combine_metrics(df_list):
    combined_df = pd.concat(df_list, ignore_index=True)

    # group by model/prompt/params/error_label
    group_cols = [
        "judge_model", "judge_prompt", "reasoning_style", "thinking_mode",
        "temperature", "top_p", "seed", "max_tokens", "error_label"
    ]

    def weighted_mean(group, col):
        # weight by number of samples
        return np.average(group[col], weights=group["n_samples"])

    combined_metrics = combined_df.groupby(group_cols).apply(
        lambda g: pd.Series({
            "n_samples_total": g["n_samples"].sum(),
            "accuracy": weighted_mean(g, "accuracy"),
            "precision": weighted_mean(g, "precision"),
            "recall": weighted_mean(g, "recall"),
            "f1_score": weighted_mean(g, "f1_score"),
            "cohens_kappa": weighted_mean(g, "cohens_kappa"),
        })
    ).reset_index()

    return combined_metrics

# ----------------------------
# CLI
# ----------------------------
parser = argparse.ArgumentParser(description="Compute combined metrics across datasets")
parser.add_argument("--human_val", type=str, required=True, help="Validation human labels JSON")
parser.add_argument("--llm_val_folder", type=str, required=True, help="Folder with LLM results for val set")
parser.add_argument("--human_test", type=str, required=True, help="Test human labels JSON")
parser.add_argument("--llm_test_folder", type=str, required=True, help="Folder with LLM results for test set")
parser.add_argument("--output_prefix", type=str, default="results/combined", help="Output CSV prefix")
args = parser.parse_args()

# ----------------------------
# Compute per-dataset metrics
# ----------------------------
val_metrics = compute_metrics(args.human_val, args.llm_val_folder, "val")
test_metrics = compute_metrics(args.human_test, args.llm_test_folder, "test")

# ----------------------------
# Save per-dataset metrics
# ----------------------------
val_metrics.to_csv(f"{args.output_prefix}_val_metrics.csv", index=False)
test_metrics.to_csv(f"{args.output_prefix}_test_metrics.csv", index=False)
print(f"✅ Saved per-dataset metrics CSVs")

# ----------------------------
# Combine metrics across datasets
# ----------------------------
combined_metrics = combine_metrics([val_metrics, test_metrics])
combined_metrics.to_csv(f"{args.output_prefix}_combined_metrics.csv", index=False)
print(f"✅ Saved combined metrics CSV")

# ----------------------------
# Rank model/prompt combos by Cohen's kappa per error label
# ----------------------------
ranked_combos = []

# Loop over each error label
for label in combined_metrics['error_label'].unique():
    subset = combined_metrics[combined_metrics['error_label'] == label].copy()
    
    # Sort descending by Cohen's kappa
    subset.sort_values("cohens_kappa", ascending=False, inplace=True)
    
    # Add rank column
    subset['rank'] = range(1, len(subset) + 1)
    
    # Keep only relevant columns
    subset_display = subset[[
        'rank', 'judge_model', 'judge_prompt', 'cohens_kappa',
        'accuracy', 'precision', 'recall', 'f1_score', 'n_samples_total'
    ]]
    
    ranked_combos.append(subset_display)

# Combine all error labels into one CSV
ranked_df = pd.concat(ranked_combos, keys=combined_metrics['error_label'].unique(), names=['error_label', 'row']).reset_index(level=1, drop=True).reset_index()
ranked_df.to_csv(f"{args.output_prefix}_ranked_by_kappa_per_label.csv", index=False)
print("✅ Saved ranked model/prompt combos per error label (by Cohen's kappa)")