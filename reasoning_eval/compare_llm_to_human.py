#!/usr/bin/env python3
"""
Compare all LLM judge outputs in a folder to human-labeled ground truth.

Outputs:
- Merged CSV with all samples + judge_model + judge_prompt
- Metrics CSV (long format) per model × prompt × error_label
- Leaderboard CSV (mean accuracy per model × prompt)
"""

import json
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

# ----------------------------
# CLI arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Compare all LLM outputs to human labels")
parser.add_argument(
    "--human_file", type=str, required=True, help="JSON file with human labels"
)
parser.add_argument(
    "--llm_folder", type=str, required=True, help="Folder with LLM output JSONs"
)
parser.add_argument(
    "--output_prefix", type=str, default="results/comparison", help="Prefix for output CSVs"
)
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
# Collect all metrics and merged rows
# ----------------------------
all_metrics = []
all_merged_rows = []

for llm_file in tqdm(list(Path(args.llm_folder).glob("*.json")), desc="Processing LLM outputs"):
    with open(llm_file, "r") as f:
        llm_data_raw = json.load(f)

    # Determine format: list vs dict
    if isinstance(llm_data_raw, dict):
        llm_results = llm_data_raw.get("results", [])
        metadata = llm_data_raw.get("metadata", {})
    elif isinstance(llm_data_raw, list):
        llm_results = llm_data_raw
        metadata = {}
    else:
        raise ValueError(f"Unexpected LLM JSON format: {llm_file}")

    judge_model = metadata.get("judge_model", "")
    judge_prompt = metadata.get("judge_prompt", "")
    enable_thinking = metadata.get("enable_thinking", False)  


    llm_df = pd.DataFrame(llm_results)

    # Flatten nested judge_output dict if present
    if "judge_output" in llm_df.columns:
        judge_expanded = pd.json_normalize(llm_df["judge_output"])
        llm_df = pd.concat([llm_df.drop(columns=["judge_output"]), judge_expanded], axis=1)

    # Align sample_id with human labels
    llm_df["sample_id"] = human_df["sample_id"].values[:len(llm_df)]

    # Add judge metadata
    llm_df["judge_model"] = judge_model
    llm_df["judge_prompt"] = judge_prompt
    llm_df["thinking_mode"] = enable_thinking

    # Merge with human labels
    merged_df = human_df.merge(llm_df, on="sample_id", suffixes=("_human", "_llm"))
    merged_df["source_file"] = llm_file.name
    all_merged_rows.append(merged_df)

    # ----------------------------
    # Compute metrics per error/label column (long format)
    # ----------------------------
    for col in label_cols:
        human_col = col + "_human"
        llm_col = col + "_llm"
        if human_col not in merged_df.columns or llm_col not in merged_df.columns:
            print(f"⚠️ Skipping metrics for missing column: {col}")
            continue

        # Ensure numeric
        merged_df[human_col] = pd.to_numeric(merged_df[human_col], errors="coerce")
        merged_df[llm_col] = pd.to_numeric(merged_df[llm_col], errors="coerce")

        # Drop rows with NaN values for this label
        df_valid = merged_df.dropna(subset=[human_col, llm_col])
        if df_valid.empty:
            print(f"⚠️ No valid data for {col}")
            continue

        y_true = df_valid[human_col]
        y_pred = df_valid[llm_col]

        # Compute metrics safely
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=np.nan)
            recall = recall_score(y_true, y_pred, zero_division=np.nan)
            kappa = cohen_kappa_score(y_true, y_pred, labels=[0,1])
            f1 = f1_score(y_true, y_pred, zero_division=np.nan)
            pearson = df_valid[[human_col, llm_col]].corr(method="pearson").iloc[0, 1]
            spearman = df_valid[[human_col, llm_col]].corr(method="spearman").iloc[0, 1]
        except Exception:
            precision = recall = kappa = pearson = spearman = None

        # **Compute class prevalence**
        class_prevalence = y_true.mean()  # fraction of positives in human labels

        all_metrics.append({
            "judge_model": judge_model,
            "judge_prompt": judge_prompt,
            "thinking_mode": enable_thinking,
            "error_label": col,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cohens_kappa": kappa,
            "pearson": pearson,
            "spearman": spearman,
            "human_positive_ratio": class_prevalence  # new column
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

# Ensure 'thinking_mode' column exists for all rows
if "thinking_mode" not in metrics_long_df.columns:
    metrics_long_df["thinking_mode"] = False

metrics_long_file = f"{args.output_prefix}_metrics_long.csv"
metrics_long_df.to_csv(metrics_long_file, index=False)
print(f"✅ Metrics CSV (long format) saved to {metrics_long_file}")

# ----------------------------
# Leaderboard (mean metrics per model × prompt)
# ----------------------------
leaderboard = metrics_long_df.groupby(["judge_model", "judge_prompt", "thinking_mode"])[
    ["accuracy", "precision", "recall", "f1_score", "cohens_kappa"]
].mean().reset_index()

leaderboard_file = f"{args.output_prefix}_leaderboard.csv"
leaderboard.to_csv(leaderboard_file, index=False)
print(f"✅ Leaderboard CSV saved to {leaderboard_file}")

# Display top models by mean accuracy
print("\n🏆 Leaderboard (top by mean accuracy) 🏆")
print(leaderboard.sort_values("cohens_kappa", ascending=False).head(500))
