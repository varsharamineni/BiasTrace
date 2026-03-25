#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
import glob
import os

# ---------------------------
# Helper: extract model_type and prompt_type from folder
# ---------------------------
def parse_model_prompt(base_dir):
    """
    Extract model_type and prompt_type from folder path.
    Assumes: outputs/<model_prompt_folder>/<timestamp>/
    """
    parts = base_dir.rstrip("/").split(os.sep)
    try:
        outputs_index = parts.index("outputs")
        model_prompt_folder = parts[outputs_index + 1]
    except ValueError:
        model_prompt_folder = parts[-2]  # fallback
    
    if model_prompt_folder.startswith("qwen_full_"):
        p = model_prompt_folder.split("_")
        model_type = "_".join(p[:3])
        prompt_type = "_".join(p[3:])
    elif model_prompt_folder.startswith("gpt-oss-120b"):
        p = model_prompt_folder.split("_")
        model_type = "gpt-oss-120b"
        prompt_type = "_".join(p[1:])
    else:
        p = model_prompt_folder.split("_")
        model_type = p[0]
        prompt_type = "_".join(p[1:])
    return model_type, prompt_type

# ---------------------------
# Load a single JSON metric file
# ---------------------------
def load_json_file(file_path, metric_name, model_type, prompt_type):
    rows = []
    with open(file_path, "r") as f:
        data = json.load(f)
    for item in data.get("results", []):
        jo = item.get("judge_output") or {}  # handle None

        if metric_name == "baseline05":
            metric_val = jo.get("score")
        elif metric_name == "baseline-frm":
            score = jo.get("score")
            metric_val = 1.0 - score if score is not None else None
        else:
            metric_val = jo.get("bias_label")


        rows.append({
            "sample_id": item.get("sample_id"),
            "model_type": model_type,
            "prompt_type": prompt_type,
            metric_name: metric_val,
            "is_correct": item.get("is_correct"),
            "stereotype_alignment": item.get("stereotype_alignment")
        })
    return pd.DataFrame(rows)

# ---------------------------
# Merge all metrics
# ---------------------------
def merge_all_metrics(base_dirs, metric_subfolders):
    all_data = []

    for base_dir in base_dirs:
        model_type, prompt_type = parse_model_prompt(base_dir)
        for metric_name, subfolder in metric_subfolders.items():
            folder_path = os.path.join(base_dir, subfolder)
            for file_path in glob.glob(os.path.join(folder_path, "**", "llm_eval_*.json"), recursive=True):
                df = load_json_file(file_path, metric_name, model_type, prompt_type)
                all_data.append(df)

    # Concatenate everything
    df_all = pd.concat(all_data, ignore_index=True)

    # Pivot to have one row per sample_id x model_type x prompt_type
    metric_cols = list(metric_subfolders.keys())
    df_metrics = df_all.pivot_table(
        index=["sample_id","model_type","prompt_type","is_correct","stereotype_alignment"],
        values=metric_cols,
        aggfunc='first'
    ).reset_index()

    return df_metrics

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    base_dirs = [
        "outputs/qwen_full_8B_simple_prompt/20250827_163953",
        "outputs/qwen_full_8B_full_prompt",
        "outputs/qwen_full_14B_simple_prompt/20250828_215719",
        "outputs/qwen_full_14B_full_prompt",
        "outputs/gpt-oss-120b_simple_prompt_low_reasoning/20251216_114545",
        "outputs/gpt-oss-120b_simple_prompt_medium_reasoning/20251217_110543",
        "outputs/gpt-oss-120b_full_prompt_low_reasoning/20251218_140849",
        "outputs/gpt-oss-120b_full_prompt_low_reasoning/20251225_204037",
        "outputs/gpt-oss-120b_full_prompt_medium_reasoning/20251218_113157",
        "outputs/gpt-oss-120b_full_prompt_medium_reasoning/20251225_224835",
        "outputs/gpt-oss-120b_full_prompt_medium_reasoning/20251226_123752"
    ]

    metric_subfolders = {
        "baseline05": "baseline_0-5_annotation",
        "baseline01": "baseline_annotation",
        "bias01_pathways": "new_metric_pathways_annotation",
        "baseline-frm": "fairness-prm_0-5_annotation"
    }

    # Merge everything
    df = merge_all_metrics(base_dirs, metric_subfolders)

    # Ensure sample_id is string
    df["sample_id"] = df["sample_id"].astype(str)

    print("Merged DataFrame shape:", df.shape)
    print(df.head())

    # Rows per model/prompt
    print("\nRows per model_type x prompt_type:")
    print(df.groupby(["model_type","prompt_type"]).size())

    # List of metric columns
    metric_cols = ["baseline05", "baseline01", "bias01_pathways", "baseline-frm"]

    # Keep only rows where all metrics exist (non-NaN)
    df = df.dropna(subset=metric_cols).copy()

       # Rows per model/prompt
    print("\nRows per model_type x prompt_type:")
    print(df.groupby(["model_type","prompt_type"]).size())

    # ---------------------------
    # 1. Derived columns
    # ---------------------------
    df["incorrect"] = (~df["is_correct"].astype(bool)).astype(int)
    df["incorrect_and_stereotype"] = (
        (~df["is_correct"].astype(bool) & df["stereotype_alignment"].astype(bool))
    ).astype(int)

    # ---------------------------
    # 2. List of metric columns
    # ---------------------------
    metric_cols = ["baseline05", "baseline01", "bias01_pathways", "baseline-frm"]
    derived_cols = ["incorrect", "incorrect_and_stereotype"]
    corr_cols = metric_cols + derived_cols

    # ---------------------------
    # 3. Overall Pearson correlations
    # ---------------------------
    print("\n=== Overall Pearson Correlation ===")
    overall_corr = df[corr_cols].corr(method="pearson")
    print(overall_corr)

    # ---------------------------
    # 4. Grouped correlations by model_type x prompt_type
    # ---------------------------
    print("\n=== Grouped Pearson Correlations ===")
    group_cols = ["model_type", "prompt_type"]
    for (model, prompt), df_group in df.groupby(group_cols):
        print(f"\n--- {model} | {prompt} (n={len(df_group)}) ---")
        group_corr = df_group[corr_cols].corr(method="pearson")
        print(group_corr)