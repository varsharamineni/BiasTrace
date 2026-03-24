#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict


# ---------------------------
# 1. Load metric JSON
# ---------------------------
def load_metric_json(file_path, col_name):
    """Load metric JSON and return DataFrame with required columns."""
    with open(file_path, "r") as f:
        data = json.load(f)

    rows = []
    for item in data.get("results", []):
        jo = item.get("judge_output", {}) or {}
        if col_name in ["baseline05", "baseline-frm"]:
            val = jo.get("score")
        else:
            val = jo.get("bias_label")
        
        rows.append({
            "sample_id": item.get("example_id"),
            "category": item.get("category"),
            "is_correct": item.get("is_correct"),
            "stereotype_alignment": item.get("stereotype_alignment"),
            "correct_answer": item.get("correct_answer"),
            "model_answer": item.get("model_answer"),
            col_name: val
        })

    return pd.DataFrame(rows)


# ---------------------------
# 2. Merge multiple metrics across categories
# ---------------------------
def merge_metrics(metric_files):
    metric_tables = defaultdict(list)

    for metric in metric_files:
        df_metric = load_metric_json(metric["file"], metric["col"])
        metric_tables[metric["col"]].append(df_metric)

    merged_metrics = []

    for i, (col, dfs) in enumerate(metric_tables.items()):
        df_all = pd.concat(dfs, ignore_index=True)

        if i == 0:
            # Keep metadata from first metric
            keep = ["sample_id", "category", "is_correct", "stereotype_alignment", col]
        else:
            keep = ["sample_id", "category", col]

        merged_metrics.append(df_all[keep])

    df = merged_metrics[0]
    for other in merged_metrics[1:]:
        df = df.merge(other, on=["sample_id", "category"], how="outer")

    return df


# ---------------------------
# 3. Derived columns
# ---------------------------
def add_derived_columns(df):
    df["incorrect"] = (~df["is_correct"].astype(bool)).astype(int)
    df["incorrect_and_stereotype"] = (
        (~df["is_correct"].astype(bool) & df["stereotype_alignment"].astype(bool))
    ).astype(int)
    return df


# ---------------------------
# 4. Plot barplots for metrics
# ---------------------------
def plot_metrics(df, metric_cols, output_dir=None):
    sns.set(style="whitegrid")
    for col in metric_cols:
        plt.figure(figsize=(6, 4))
        
        # Bin continuous metrics
        if df[col].dtype in [float, int] and df[col].nunique() > 10:
            binned_col = col + "_bin"
            df[binned_col] = pd.cut(df[col], bins=5)
            plot_col = binned_col
        else:
            plot_col = col
        
        sns.barplot(x=plot_col, y="is_correct", data=df)
        plt.title(f"Answer correctness vs {col}")
        plt.ylabel("Fraction Correct")
        plt.xlabel(col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"barplot_{col}.pdf"), dpi=300)
        plt.close()  # Close figure to prevent memory issues


# ---------------------------
# 5. Main
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
        #"bias01_pathways": "new_metric_pathways_annotation",
        "baseline-frm": "fairness-prm_0-5_annotation"
    }

    # 1. Discover all files
    category_files = defaultdict(lambda: defaultdict(list))
    for base_dir in base_dirs:
        for col, subfolder in metric_subfolders.items():
            folder_path = os.path.join(base_dir, subfolder)
            for json_file in glob.glob(os.path.join(folder_path, "**", "llm_eval_*.json"), recursive=True):
                category = os.path.basename(os.path.dirname(json_file))
                category_files[category][col].append(json_file)

    # 2. Keep categories with all metrics
    valid_categories = {
        cat: files
        for cat, files in category_files.items()
        if all(k in files and len(files[k]) > 0 for k in metric_subfolders.keys())
    }

    # 3. Flatten into list for loading
    metrics_to_load = []
    for cat, files in valid_categories.items():
        for col in metric_subfolders.keys():
            for f in files[col]:
                metrics_to_load.append({"file": f, "col": col, "category": cat})

    # 4. Merge all metrics
    df = merge_metrics(metrics_to_load)
    df["baseline05_bin"] = (df["baseline05"] > 0).astype(int)
    df = add_derived_columns(df)

    # 5. Unique metric names for correlation and summaries
    unique_metrics = list({m["col"] for m in metrics_to_load})
    bin_cols = [col+"_bin" for col in unique_metrics if col+"_bin" in df.columns]
    metric_cols = unique_metrics + bin_cols

    # 6. Correlation matrix
    corr_cols = metric_cols + ["is_correct", "incorrect", "incorrect_and_stereotype"]
    corr_matrix = df[corr_cols].corr(method="pearson")
    corr_matrix.to_csv("correlation_matrix.csv")
    print("\nPearson correlation matrix:")
    print(corr_matrix)

    # 7. Mean correctness per metric
    print("\nMean correctness by metric:")
    for col in unique_metrics:
        print(f"\n{col}:")
        print(df.groupby(col)["is_correct"].mean())

    # 8. Mean incorrect and incorrect+stereotype per metric
    print("\nMean incorrect and incorrect_and_stereotype by metric:")
    for col in unique_metrics:
        print(f"\n{col}:")
        print(df.groupby(col)[["incorrect","incorrect_and_stereotype"]].mean())

    # 9. Plot barplots
    plot_metrics(df, unique_metrics, output_dir="plots")