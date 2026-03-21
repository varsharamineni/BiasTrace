#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os
import glob
from collections import defaultdict


# ---------------------------
# 1. Load metric JSON
# ---------------------------
def load_metric_json(file_path, col_name):
    """
    Load metric JSON and return DataFrame with required columns.
    Handles 'score' for 0-0.5 metrics and 'bias_label' for binary metrics.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    rows = []
    for item in data.get("results", []):
        jo = item.get("judge_output", {}) or {}
        val = jo.get("score") if "05" in col_name else jo.get("bias_label")
        rows.append({
            "sample_id": item.get("example_id"),
            "category": item.get("category"),
            "is_correct": item.get("is_correct"),
            "stereotype_alignment": item.get("stereotype_alignment"),
            "correct_answer": item.get("correct_answer"),
            "model_answer": item.get("model_answer"),
            col_name: val
        })

    df = pd.DataFrame(rows)

    return df

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
            keep = [
                "sample_id",
                "category",
                "is_correct",
                "stereotype_alignment",
                col
            ]
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
        # Safe version
    df["incorrect"] = (~df["is_correct"].astype(bool)).astype(int)

    # For incorrect_and_stereotype
    df["incorrect_and_stereotype"] = (
        (~df["is_correct"].astype(bool) & df["stereotype_alignment"].astype(bool))
    ).astype(int)
    
    return df

# ---------------------------
# 4. Cramér's V
# ---------------------------
def cramers_v(x, y):
    """
    Compute Cramér's V between two categorical columns
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r-1, k-1))))

def compute_cramers_matrix(df, cols):
    """
    Compute Cramér's V matrix for a list of columns
    """
    cramers_matrix = pd.DataFrame(index=cols, columns=cols)
    for col1 in cols:
        for col2 in cols:
            cramers_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
    return cramers_matrix.astype(float)

# ---------------------------
# 5. Plot barplots for metrics
# ---------------------------
def plot_metrics(df, metrics, output_dir=None):
    sns.set(style="whitegrid")
    for m in metrics:
        col = m["col"]
        plt.figure(figsize=(6,4))
        sns.barplot(x=col, y="is_correct", data=df)
        plt.title(f"Answer correctness vs {col}")
        plt.ylabel("Fraction Correct")
        plt.xlabel(col)
        plt.tight_layout()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"barplot_{col}.pdf"), dpi=300)
        plt.show()

# ---------------------------
# 6. Main
# ---------------------------
if __name__ == "__main__":

    base_dir = "outputs/qwen_full_8B_simple_prompt/20250827_163953"

    metric_subfolders = {
        "baseline05": "baseline_0-5_annotation",
        "baseline01": "baseline_annotation",
        "bias01": "new_metric_annotation",
        "bias01_pathways": "new_metric_pathways_annotation"
    }

    # 1. Discover all files and group by category
    category_files = defaultdict(lambda: defaultdict(list))

    for col, subfolder in metric_subfolders.items():
        folder_path = os.path.join(base_dir, subfolder)
        for json_file in glob.glob(os.path.join(folder_path, "**", "llm_eval_*.json"), recursive=True):
            # Extract category from path (assumes subfolder/category/llm_eval_*.json)
            parts = json_file.split(os.sep)
            category = parts[-2]  # second-to-last folder is the category
            category_files[category][col].append(json_file)

    # 2. Keep only categories with all three metrics
    valid_categories = {
        cat: files
        for cat, files in category_files.items()
        if all(k in files for k in ["baseline05","baseline01","bias01", "bias01_pathways"]) or \
        all(len(files[k]) > 0 for k in ["baseline05","baseline01","bias01", "bias01_pathways"])
    }

    # 3. Flatten into a list for your loading loop
    metrics_to_load = []
    for cat, files in valid_categories.items():
        for col in ["baseline05", "baseline01", "bias01", "bias01_pathways"]:
            metrics_to_load.append({
                "file": files[col][0],  # pick first JSON file
                "col": col,
                "category": cat
            })

    # Print paths
    for m in metrics_to_load:
        print(m)

    # 1. Merge all metrics
    df = merge_metrics(metrics_to_load)
    df["baseline05_bin"] = (df["baseline05"] > 0).astype(int)
    print(df.columns.tolist())
    
    # 2. Derived columns
    df = add_derived_columns(df)

    # 3. Correlation matrix
    metric_cols = []
    for m in metrics_to_load:
        metric_cols.append(m["col"])
        if m["col"] + "_bin" in df.columns:
            metric_cols.append(m["col"] + "_bin")
    
    corr_cols = metric_cols + [
        "is_correct",
        "incorrect",
        "incorrect_and_stereotype"
    ]

    corr_matrix = df[corr_cols].corr(method="pearson")

    print("\nPearson correlation matrix:")
    print(corr_matrix)

    corr_matrix.to_csv("correlation_matrix.csv")

    # 4. Mean correctness by metric
    print("\nMean correctness by label:")
    for m in metrics_to_load:
        col = m["col"]
        print(f"\n{col}:")
        print(df.groupby(col)["is_correct"].mean())

    # 5. Mean incorrect and incorrect+stereotype by metric
    print("\nMean incorrect and incorrect_and_stereotype by label:")
    for m in metrics_to_load:
        col = m["col"]
        print(f"\n{col}:")
        print(df.groupby(col)[["incorrect","incorrect_and_stereotype"]].mean())

    # 6. Plot barplots
    plot_metrics(df, metrics_to_load, output_dir="plots")

    # 7. Cramér's V
    categorical_cols = metric_cols + ["incorrect", "incorrect_and_stereotype"]
    cramers_matrix = compute_cramers_matrix(df, categorical_cols)
    print("\nCramér's V matrix:")
    print(cramers_matrix)
    cramers_matrix.to_csv("cramers_v_matrix.csv", index=True)