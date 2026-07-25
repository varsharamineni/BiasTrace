#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
import numpy as np

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



def is_unknown_option_selected(item):
    """
    Returns 1 if the model selected an 'unknown' type answer option.
    """
    answer = (item.get("model_answer") or "").strip().lower()
    options = item.get("answer_options") or []

    unknown_phrases = [
        "can't be determined",
        "cannot be determined",
        "not enough information",
        "unknown",
        "cannot answer"
    ]

    # Check if the selected answer itself is an unknown-type option
    return int(any(p in answer for p in unknown_phrases))

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
            metric_val = score 
        else:
            metric_val = jo.get("bias_label")

        rows.append({
            "sample_id": item.get("sample_id"),
            "model_type": model_type,
            "prompt_type": prompt_type,
            metric_name: metric_val,
            "is_correct": item.get("is_correct"),
            "stereotype_alignment": item.get("stereotype_alignment"),
            "unknown_selected": is_unknown_option_selected(item)
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
        index=["sample_id","model_type","prompt_type","is_correct","stereotype_alignment", "unknown_selected"],
        values=metric_cols,
        aggfunc='first'
    ).reset_index()

    return df_metrics


# ---------------------------
# 6. Cohen's Kappa matrix
# ---------------------------
def cohen_kappa_matrix(df, cols):
    """
    Compute a Cohen's Kappa matrix for a list of columns.
    Returns a DataFrame where (i,j) is Cohen's kappa between cols[i] and cols[j].
    """
    n = len(cols)
    kappa_mat = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                kappa_mat[i, j] = 1.0
            else:
                # Drop NaNs for pairwise comparison
                valid_idx = df[[cols[i], cols[j]]].dropna().index
                if len(valid_idx) == 0:
                    kappa = np.nan
                else:
                    kappa = cohen_kappa_score(df.loc[valid_idx, cols[i]], df.loc[valid_idx, cols[j]])
                kappa_mat[i, j] = kappa
    
    return pd.DataFrame(kappa_mat, index=cols, columns=cols)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":

    # Ensure plots folder exists
    os.makedirs("plots", exist_ok=True)

    # --- Full table display settings ---
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")


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
        "bias01_pathways_diff": "new_metric_pathways_annotation",
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
    metric_cols = ["baseline05", "baseline01", "bias01_pathways", "bias01_pathways_diff", "baseline-frm"]

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
    df["incorrect_because_unknown"] = (
    (~df["is_correct"].astype(bool)) & (df["unknown_selected"] == 1)
    ).astype(int)
    df["baseline05_bin"] = df["baseline05"].apply(lambda x: 1 if x > 0 else 0)

    # ---------------------------
    # 2. List of metric columns
    # ---------------------------
    metric_cols = ["baseline05", "baseline05_bin", "baseline01", "bias01_pathways", "bias01_pathways_diff", "baseline-frm"]
    derived_cols = ["incorrect", "incorrect_and_stereotype"]
    corr_cols = metric_cols + derived_cols

    # ---------------------------
    # 3. Overall Pearson correlations
    # ---------------------------
    print("\n=== Overall Pearson Correlation ===")
    overall_corr = df[corr_cols].corr(method="pearson")
    overall_corr.to_csv("plots/overall_pearson_corr.csv")
    print(overall_corr)

    # ---------------------------
    # 4. Grouped correlations by model_type x prompt_type
    # ---------------------------
    print("\n=== Grouped Pearson Correlations ===")
    group_cols = ["model_type", "prompt_type"]
    for (model, prompt), df_group in df.groupby(group_cols):
        print(f"\n--- {model} | {prompt} (n={len(df_group)}) ---")
        group_corr = df_group[corr_cols].corr(method="pearson")
        group_corr.to_csv(f"plots/pearson_corr_{model}_{prompt}.csv")
        print(group_corr)

    


    # ---------------------------
    # Cohens kappa
    # ---------------------------

    kappa_cols = ["bias01_pathways", "bias01_pathways_diff", "baseline01", "baseline05_bin", "incorrect", "incorrect_and_stereotype"]

    print("\n=== Overall Cohen's Kappa Matrix ===")
    kappa_overall = cohen_kappa_matrix(df, kappa_cols)
    print(kappa_overall)

    print("\n=== Grouped Cohen's Kappa Matrices ===")
    for (model, prompt), df_group in df.groupby(group_cols):
        print(f"\n--- {model} | {prompt} (n={len(df_group)}) ---")
        kappa_group = cohen_kappa_matrix(df_group, kappa_cols)
        kappa_group.to_csv(f"plots/cohen_kappa_{model}_{prompt}.csv")
        print(kappa_group)


    # Create plots folder if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # ---------------------------
    # 5. Bar plots of incorrect / incorrect_and_stereotype per metric
    # ---------------------------

    plot_cols = ["incorrect"]
    metric_cols_new = ["baseline05", "baseline05_bin", "baseline01", "bias01_pathways", "bias01_pathways_diff"]

    for derived_col in plot_cols:
        for metric in metric_cols_new:
            # Aggregate mean of derived_col per metric value
            agg_df = df.groupby(metric)[derived_col].mean().reset_index()

            plt.figure(figsize=(6,4))
            sns.barplot(
                data=agg_df,
                x=metric,
                y=derived_col,
                color="skyblue"
            )
            plt.title(f"Average {derived_col} per {metric}")
            plt.ylabel(f"Average {derived_col}")
            plt.xlabel(metric)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save figure
            save_path = f"plots/{derived_col}_bar_vs_{metric}.png"
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Saved bar plot: {save_path}")

    # ---------------------------
    # 0. Manual mapping
    # ---------------------------
    group_mapping = {
        "gpt-oss-120b | simple_prompt_low_reasoning": "GPT-OSS-120B | Simple | Low",
        "gpt-oss-120b | simple_prompt_medium_reasoning": "GPT-OSS-120B | Simple | Medium",
        "gpt-oss-120b | full_prompt_low_reasoning": "GPT-OSS-120B | Guided | Low",
        "gpt-oss-120b | full_prompt_medium_reasoning": "GPT-OSS-120B | Guided | Medium",
        "qwen_full_14B | simple_prompt": "Qwen3-14B | Simple",
        "qwen_full_14B | full_prompt": "Qwen3-14B | Guided",
        "qwen_full_8B | simple_prompt": "Qwen3-8B | Simple",
        "qwen_full_8B | full_prompt": "Qwen3-8B | Guided",
    }

    # Create a temporary combined key
    df["group_key"] = df["model_type"] + " | " + df["prompt_type"]

    # Map to publication-ready labels
    df["group_label"] = df["group_key"].map(group_mapping)

    # Drop rows that are not in the mapping
    df = df.dropna(subset=["group_label"])

    desired_order = list(group_mapping.values())  # preserves dict order
    df["group_label"] = pd.Categorical(df["group_label"], categories=desired_order, ordered=True)
    
    # ---------------------------
    # 1. Metrics and target
    # ---------------------------
    metric_cols = ["bias01_pathways", "baseline01", "baseline05", "baseline05_bin", "baseline-frm"]
    target_col = "incorrect_and_stereotype"

    rename_dict = {
        "baseline01": "Baseline 0/1",
        "baseline05": "Baseline 0-5",
        "baseline05_bin": "Baseline 0-5 bin",
        "baseline-frm": "Baseline FRM",
        "bias01_pathways": "BiasTrace Prompt"
    }

    # ---------------------------
    # 2. Compute correlations per group using mapped labels
    # ---------------------------
    corr_records = []

    for label, df_group in df.groupby("group_label"):
        corr_series = df_group[metric_cols + [target_col]].corr(method="pearson")[target_col].loc[metric_cols]
        
        for metric, corr_val in corr_series.items():
            corr_records.append({
                "group_label": label,
                "metric": rename_dict[metric],
                "correlation": corr_val
            })

    corr_df = pd.DataFrame(corr_records)

    # ---------------------------
    # 3. Pivot and reorder metrics
    # ---------------------------
    corr_matrix = corr_df.pivot(index="group_label", columns="metric", values="correlation")
    ordered_cols = ["BiasTrace Prompt", "Baseline 0/1", "Baseline 0-5", "Baseline 0-5 bin", "Baseline FRM"]
    corr_matrix = corr_matrix[ordered_cols]
    corr_matrix["Baseline FRM"] = corr_matrix["Baseline FRM"].abs()

    # ---------------------------
    # 4. Plot heatmap
    # ---------------------------
    import matplotlib as mpl
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    sns.reset_defaults()
    plt.rcParams.update(plt.rcParamsDefault)

    plt.figure(figsize=(10, 6))
    #sns.set_style("whitegrid", font_scale=0.9)
    sns.set()
    sns.set_theme(style="white")

    ax = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        cbar=False,
        linewidths=0.5,
        #linecolor="white",
        annot_kws={"size": 16}
    )

    #plt.title("Correlation of Reasoning Evaluation with Biased Outcomes on BBQ Dataset", fontsize=14, color="black")
    plt.xlabel("Reasoning Bias Score", fontsize=16, color="black", fontweight="medium")
    plt.ylabel("Model | Prompt | Reasoning Level", fontsize=16, color="black", fontweight="medium")
    plt.xticks(rotation=20, ha="right", fontsize=16, fontweight="medium")
    plt.yticks(rotation=0, fontsize=16, fontweight="medium")
    plt.tight_layout()

    # ---------------------------
    # 5. Save as PDF
    # ---------------------------
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/corr_incorrect_stereotype_vertical.pdf", format="pdf", bbox_inches="tight")
    plt.show()


 