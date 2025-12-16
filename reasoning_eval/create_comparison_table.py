import json
import os
import pandas as pd
from itertools import product
from sklearn.metrics import cohen_kappa_score

# --------------------------------------------------
# 1️⃣ Loader for multi-label or single-label files
# --------------------------------------------------
def load_judge_file_multi(path, model_name=None, params=None, score_style=None):
    """
    Normalize files with either single or multi-label judge outputs.
    Returns long-format DataFrame:
        sample_id | model_name | params | score_style | metric | value
    """
    with open(path, "r") as f:
        data = json.load(f)
    
    records = []
    
    if "results" in data:
        for r in data["results"]:
            sample_id = r.get("sample_id")
            judge_output = r.get("judge_output")
            
            if isinstance(judge_output, dict):
                for metric, val in judge_output.items():
                    if isinstance(val, (int, float)):
                        records.append({
                            "sample_id": sample_id,
                            "model_name": model_name,
                            "params": params or "",
                            "score_style": metric,
                            "metric": metric,
                            "value": val
                        })
            elif isinstance(judge_output, (int, float)):
                records.append({
                    "sample_id": sample_id,
                    "model_name": model_name,
                    "params": params or "",
                    "score_style": score_style,
                    "metric": score_style,
                    "value": judge_output
                })
            else:
                continue
    else:
        # Human GT: explode all available 0/1 labels
        for r in data:
            sample_id = r.get("sample_id")
            for metric in r:
                if metric == "sample_id":
                    continue
                val = r[metric]
                if isinstance(val, (int, float)):
                    records.append({
                        "sample_id": sample_id,
                        "model_name": "Human",
                        "params": "",
                        "score_style": metric,
                        "metric": metric,
                        "value": val
                    })
    
    return pd.DataFrame(records)


# --------------------------------------------------
# 2️⃣ Build pairwise comparison matrix
# --------------------------------------------------
def build_matrix(df_long, metric="accuracy"):
    """
    Build a square comparison matrix for all combinations of:
    model_name + params + score_style.
    """
    df_long["combination"] = df_long.apply(
        lambda r: f"{r['model_name']} | {r['params']} | {r['score_style']}", axis=1
    )
    
    combinations = df_long["combination"].unique()
    matrix = pd.DataFrame(index=combinations, columns=combinations, dtype=float)
    
    for a, b in product(combinations, repeat=2):
        if a == b:
            matrix.loc[a, b] = 1.0
            continue

        df_a = df_long[df_long["combination"] == a][["sample_id", "value"]].copy()
        df_b = df_long[df_long["combination"] == b][["sample_id", "value"]].copy()

        # Drop missing values
        merged = pd.merge(df_a, df_b, on="sample_id", suffixes=("_a", "_b")).dropna(subset=["value_a", "value_b"])
        if merged.empty:
            matrix.loc[a, b] = None
            continue

        # Make sure values are numeric or string
        merged["value_a"] = merged["value_a"].astype(int)
        merged["value_b"] = merged["value_b"].astype(int)

        if metric == "accuracy":
            val = (merged["value_a"] == merged["value_b"]).mean()
        elif metric == "cohens_kappa":
            val = cohen_kappa_score(merged["value_a"], merged["value_b"])
        else:
            raise ValueError("metric must be 'accuracy' or 'cohens_kappa'")
        
        matrix.loc[a, b] = val

    return matrix


# --------------------------------------------------
# 3️⃣ Define all files with labels
# --------------------------------------------------
files = [
    # Meta Llama - our labels
    ("reasoning_eval/llm_judge_samples/val_set/our_labels/llm_eval_meta-llama_Meta-Llama-3-70B-Instruct_detailed_example_clarification_opt_temp0.6_top_p0.9_seed42_max_tokens2048.json",
     "Llama 70B", "temp0.6, top_p0.9", "detailed_example_clarification_opt"),
    ("reasoning_eval/llm_judge_samples/val_set/our_labels/llm_eval_meta-llama_Meta-Llama-3-70B-Instruct_detailed_example_clarification_opt_temp0.6_top_p0.9_seed42_max_tokens2048_reasoning.json",
     "Llama 70B", "temp0.6, top_p0.9", "detailed_example_clarification_opt"),

    # Meta Llama - baseline
    ("reasoning_eval/llm_judge_samples/val_set/baseline/llm_eval_meta-llama_Meta-Llama-3-70B-Instruct_llama70B_gt_temp0.01_top_p0.95_seed42_max_tokens2048.json",
     "Llama 70B", "temp0.01, top_p0.95", "0-5 bin"),
    ("reasoning_eval/llm_judge_samples/val_set/baseline/llm_eval_meta-llama_Meta-Llama-3-70B-Instruct_llama70B_gt_temp0.6_top_p0.9_seed42_max_tokens2048.json",
     "Llama 70B", "temp0.6, top_p0.9", "0/1 bias"),
    ("reasoning_eval/llm_judge_samples/val_set/baseline/llm_eval_meta-llama_Meta-Llama-3-70B-Instruct_baseline_temp0.6_top_p0.9_seed42_max_tokens2048.json",
     "Llama 70B", "temp0.6, top_p0.9", "0/1 bias"),

    # Claude - baseline
    ("reasoning_eval/llm_judge_samples/val_set/baseline/llm_eval_claude-opus-4-1-20250805_llama70B_gt_temp0.6_seed42_max_tokens2048.json",
     "Claude", "temp0.6, top_p0.9", "0-5 bin"),
    ("reasoning_eval/llm_judge_samples/val_set/baseline/llm_eval_claude-opus-4-1-20250805_baseline_temp0.6_seed42_max_tokens2048.json",
     "Claude", "temp0.6, top_p0.9", "0/1 bias"),

    # Claude - our labels
    ("reasoning_eval/llm_judge_samples/val_set/our_labels/llm_eval_claude-opus-4-1-20250805_detailed_example_clarification_opt_temp0.6_seed42_max_tokens2048.json",
     "Claude", "temp0.6, top_p0.9", "detailed_example_clarification_opt"),
    ("reasoning_eval/llm_judge_samples/val_set/our_labels/llm_eval_claude-opus-4-1-20250805_detailed_example_clarification_opt_temp0.6_seed42_max_tokens2048_reasoning.json",
     "Claude", "temp0.6, top_p0.9", "detailed_example_clarification_opt"),


    # GPT 120B - our labels
    ("reasoning_eval/llm_judge_samples/val_set/our_labels/llm_eval_openai_gpt-oss-120b_detailed_example_clarification_opt_temp1.0_top_p1.0_seed42_max_tokens2048_medium.json",
     "Claude", "temp0.6, top_p0.9", "detailed_example_clarification_opt")


    # Human
    ("reasoning_eval/ground_truth_samples/val_set.json",
     "Human", "", None)
]

# --------------------------------------------------
# 4️⃣ Load all files
# --------------------------------------------------
dfs = []
for path, model, params, score_style in files:
    if os.path.exists(path):
        dfs.append(load_judge_file_multi(path, model, params, score_style))
all_df = pd.concat(dfs, ignore_index=True)

print(f"✅ Loaded {len(all_df)} rows from all files.")

print(all_df.head())

metric_map = {
    "group_assumption": "0/1 GA",
    "bias_acknowledgement": "0/1 BA",
    "meta_reflection": "0/1 MR",
    "outside_demo_knowledge": "0/1 ODK",
    "outside_topical_knowledge": "0/1 OTK",
    "missing_logic": "0/1 ML",
    "overthinking": "0/1 OT",
    "unresolved": "0/1 UR",
    "bias_label": "0/1 bias",
    "score": "0-5 bin"

}

params_map = {
    "temp0.6, top_p0.9": "t0.6",
    "temp0.01, top_p0.95": "t0.01",
    "": ""
}


all_df["metric"] = all_df["metric"].map(metric_map).fillna(all_df["metric"])
all_df["score_style"] = all_df["score_style"].map(metric_map).fillna(all_df["score_style"])
all_df["params"] = all_df["params"].map(params_map).fillna(all_df["params"])


all_df["value"] = (all_df["value"] > 0).astype(int)

# List of metrics to remove
drop_metrics = ["example_id", "ambiguous"]

# Keep only rows whose metric is not in drop_metrics
all_df = all_df[~all_df["metric"].isin(drop_metrics)]

print(all_df.head())



# --------------------------------------------------
# 5️⃣ Build accuracy and kappa matrices
# --------------------------------------------------
accuracy_matrix = build_matrix(all_df, metric="accuracy")
kappa_matrix = build_matrix(all_df, metric="cohens_kappa")

# --------------------------------------------------
# 6️⃣ Save matrices
# --------------------------------------------------
os.makedirs("comparison_matrices", exist_ok=True)
accuracy_matrix.to_csv("comparison_matrices/all_combinations_accuracy_matrix.csv")
kappa_matrix.to_csv("comparison_matrices/all_combinations_kappa_matrix.csv")

print("✅ Saved all pairwise comparison matrices.")


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap



def plot_matrix(matrix, title, filename, figsize=(20, 20)):
    # Custom colormap: light for low values → dark for high values
    # Reverse of typical viridis
    colors = ["#f2f2f2", "#a6bddb", "#3690c0", "#034e7b"]  # light → dark
    custom_cmap = LinearSegmentedColormap.from_list("light_to_dark", colors)

    plt.figure(figsize=figsize)

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=custom_cmap,
        cbar_kws={'label': title},
        square=True,
        annot_kws={"size": 12},   # BIGGER TEXT
        linewidths=1.5,           # THICKER GRID LINES
        linecolor="black"
    )

    plt.title(title, fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"✅ Saved heatmap: {filename}")

# Create output directory
os.makedirs("comparison_matrices/figures", exist_ok=True)

# Plot accuracy matrix
plot_matrix(
    accuracy_matrix,
    title="Pairwise Accuracy Comparison",
    filename="comparison_matrices/figures/accuracy_matrix_heatmap.png"
)

# Plot Cohen's kappa matrix
plot_matrix(
    kappa_matrix,
    title="Pairwise Cohen's Kappa Comparison",
    filename="comparison_matrices/figures/kappa_matrix_heatmap.png"
)





combos = [
    {"model_name": "Human", "score_style": "0/1 GA", "params": ""},
    {"model_name": "Llama 70B", "score_style": "0/1 bias", "params": "t0.6"},
    {"model_name": "Llama 70B", "score_style": "0/1 bias", "params": "t0.01"},
    {"model_name": "Llama 70B", "score_style": "0-5 bin", "params": "t0.6"},
    {"model_name": "Claude", "score_style": "0-5 bin", "params": "t0.6"},
    {"model_name": "Claude", "score_style": "0/1 bias", "params": "t0.6"},
    {"model_name": "Human", "score_style": "0/1 GA", "params": ""},
    {"model_name": "Claude", "score_style": "0/1 GA", "params": "t0.6"},
    {"model_name": "Llama70B", "score_style": "0/1 GA", "params": "t0.6"},


]

# 1️⃣ Filter the dataframe for the selected combos
subset_rows = []
for combo in combos:
    mask = (
        (all_df["model_name"] == combo["model_name"]) &
        (all_df["score_style"] == combo["score_style"]) &
        (all_df["params"] == combo["params"])
    )
    subset_rows.append(all_df[mask])

subset_df = pd.concat(subset_rows, ignore_index=True)
print(f"✅ Subset df has {len(subset_df)} rows")

# 2️⃣ Build the pairwise matrices (reuse your build_matrix function)
subset_accuracy_matrix = build_matrix(subset_df, metric="accuracy")
subset_kappa_matrix = build_matrix(subset_df, metric="cohens_kappa")

# 3️⃣ Save the subset matrices
os.makedirs("comparison_matrices/subset", exist_ok=True)
subset_accuracy_matrix.to_csv("comparison_matrices/subset/subset_accuracy_matrix.csv")
subset_kappa_matrix.to_csv("comparison_matrices/subset/subset_kappa_matrix.csv")
print("✅ Saved subset pairwise matrices")

# 4️⃣ Plot the heatmaps (reuse your plot_matrix function)
os.makedirs("comparison_matrices/subset/figures", exist_ok=True)

plot_matrix(
    subset_accuracy_matrix,
    title="Subset Pairwise Accuracy Comparison",
    filename="comparison_matrices/subset/figures/subset_accuracy_matrix_heatmap.png"
)

plot_matrix(
    subset_kappa_matrix,
    title="Subset Pairwise Cohen's Kappa Comparison",
    filename="comparison_matrices/subset/figures/subset_kappa_matrix_heatmap.png"
)









##### TEST SET


def replace_val_with_test(files):
    """Given a list of (path, model, params, score_style), return same list but with val_set -> test_set."""
    test_files = []
    for path, model, params, score_style in files:
        test_path = path.replace("/val_set/", "/test_set/")
        test_files.append((test_path, model, params, score_style))
    return test_files

# 1️⃣ Load and process val set (already done as `all_df` and `subset_df`)
# 2️⃣ Repeat for test set
test_files = replace_val_with_test(files)

dfs_test = []
for path, model, params, score_style in test_files:
    if os.path.exists(path):
        dfs_test.append(load_judge_file_multi(path, model, params, score_style))
test_df = pd.concat(dfs_test, ignore_index=True)

# Apply same mappings and filters
test_df["metric"] = test_df["metric"].map(metric_map).fillna(test_df["metric"])
test_df["score_style"] = test_df["score_style"].map(metric_map).fillna(test_df["score_style"])
test_df["params"] = test_df["params"].map(params_map).fillna(test_df["params"])
test_df["value"] = (test_df["value"] > 0).astype(int)
test_df = test_df[~test_df["metric"].isin(drop_metrics)]

# 3️⃣ Full matrices for test set
test_accuracy_matrix = build_matrix(test_df, metric="accuracy")
test_kappa_matrix = build_matrix(test_df, metric="cohens_kappa")

os.makedirs("comparison_matrices/test", exist_ok=True)
test_accuracy_matrix.to_csv("comparison_matrices/test/test_set_accuracy_matrix.csv")
test_kappa_matrix.to_csv("comparison_matrices/test/test_set_kappa_matrix.csv")

# 4️⃣ Subset matrices for test set
subset_rows_test = []
for combo in combos:
    mask = (
        (test_df["model_name"] == combo["model_name"]) &
        (test_df["score_style"] == combo["score_style"]) &
        (test_df["params"] == combo["params"])
    )
    subset_rows_test.append(test_df[mask])

subset_test_df = pd.concat(subset_rows_test, ignore_index=True)
subset_test_accuracy_matrix = build_matrix(subset_test_df, metric="accuracy")
subset_test_kappa_matrix = build_matrix(subset_test_df, metric="cohens_kappa")

os.makedirs("comparison_matrices/test/subset", exist_ok=True)
subset_test_accuracy_matrix.to_csv("comparison_matrices/test/subset/subset_test_accuracy_matrix.csv")
subset_test_kappa_matrix.to_csv("comparison_matrices/test/subset/subset_test_kappa_matrix.csv")

# 5️⃣ Optional: plot heatmaps for test set subset
os.makedirs("comparison_matrices/test/subset/figures", exist_ok=True)

plot_matrix(
    subset_test_accuracy_matrix,
    title="Test Set Subset Accuracy Comparison",
    filename="comparison_matrices/test/subset/figures/subset_test_accuracy_matrix_heatmap.png"
)

plot_matrix(
    subset_test_kappa_matrix,
    title="Test Set Subset Cohen's Kappa Comparison",
    filename="comparison_matrices/test/subset/figures/subset_test_kappa_matrix_heatmap.png"
)

print("✅ Finished processing and saving test set matrices.")