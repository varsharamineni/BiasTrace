import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np

# ---------------------------
# 1. Load JSON helpers
# ---------------------------
def load_label_json(file_path, col_name, field="bias_label"):
    """
    Generic loader for JSON metrics (0/1 or other labels)
    """

    if "05" in col_name or col_name == "bias01_pathways_simple_diff_pathways_ab2":
        field = "score"
    else:
        field = "bias_label"

    with open(file_path, "r") as f:
        data = json.load(f)

    rows = []
    for item in data.get("results", []):
        rows.append({"sample_id": item["sample_id"], col_name: item.get("judge_output", {}).get(field, None)})
    return pd.DataFrame(rows)

def load_metadata_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    rows = []
    for item in data:
        rows.append({
            "sample_id": item["sample_id"],
            "is_correct": item.get("is_correct"),
            "correct_answer": item.get("correct_answer"),
            "model_answer": item.get("model_answer"),
            "incorrect_and_stereotype": item.get("incorrect_and_stereotype", False),
            "stereotype_aligned": item.get("stereotype_aligned", False)
        })
    return pd.DataFrame(rows)

# ---------------------------
# 2. Define metrics
# ---------------------------
metrics = [
    {"file": "reasoning_eval/llm_judge_samples/test_set/baseline_labels/llm_eval_deepseek-chat_baseline_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "baseline05"},
    {"file": "reasoning_eval/llm_judge_samples/test_set/baseline_labels/llm_eval_deepseek-chat_llama70B_gt_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "baseline01"},
    {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_metric_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias01"},
    {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_metric1_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias01_full"},
    {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_metric2_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias01_full_relation"},
    {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_metric3_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias01_relation"},
     {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_metric4_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias01_overthink"},
     {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_metric5_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias05"},
      {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_metric6_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias01_reason"},
     {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_metric7_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias01_reason1"},
    {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_prompt_bias_pathways_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias01_pathways"},
     {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_prompt_overthink_detailed_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias01_overthink_detailed"},
        {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_prompt_bias_pathways_simple_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias01_pathways_simple"},
            {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_prompt_bias_pathways_simple_example_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias01_pathways_simple_example"},
        {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_prompt_bias_pathways_simple_nolead_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias01_pathways_simple_noload"},
         {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_prompt_bias_pathways_simple_diff_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias01_pathways_simple_diff_pathways"},
        {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_prompt_bias_pathways_simple_ab1_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias01_pathways_simple_diff_pathways_ab1"},
     {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_prompt_bias_pathways_simple_ab2_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias01_pathways_simple_diff_pathways_ab2"},
     {"file": "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_prompt_bias_pathways_simple_structure_temp1.0_top_p0.9_seed42_max_tokens2048.json",
     "col": "bias01_pathways_simple_diff_pathways_structure"}
]

# ---------------------------
# 3. Load and merge all data
# ---------------------------
metadata = load_metadata_json("reasoning_eval/ground_truth_samples/test_set.json")
df = metadata.copy()


for metric in metrics:
    loader = metric.get("loader", load_label_json)  # default loader
    df_metric = loader(metric["file"], metric["col"])
    df = df.merge(df_metric, on="sample_id", how="left")
    
    # Automatically create binned version if metric has "05"
    if "05" in metric["col"]:
        bin_col = metric["col"] + "_bin"
        df[bin_col] = (df[metric["col"]] > 0).astype(int)
# ---------------------------
# 4. Derived columns
# ---------------------------
df["incorrect"] = (~df["is_correct"]).astype(int)
df["incorrect_and_stereotype"] = ((~df["is_correct"]) & (df["stereotype_aligned"])).astype(int)
#df["baseline05_bin"] = (df.get("baseline05", 0) > 0).astype(int)

# ---------------------------
# 5. Correlations
# ---------------------------
corr_cols = [m["col"] for m in metrics] + [m["col"]+"_bin" for m in metrics if "05" in m["col"]] + ["incorrect", "incorrect_and_stereotype"]
print("\nCorrelation matrix:")
print(df[corr_cols].corr())

# ---------------------------
# 6. Mean correctness by metric
# ---------------------------
print("\nMean correctness by label:")
for col in [m["col"] for m in metrics]:
    print(f"\n{col}:")
    print(df.groupby(col)["is_correct"].mean())

# ---------------------------
# 7. Mean incorrect by metric
# ---------------------------
print("\nMean incorrect and incorrect_and_stereotype by label:")
for col in [m["col"] for m in metrics]:
    print(f"\n{col}:")
    print(df.groupby(col)[["incorrect", "incorrect_and_stereotype"]].mean())

# ---------------------------
# 8. Plots
# ---------------------------
sns.set(style="whitegrid")
for col in [m["col"] for m in metrics]:
    plt.figure(figsize=(6,4))
    sns.barplot(x=col, y="is_correct", data=df)
    plt.title(f"Answer correctness vs {col}")
    plt.ylabel("Fraction Correct")
    plt.xlabel(col)
    plt.show()

# ---------------------------
# 9. Cramér's V
# ---------------------------
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r-1, k-1))))

categorical_cols = corr_cols
cramers_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)
for col1 in categorical_cols:
    for col2 in categorical_cols:
        cramers_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
cramers_matrix = cramers_matrix.astype(float)
print("\nCramér's V matrix:")
print(cramers_matrix)


# ---------------------------
# 10. Cohen's d
# ---------------------------
def cohens_d(x, y):
    """
    Compute Cohen's d between two groups x and y
    """
    x = np.array(x.dropna())
    y = np.array(y.dropna())
    
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan

    mean_x, mean_y = np.mean(x), np.mean(y)
    std_x, std_y = np.std(x, ddof=1), np.std(y, ddof=1)

    # pooled std
    pooled_std = np.sqrt(((nx - 1)*std_x**2 + (ny - 1)*std_y**2) / (nx + ny - 2))
    
    if pooled_std == 0:
        return np.nan

    return (mean_x - mean_y) / pooled_std

print("\nCohen's d (effect size):")

outcomes = ["is_correct", "incorrect", "incorrect_and_stereotype"]

for col in [m["col"] for m in metrics]:
    
    # Use binarized version if needed
    if df[col].nunique() > 2:
        continue  # skip non-binary for now (or handle separately)

    print(f"\nMetric: {col}")
    
    group0 = df[df[col] == 0]
    group1 = df[df[col] == 1]

    for outcome in outcomes:
        d = cohens_d(group0[outcome], group1[outcome])
        print(f"  {outcome}: d = {d:.4f}")


# ---------------------------
# 11. Cohen's d with flexible binarization
# ---------------------------
def binarize_metric(series):
    """
    Binarize a metric for effect size calculation:
    - If integer 0/1 → return as-is
    - If 0-5 metric → 0 if 0, 1 if >0
    - If float → 0 if <0.5, 1 if >=0.5
    """
    if series.dtype.kind in 'biu':  # already int/bool
        if series.nunique() == 2:
            return series
        elif series.max() <= 5:  # assume 0-5 scale
            return (series > 0).astype(int)
        else:  # other integers, treat >0 as 1
            return (series > 0).astype(int)
    else:  # float
        return (series >= 0.5).astype(int)


def cohens_d(x, y):
    """
    Compute Cohen's d between two groups x and y
    """
    x = np.array(x.dropna())
    y = np.array(y.dropna())
    
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan

    mean_x, mean_y = np.mean(x), np.mean(y)
    std_x, std_y = np.std(x, ddof=1), np.std(y, ddof=1)

    pooled_std = np.sqrt(((nx - 1)*std_x**2 + (ny - 1)*std_y**2) / (nx + ny - 2))
    
    if pooled_std == 0:
        return np.nan

    return (mean_x - mean_y) / pooled_std


# Compute Cohen's d for all metrics
outcomes = ["is_correct", "incorrect", "incorrect_and_stereotype"]

print("\nCohen's d (effect size) with thresholding:")

for col in [m["col"] for m in metrics]:
    metric_bin = binarize_metric(df[col])
    
    group0 = df[metric_bin == 0]
    group1 = df[metric_bin == 1]
    
    print(f"\nMetric: {col}")
    for outcome in outcomes:
        d = cohens_d(group0[outcome], group1[outcome])
        print(f"  {outcome}: d = {d:.4f}")

# ---------------------------
# 12. Cohen's d matrix
# ---------------------------
outcomes = ["is_correct", "incorrect", "incorrect_and_stereotype"]

# Initialize empty DataFrame
d_matrix = pd.DataFrame(index=[m["col"] for m in metrics], columns=outcomes, dtype=float)

# Compute Cohen's d for each metric x outcome
for col in [m["col"] for m in metrics]:
    metric_bin = binarize_metric(df[col])
    group0 = df[metric_bin == 0]
    group1 = df[metric_bin == 1]
    
    for outcome in outcomes:
        d_matrix.loc[col, outcome] = cohens_d(group0[outcome], group1[outcome])

print("\nCohen's d matrix:")
print(d_matrix)

# Optional: plot as a heatmap
plt.figure(figsize=(10,6))
sns.heatmap(d_matrix, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Cohen's d (Effect Size) Matrix")
plt.ylabel("Metric")
plt.xlabel("Outcome")
plt.show()
