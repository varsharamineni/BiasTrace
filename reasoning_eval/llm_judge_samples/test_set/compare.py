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

    if "05" in col_name:
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
     "col": "bias01_pathways_simple"}
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
