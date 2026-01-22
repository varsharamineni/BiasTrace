import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np 

# ---------------------------
# 1. Load JSON files
# ---------------------------

def load_baseline05_json(file_path):
    """
    Loads baseline 0-5 JSON file (score field)
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    rows = []
    for item in data["results"]:
        sample_id = item["sample_id"]
        score = item.get("judge_output", {}).get("score", None)
        rows.append({"sample_id": sample_id, "baseline05": score})
    return pd.DataFrame(rows)


def load_label01_json(file_path, label_name):
    """
    Loads baseline 0/1 or bias 0/1 JSON file (judge_output['bias_label'])
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    rows = []
    for item in data["results"]:
        sample_id = item["sample_id"]
        bias_label = item.get("judge_output", {}).get("bias_label", None)
        rows.append({"sample_id": sample_id, label_name: bias_label})
    return pd.DataFrame(rows)


def load_metadata_json(file_path):
    """
    Loads metadata JSON and returns DataFrame with sample_id and correctness info.
    """
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
# 2. Load your files
# ---------------------------
baseline05 = load_baseline05_json(
    "reasoning_eval/llm_judge_samples/test_set/baseline_labels/llm_eval_deepseek-chat_baseline_temp1.0_top_p0.9_seed42_max_tokens2048.json"
)
baseline01 = load_label01_json(
    "reasoning_eval/llm_judge_samples/test_set/baseline_labels/llm_eval_deepseek-chat_llama70B_gt_temp1.0_top_p0.9_seed42_max_tokens2048.json",
    "baseline01"
)
bias01 = load_label01_json(
    "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_metric_temp1.0_top_p0.9_seed42_max_tokens2048.json",
    "bias01"
)
bias01_full = load_label01_json(
    "reasoning_eval/llm_judge_samples/test_set/new_metric_labels/llm_eval_deepseek-chat_new_metric1_temp1.0_top_p0.9_seed42_max_tokens2048.json",
    "bias01_full"
)
metadata = load_metadata_json(
    "reasoning_eval/ground_truth_samples/test_set.json"
)

# ---------------------------
# 3. Merge all data
# ---------------------------
df = metadata.merge(baseline01, on="sample_id", how="left")
df = df.merge(baseline05, on="sample_id", how="left")
df = df.merge(bias01, on="sample_id", how="left")
df = df.merge(bias01_full, on="sample_id", how="left")

print("Merged DataFrame head:")
print(df.head())

# ---------------------------
# 4. Correlation between labels
# ---------------------------
print("\nCorrelation between labels:")
print(df[["baseline01", "baseline05", "bias01", "bias01_full"]].corr())

# ---------------------------
# 5. Relation of labels to correctness
# ---------------------------
print("\nMean correctness by label:")
for col in ["baseline01", "baseline05", "bias01", "bias01_full"]:
    print(f"\n{col}:")
    print(df.groupby(col)["is_correct"].mean())

# ---------------------------
# 6. Cross-tab analysis (joint label effects)
# ---------------------------
print("\nMean correctness by baseline01 x bias01:")
print(pd.crosstab(df["baseline01"], df["bias01"], values=df["is_correct"], aggfunc="mean"))

print("\nMean correctness by baseline01 x bias01_full:")
print(pd.crosstab(df["baseline01"], df["bias01_full"], values=df["is_correct"], aggfunc="mean"))

# ---------------------------
# 7. Visualizations
# ---------------------------
sns.set(style="whitegrid")

# Boxplot for baseline05
plt.figure(figsize=(8,5))
sns.boxplot(x="baseline05", y="is_correct", data=df)
plt.title("Answer correctness vs Baseline 0-5")
plt.ylabel("Fraction Correct")
plt.xlabel("Baseline 0-5 label")
plt.show()

# Barplot for baseline01
plt.figure(figsize=(6,4))
sns.barplot(x="baseline01", y="is_correct", data=df)
plt.title("Answer correctness vs Baseline 0/1")
plt.ylabel("Fraction Correct")
plt.xlabel("Baseline 0/1 label")
plt.show()

# Barplot for bias01
plt.figure(figsize=(6,4))
sns.barplot(x="bias01", y="is_correct", data=df)
plt.title("Answer correctness vs Bias 0/1")
plt.ylabel("Fraction Correct")
plt.xlabel("Bias label 0/1")
plt.show()

# Barplot for bias01_full
plt.figure(figsize=(6,4))
sns.barplot(x="bias01_full", y="is_correct", data=df)
plt.title("Answer correctness vs Bias01 Full")
plt.ylabel("Fraction Correct")
plt.xlabel("Bias01 Full label")
plt.show()

# ---------------------------
# 8. Add incorrect and incorrect & stereotype columns
# ---------------------------
df["incorrect"] = (~df["is_correct"]).astype(int)
df["incorrect_and_stereotype"] = ((~df["is_correct"]) & (df["stereotype_aligned"])).astype(int)

# Add binary baseline05 for correlation
df['baseline05_bin'] = (df['baseline05'] > 0).astype(int)

# ---------------------------
# 9. Correlation between labels and incorrect metrics
# ---------------------------
corr_cols = ["baseline05", "baseline05_bin", "baseline01", "bias01", "bias01_full", "incorrect", "incorrect_and_stereotype"]
print("\nCorrelation matrix:")
print(df[corr_cols].corr())

# ---------------------------
# 10. Mean values by label
# ---------------------------
print("\nMean incorrect and incorrect_and_stereotype by label:")

for col in ["baseline01", "baseline05", "bias01", "bias01_full"]:
    print(f"\n{col}:")
    print(df.groupby(col)[["incorrect", "incorrect_and_stereotype"]].mean())

# ---------------------------
# 11. Cramér's V for categorical columns
# ---------------------------
def cramers_v(x, y):
    """
    Compute Cramér's V statistic for categorical-categorical association.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r-1, k-1))))

categorical_cols = ["baseline01", "bias01", "bias01_full", "incorrect", "incorrect_and_stereotype", "baseline05_bin"]

cramers_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)

for col1 in categorical_cols:
    for col2 in categorical_cols:
        cramers_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

cramers_matrix = cramers_matrix.astype(float)
print("Cramér's V matrix:")
print(cramers_matrix)
