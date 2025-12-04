import json
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, f1_score, cohen_kappa_score

# -------------------------------------------------------
# 1. Load files
# -------------------------------------------------------
with open("reasoning_eval/llm_judge_samples/val_set/baseline/llm_eval_claude-opus-4-1-20250805_baseline.json", "r") as f:
    judge_data = json.load(f)

pred_results = judge_data["results"]
judge_df = pd.json_normalize(pred_results)

# Compute binary predicted bias
if "judge_output.baseline" in judge_df.columns:
    judge_df["pred_bias"] = judge_df["judge_output.baseline"].apply(lambda x: 0 if x == 0 else 1)
else:
    judge_df["pred_bias"] = judge_df["judge_output"].apply(lambda x: 1 if any(v > 0 for v in x.values()) else 0)

judge_df_small = judge_df[["sample_id", "pred_bias"]]

# Ground-truth labels
with open("reasoning_eval/ground_truth_samples/val_set.json", "r") as f:
    gt_data = json.load(f)

gt_df = pd.DataFrame(gt_data)

# -------------------------------------------------------
# 2. Merge on sample_id
# -------------------------------------------------------
df = judge_df_small.merge(gt_df, on="sample_id", how="inner")

# -------------------------------------------------------
# 3. Annotation dimensions + extra fields
# -------------------------------------------------------
dimensions = [
    "group_assumption",
    "bias_acknowledgement",
    "meta_reflection",
    "outside_demo_knowledge",
    "outside_topical_knowledge",
    "unresolved",
    "overthinking",
    "missing_logic",
    "is_correct",
    "stereotype_aligned"
]

# Ensure fields exist and are integers
for dim in dimensions:
    df[dim] = df[dim].astype(int)

# Create is_incorrect by flipping 0/1
df["is_incorrect"] = 1 - df["is_correct"]

# Create combined label: 1 if both is_incorrect and stereotype_aligned are 1
df["is_incorrect_and_stereotype"] = ((df["is_incorrect"] == 1) & (df["stereotype_aligned"] == 1)).astype(int)

dimensions.append("is_incorrect")
dimensions.append("is_incorrect_and_stereotype")

p = df["pred_bias"].to_numpy()

# -------------------------------------------------------
# 4. Compute all metrics in a single loop
# -------------------------------------------------------
results = []

for dim in dimensions:
    y = df[dim].to_numpy()

    # Compute metrics safely
    if y.sum() == 0:
        pearson = np.nan
        mcc = np.nan
        f1 = np.nan
        kappa = np.nan
        p1 = np.nan
        p0 = np.nan
    else:
        pearson = np.corrcoef(p, y)[0, 1]
        mcc = matthews_corrcoef(y, p)
        f1 = f1_score(y, p)
        kappa = cohen_kappa_score(y, p)

        # Conditional probabilities: P(dim=1 | pred_bias=1) and 
        subset_1 = y[p == 1]
        subset_0 = y[p == 0]

        p1_1 = subset_1.mean() if len(subset_1) > 0 else np.nan  # P(dim=1 | pred_bias=1)
        p0_1 = 1 - p1_1 if not np.isnan(p1_1) else np.nan        # P(dim=0 | pred_bias=1)

        p1_0 = subset_0.mean() if len(subset_0) > 0 else np.nan  # P(dim=1 | pred_bias=0)
        p0_0 = 1 - p1_0 if not np.isnan(p1_0) else np.nan        # P(dim=0 | pred_bias=0)
        
    results.append({
        "dimension": dim,
        "pearson_corr": pearson,
        "mcc": mcc,
        "f1_score": f1,
        "cohen_kappa": kappa,
        "P(dim=1 | pred_bias=1)": p1_1,
        "P(dim=0 | pred_bias=1)": p0_1,
        "P(dim=1 | pred_bias=0)": p1_0,
        "P(dim=0 | pred_bias=0)": p0_0
    })

results_df = pd.DataFrame(results)

# -------------------------------------------------------
# 5. Save + display
# -------------------------------------------------------
results_df.to_csv("bias_dimension_correlations_all_metrics.csv", index=False)
print("\n=== Correlation + Interagreement + Conditional Probabilities ===\n")
print(results_df.round(4))
