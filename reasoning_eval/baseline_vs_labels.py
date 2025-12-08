import json
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, f1_score, cohen_kappa_score

# -------------------------------------------------------
# 1. Load files
# -------------------------------------------------------
JUDGE_FILE = "reasoning_eval/llm_judge_samples/test_set/baseline/llm_eval_meta-llama_Meta-Llama-3-70B-Instruct_baseline.json"
GT_FILE = "reasoning_eval/ground_truth_samples/test_set.json"

with open(JUDGE_FILE, "r") as f:
    judge_data = json.load(f)

pred_results = judge_data["results"]
judge_df = pd.json_normalize(pred_results)

# -------------------------------------------------------
# 2. Identify judge_output column dynamically
# -------------------------------------------------------
judge_output_cols = [c for c in judge_df.columns if c.startswith("judge_output.")]
if len(judge_output_cols) == 0:
    raise ValueError("No judge_output column found in the judge data.")
judge_output_col = judge_output_cols[0]  # Use the first found column

# Compute binary predicted bias
judge_df["pred_bias"] = judge_df[judge_output_col].apply(lambda x: 0 if x == 0 else 1)

# Keep minimal columns for merge
judge_df_small = judge_df[["sample_id", "pred_bias"]]

# -------------------------------------------------------
# 3. Load ground-truth labels
# -------------------------------------------------------
with open(GT_FILE, "r") as f:
    gt_data = json.load(f)

gt_df = pd.DataFrame(gt_data)

# -------------------------------------------------------
# 4. Merge on sample_id
# -------------------------------------------------------
df = judge_df_small.merge(gt_df, on="sample_id", how="inner")

# -------------------------------------------------------
# 5. Annotation dimensions + extra fields
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

# Ensure all dimensions exist and are integers
for dim in dimensions:
    if dim not in df.columns:
        df[dim] = 0  # default if missing
    df[dim] = df[dim].astype(int)

# Additional derived fields
df["is_incorrect"] = 1 - df["is_correct"]
df["is_incorrect_and_stereotype"] = ((df["is_incorrect"] == 1) & (df["stereotype_aligned"] == 1)).astype(int)

# Append new dimensions
dimensions += ["is_incorrect", "is_incorrect_and_stereotype"]

# Binary predictions
p = df["pred_bias"].to_numpy()

# -------------------------------------------------------
# 6. Compute all metrics
# -------------------------------------------------------
results = []

for dim in dimensions:
    y = df[dim].to_numpy()

    if y.sum() == 0:
        pearson = mcc = f1 = kappa = np.nan
        p1_1 = p0_1 = p1_0 = p0_0 = np.nan
    else:
        pearson = np.corrcoef(p, y)[0, 1]
        mcc = matthews_corrcoef(y, p)
        f1 = f1_score(y, p)
        kappa = cohen_kappa_score(y, p)

        # Conditional probabilities
        subset_1 = y[p == 1]
        subset_0 = y[p == 0]

        p1_1 = subset_1.mean() if len(subset_1) > 0 else np.nan
        p0_1 = 1 - p1_1 if not np.isnan(p1_1) else np.nan

        p1_0 = subset_0.mean() if len(subset_0) > 0 else np.nan
        p0_0 = 1 - p1_0 if not np.isnan(p1_0) else np.nan

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
# 7. Save + display
# -------------------------------------------------------
OUTPUT_CSV = "bias_dimension_correlations_all_metrics.csv"
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved results to {OUTPUT_CSV}\n")
print(results_df.round(4))
