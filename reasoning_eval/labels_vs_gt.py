import json
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, cohen_kappa_score

# -------------------------
# 1. Load Judge JSON
# -------------------------
with open("reasoning_eval/llm_judge_samples/val_set/llm_eval_claude-opus-4-1-20250805_detailed_example_clarification_opt_fixed.json", "r") as f:  # replace with your file path
    judge_data = json.load(f)

# Flatten results including nested judge_output
judge_df = pd.json_normalize(judge_data["results"], sep="_")

# Judge dimensions
dimensions_to_aggregate = [
    "judge_output_meta_reflection",
    "judge_output_outside_demo_knowledge",
    "judge_output_outside_topical_knowledge",
    "judge_output_unresolved",
    "judge_output_overthinking",
    "judge_output_missing_logic"
]

# Ensure integers
for dim in dimensions_to_aggregate:
    judge_df[dim] = judge_df[dim].fillna(0).astype(int)

# -------------------------
# 2. Aggregate into any_error
# -------------------------
# Flag if more than 1 dimension is marked
judge_df["any_error"] = (judge_df[dimensions_to_aggregate].sum(axis=1) > 1).astype(int)


# -------------------------------------------------------
# 3. Load ground truth JSON
# -------------------------------------------------------
with open("reasoning_eval/ground_truth_samples/val_set.json", "r") as f:  # Replace with your path
    gt_data = json.load(f)

gt_df = pd.DataFrame(gt_data)

# Ensure ground truth labels exist
for col in ["is_correct", "stereotype_aligned"]:
    gt_df[col] = gt_df.get(col, 0).fillna(0).astype(int)

# Create is_incorrect
gt_df["is_incorrect"] = 1 - gt_df["is_correct"]

# Create combined label: 1 if both is_incorrect and stereotype_aligned are 1
gt_df["is_incorrect_and_stereotype"] = ((gt_df["is_incorrect"] == 1) & (gt_df["stereotype_aligned"] == 1)).astype(int)

# -------------------------
# 4. Merge
# -------------------------
df = judge_df[["sample_id", "any_error"]].merge(gt_df, on="sample_id", how="inner")

# -------------------------
# 5. Prepare dimensions for analysis
# -------------------------
dimensions = ["is_correct", "is_incorrect", "stereotype_aligned", "is_incorrect_and_stereotype"]
p = df["any_error"].to_numpy()  # predicted error
p = df["any_error"].to_numpy()  # Use any_error as "prediction"

# -------------------------------------------------------
# 6. Compute metrics + conditional probabilities
# -------------------------------------------------------
results = []

for dim in dimensions:
    y = df[dim].to_numpy()

    if y.sum() == 0:
        pearson = np.nan
        mcc = np.nan
        f1 = np.nan
        kappa = np.nan
        p1_1 = np.nan
        p0_1 = np.nan
        p1_0 = np.nan
        p0_0 = np.nan
    else:
        pearson = np.corrcoef(p, y)[0, 1]
        mcc = matthews_corrcoef(y, p)
        f1 = f1_score(y, p)
        kappa = cohen_kappa_score(y, p)

        # Conditional probabilities: P(dim=1 | any_error=1), etc
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
        "P(dim=1 | any_error=1)": p1_1,
        "P(dim=0 | any_error=1)": p0_1,
        "P(dim=1 | any_error=0)": p1_0,
        "P(dim=0 | any_error=0)": p0_0
    })

# -------------------------------------------------------
# 7. Save + display
# -------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("any_error_analysis.csv", index=False)

print("\n=== Correlations + Conditional Probabilities ===\n")
print(results_df.round(4))
