import json
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, cohen_kappa_score

# =====================================================
# 1. Load Judge JSON
# =====================================================
with open(
    "reasoning_eval/llm_judge_samples/val_set/llm_eval_claude-opus-4-1-20250805_detailed_example_clarification_opt_fixed.json",
    "r"
) as f:
    judge_data = json.load(f)

judge_df = pd.json_normalize(judge_data["results"], sep="_")

# =====================================================
# 2. Define Judge Dimensions
# =====================================================
dimensions_to_aggregate = [
    'judge_output_group_assumption',
    "judge_output_bias_acknowledgement",
    "judge_output_meta_reflection",
    "judge_output_outside_demo_knowledge",
    "judge_output_outside_topical_knowledge",
    "judge_output_unresolved",
    "judge_output_overthinking",
    "judge_output_missing_logic"
]

for dim in dimensions_to_aggregate:
    judge_df[dim] = judge_df[dim].fillna(0).astype(int)

# =====================================================
# 3. Aggregations
# =====================================================

# --- Signed weighted aggregation (-1 for bias acknowledgment)
weighted_df = judge_df[dimensions_to_aggregate].copy()
weighted_df["judge_output_bias_acknowledgement"] *= -1

judge_df["error_score"] = weighted_df.sum(axis=1)

# 1) Binary (net positive failure pressure)
judge_df["any_error_binary"] = (judge_df["error_score"] > 0).astype(int)

# 2) Raw signed severity
judge_df["any_error_signed"] = judge_df["error_score"]

# 3) Pure count (no -1 mitigation)
judge_df["any_error_count"] = judge_df[dimensions_to_aggregate].sum(axis=1)

# 4) Strict threshold: 2+ raw errors
judge_df["any_error_strict"] = (judge_df["any_error_count"] >= 2).astype(int)

# =====================================================
# 4. Load Ground Truth
# =====================================================
with open("reasoning_eval/ground_truth_samples/val_set.json", "r") as f:
    gt_data = json.load(f)

gt_df = pd.DataFrame(gt_data)

for col in ["is_correct", "stereotype_aligned"]:
    gt_df[col] = gt_df.get(col, 0).fillna(0).astype(int)

gt_df["is_incorrect"] = 1 - gt_df["is_correct"]

gt_df["is_incorrect_and_stereotype"] = (
    (gt_df["is_incorrect"] == 1) &
    (gt_df["stereotype_aligned"] == 1)
).astype(int)

# =====================================================
# 5. Merge
# =====================================================
df = judge_df[
    [
        "sample_id",
        "any_error_binary",
        "any_error_signed",
        "any_error_count",
        "any_error_strict",
    ]
].merge(gt_df, on="sample_id", how="inner")

# =====================================================
# 6. Statistical Utilities
# =====================================================
def bootstrap_mcc(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    scores = []
    idx = np.arange(len(y))

    for _ in range(n):
        sample = rng.choice(idx, size=len(idx), replace=True)
        scores.append(matthews_corrcoef(y[sample], p[sample]))

    return np.percentile(scores, [2.5, 97.5])


def permutation_test_mcc(y, p, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    true_mcc = matthews_corrcoef(y, p)
    null = []

    for _ in range(n):
        null.append(matthews_corrcoef(y, rng.permutation(p)))

    null = np.array(null)
    pval = (null >= true_mcc).mean()
    return true_mcc, pval

# =====================================================
# 7. Unified Evaluation
# =====================================================
dimensions = [
    "is_correct",
    "is_incorrect",
    "stereotype_aligned",
    "is_incorrect_and_stereotype"
]

prediction_types = [
    "any_error_binary",
    "any_error_strict"
]

results = []

for pred_col in prediction_types:

    p = df[pred_col].to_numpy()

    for dim in dimensions:
        y = df[dim].to_numpy()

        if y.sum() == 0 or p.sum() == 0:
            continue

        # Phi / Pearson
        pearson = np.corrcoef(p, y)[0, 1]
        mcc = matthews_corrcoef(y, p)
        kappa = cohen_kappa_score(y, p)

        # Direction-safe F1
        if dim == "is_correct":
            f1 = f1_score(1 - y, p)
        else:
            f1 = f1_score(y, p)

        # Conditional probabilities
        subset_1 = y[p == 1]
        subset_0 = y[p == 0]

        p1_1 = subset_1.mean()
        p0_1 = 1 - p1_1
        p1_0 = subset_0.mean()
        p0_0 = 1 - p1_0

        # Significance + CI
        mcc_ci_low, mcc_ci_high = bootstrap_mcc(y, p)
        _, mcc_pval = permutation_test_mcc(y, p)

        results.append({
            "prediction_type": pred_col,
            "dimension": dim,
            "pearson_phi": pearson,
            "mcc": mcc,
            "mcc_ci_low": mcc_ci_low,
            "mcc_ci_high": mcc_ci_high,
            "mcc_perm_pval": mcc_pval,
            "f1_score": f1,
            "kappa": kappa,
            "P(dim=1 | pred=1)": p1_1,
            "P(dim=0 | pred=1)": p0_1,
            "P(dim=1 | pred=0)": p1_0,
            "P(dim=0 | pred=0)": p0_0
        })

# =====================================================
# 8. Save + Display
# =====================================================
results_df = pd.DataFrame(results)
results_df.to_csv("any_error_aggregation_comparison.csv", index=False)

print("\n=== Aggregation Comparison Results ===\n")
print(results_df.round(4))
