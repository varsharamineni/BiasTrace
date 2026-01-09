import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, matthews_corrcoef
from scipy.stats import pearsonr

# =========================
# PATHS
# =========================
baseline1_path = "reasoning_eval/llm_judge_samples/test_set/baseline/llm_eval_meta-llama_Meta-Llama-3-70B-Instruct_baseline.json"
baseline2_path = "reasoning_eval/llm_judge_samples/test_set/baseline/llm_eval_meta-llama_Meta-Llama-3-70B-Instruct_llama70B_gt.json"
pred_labels_path = "reasoning_eval/llm_judge_samples/test_set/our_labels/llm_eval_meta-llama_Meta-Llama-3-70B-Instruct_detailed_example_clarification_opt_think_fixed.json"
gt_labels_path = "reasoning_eval/ground_truth_samples/test_set.json"

# =========================
# LABELS
# =========================
label_cols = [
    "group_assumption",
    "bias_acknowledgement",
    "meta_reflection",
    "outside_demo_knowledge",
    "outside_topical_knowledge",
    "unresolved",
    "overthinking",
    "missing_logic",
]

# =========================
# LOAD HELPERS
# =========================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# =========================
# LOAD FILES
# =========================
baseline1 = load_json(baseline1_path)
baseline2 = load_json(baseline2_path)
preds = load_json(pred_labels_path)
human = load_json(gt_labels_path)

# =========================
# TO DATAFRAMES
# =========================
df_b1 = pd.DataFrame(baseline1["results"])
df_b2 = pd.DataFrame(baseline2["results"])
df_pred = pd.DataFrame(preds["results"])
df_human = pd.DataFrame(human)

# =========================
# EXTRACT BASELINES
# =========================
df_b1["baseline1"] = df_b1["judge_output"].apply(lambda x: list(x.values())[0])
df_b1["baseline1_bin"] = (df_b1["baseline1"] > 0).astype(int)

df_b2["baseline2"] = df_b2["judge_output"].apply(lambda x: list(x.values())[0])

# =========================
# EXTRACT OUR 8 LABELS
# =========================
for col in label_cols:
    df_pred[col] = df_pred["judge_output"].apply(lambda x: x.get(col, None))

# =========================
# BUILD COMPARISON TABLE
# =========================
df_compare = df_human[["sample_id"] + label_cols].copy()

df_compare = df_compare.merge(
    df_pred[["sample_id"] + label_cols],
    on="sample_id",
    suffixes=("_human", "_our"),
)

df_compare = df_compare.merge(
    df_b1[["sample_id", "baseline1", "baseline1_bin"]],
    on="sample_id",
)

df_compare = df_compare.merge(
    df_b2[["sample_id", "baseline2"]],
    on="sample_id",
)

# =========================
# METRICS FUNCTION
# =========================
def safe_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return {
            "accuracy": (y_true == y_pred).mean(),
            "f1": 0.0,
            "kappa": 0.0,
            "mcc": 0.0,
            "pearson": np.nan,
        }

    return {
        "accuracy": (y_true == y_pred).mean(),
        "f1": f1_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "pearson": pearsonr(y_true, y_pred)[0],
    }

# =========================
# RUN BASELINE vs EACH HUMAN LABEL
# =========================
rows = []

for label in label_cols:
    human_col = f"{label}_human"

    # baseline 1 bin
    m1 = safe_metrics(df_compare[human_col], df_compare["baseline1_bin"])

    # baseline 2
    m2 = safe_metrics(df_compare[human_col], df_compare["baseline2"])

    rows.append({
        "label": label,
        "baseline": "baseline1_bin",
        **m1
    })

    rows.append({
        "label": label,
        "baseline": "baseline2",
        **m2
    })

df_results = pd.DataFrame(rows)


# ============================
# BASELINE 1 vs BASELINE 2
# ============================

baseline_comparison_results = []

m_baselines = safe_metrics(
    df_compare["baseline1_bin"],
    df_compare["baseline2"]
)

baseline_comparison_results.append({
    "comparison": "baseline1_bin_vs_baseline2",
    **m_baselines
})

df_baseline_comparison = pd.DataFrame(baseline_comparison_results)

df_baseline_comparison.to_csv(
    "baseline1_vs_baseline2_metrics.csv",
    index=False
)

print("\n✅ Baseline vs Baseline Metrics:")
print(df_baseline_comparison)

# =========================
# SAVE
# =========================
output_path = "reasoning_eval/baseline_vs_each_human_label_metrics.csv"
df_results.to_csv(output_path, index=False)

print("\n✅ Baselines compared to EACH human label")
print(df_results)
print(f"\n✅ Saved to {output_path}")

# ============================================================
# BUILD PAIRWISE AGREEMENT MATRIX (accuracy – kappa)
# ============================================================

model_score_types = {
    "0-5 bin": "baseline1_bin",           
    "0/1": "baseline2",       
    "GA 0/1": "group_assumption_our",
    "BA 0/1": "bias_acknowledgement_our",
    "MR 0/1": "meta_reflection_our",
    "ODK 0/1": "outside_demo_knowledge_our",
    "OTK 0/1": "outside_topical_knowledge_our",
    "UR 0/1": "unresolved_our",
    "OT 0/1": "overthinking_our",
    "ML 0/1": "missing_logic_our",
    "*GA 0/1": "group_assumption_human",
    "*BA 0/1": "bias_acknowledgement_human",
    "*MR 0/1": "meta_reflection_human",
    "*ODK 0/1": "outside_demo_knowledge_human",
    "*OTK 0/1": "outside_topical_knowledge_human",
    "*UR 0/1": "unresolved_human",
    "*OT 0/1": "overthinking_human",
    "*ML 0/1": "missing_logic_human",
}

# Create empty square dataframe
labels = list(model_score_types.keys())
pairwise_table = pd.DataFrame(index=labels, columns=labels)

for name_i, col_i in model_score_types.items():
    for name_j, col_j in model_score_types.items():

        # Some columns might not be present in df_compare → mark as N/A
        if col_i not in df_compare.columns or col_j not in df_compare.columns:
            pairwise_table.loc[name_i, name_j] = "N/A"
            continue

        metrics = safe_metrics(df_compare[col_i], df_compare[col_j])
        acc = metrics["accuracy"]
        kappa = metrics["kappa"]

        pairwise_table.loc[name_i, name_j] = f"{acc:.3f} | {kappa:.3f}"

# Save
pairwise_path = "reasoning_eval/pairwise_agreement_matrix.csv"
pairwise_table.to_csv(pairwise_path)

print("\n====================================")
print("📊 Pairwise Agreement Matrix")
print("====================================")
print(pairwise_table)
print(f"\nSaved to: {pairwise_path}")
