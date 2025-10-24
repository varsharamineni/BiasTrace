import json
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import argparse

# ----------------------------
# 1️⃣ Parse CLI arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Compare LLM annotations with human labels")
parser.add_argument("--human_file", type=str, required=True, help="Path to human labels JSON")
parser.add_argument("--llm_file", type=str, required=True, help="Path to LLM outputs JSON")
parser.add_argument("--output_prefix", type=str, default="comparison", help="Prefix for output CSV files")
args = parser.parse_args()

# ----------------------------
# 2️⃣ Load JSONs
# ----------------------------
with open(args.human_file, "r") as f:
    human_data = json.load(f)

with open(args.llm_file, "r") as f:
    llm_data = json.load(f)

human_df = pd.DataFrame(human_data)
llm_df = pd.DataFrame(llm_data)

# ----------------------------
# 3️⃣ Normalize columns
# ----------------------------
# Remove leading/trailing spaces and lowercase column names
human_df.columns = human_df.columns.str.strip().str.lower()
llm_df.columns = llm_df.columns.str.strip().str.lower()

# Optional: rename LLM fields to match human fields
llm_rename = {
    "bias_acknowledgment": "bias_acknowledgement",
    "overthink": "overthinking"
}
llm_df = llm_df.rename(columns=llm_rename)

# ----------------------------
# 4️⃣ Align sample_id
# ----------------------------
# LLM has 'index', human has 'sample_id'
llm_df["sample_id"] = human_df["sample_id"].values[:len(llm_df)]
llm_df = llm_df.drop(columns=["index"], errors="ignore")

# ----------------------------
# 5️⃣ Define label columns
# ----------------------------
label_cols = [
    "group_assumption",
    "bias_acknowledgement",
    "meta_reflection",
    "outside_demo_knowledge",
    "outside_topical_knowledge",
    "unresolved",
    "overthinking",
    "missing_logic"
]

# ----------------------------
# 6️⃣ Merge human + LLM
# ----------------------------
comparison_df = human_df.merge(llm_df, on="sample_id", suffixes=("_human", "_llm"))

# ----------------------------
# 7️⃣ Compute metrics
# ----------------------------
accuracy = {}
kappa_dict = {}
pearson_dict = {}
spearman_dict = {}

for col in label_cols:
    human_col = col + "_human"
    llm_col = col + "_llm"

    accuracy[col] = (comparison_df[human_col] == comparison_df[llm_col]).mean()
    kappa_dict[col] = cohen_kappa_score(comparison_df[human_col], comparison_df[llm_col])
    pearson_dict[col] = comparison_df[[human_col, llm_col]].corr(method="pearson").iloc[0, 1]
    spearman_dict[col] = comparison_df[[human_col, llm_col]].corr(method="spearman").iloc[0, 1]

# ----------------------------
# 8️⃣ Extract disagreements
# ----------------------------
disagreements = comparison_df[
    (comparison_df[[c + "_human" for c in label_cols]].values !=
     comparison_df[[c + "_llm" for c in label_cols]].values).any(axis=1)
]

disagree_file = f"{args.output_prefix}_disagreements.csv"
disagreements.to_csv(disagree_file, index=False)
print(f"Disagreements saved to {disagree_file} (n={len(disagreements)})")

# ----------------------------
# 9️⃣ Save merged comparison
# ----------------------------
merged_file = f"{args.output_prefix}_merged.csv"
comparison_df.to_csv(merged_file, index=False)
print(f"Merged comparison saved to {merged_file}")

# ----------------------------
# 🔟 Save metrics CSV
# ----------------------------
metrics_file = f"{args.output_prefix}_metrics.csv"
metrics_df = pd.DataFrame({
    "label": label_cols,
    "accuracy": [accuracy[l] for l in label_cols],
    "cohens_kappa": [kappa_dict[l] for l in label_cols],
    "pearson_corr": [pearson_dict[l] for l in label_cols],
    "spearman_corr": [spearman_dict[l] for l in label_cols]
})
metrics_df.to_csv(metrics_file, index=False)
print(f"Metrics saved to {metrics_file}")

# ----------------------------
# ✅ Done
# ----------------------------
print("\nAll outputs generated successfully.")
