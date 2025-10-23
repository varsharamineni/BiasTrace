"""
Inter-Annotator Agreement Analysis Script (Custom Headings)
-----------------------------------------------------------
- Works with CSVs that have:
  labeller, sample_id, 8 binary label columns
- Computes Fleiss' Kappa, Krippendorff's Alpha per label
- Computes overall agreement
- Computes pairwise Cohen's kappa per label
"""

import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
from sklearn.metrics import cohen_kappa_score
import krippendorff
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

# ==============================
# 1. LOAD DATA
# ==============================

df = pd.read_csv("reasoning_error_taxonomy/updated_initial_annotations.csv")  # replace with your CSV file path
print("✅ Loaded data:")
print(df.head(), "\n")

annotator_col = "labeller"
item_col = "sample_id"

# Dynamically detect label columns (all except annotator and item)
labels = [c for c in df.columns if c not in [annotator_col, item_col]]

# Detect unique annotators dynamically
annotators = df[annotator_col].unique()
print(f"Detected {len(annotators)} annotators: {annotators}\n")

# ==============================
# 2. AGREEMENT FUNCTIONS
# ==============================

def fleiss_kappa_for_label(df, label):
    pivot = df.pivot(index=item_col, columns=annotator_col, values=label)
    pivot = pivot.dropna()
    if pivot.empty:
        return np.nan
    counts = np.apply_along_axis(
        lambda x: np.bincount(x.astype(int), minlength=2),
        axis=1,
        arr=pivot.to_numpy()
    )
    return fleiss_kappa(counts)

def kripp_alpha_for_label(df, label):
    pivot = df.pivot(index=item_col, columns=annotator_col, values=label)
    values = pivot.to_numpy()
    unique_values = np.unique(values)
    if len(unique_values) <= 1:
        print(f"⚠️ Skipping Krippendorff's α for label '{label}': only one unique value {unique_values}")
        return np.nan
    return krippendorff.alpha(values, level_of_measurement='nominal')

def pairwise_cohen_kappas(df, label):
    pivot = df.pivot(index=item_col, columns=annotator_col, values=label)
    kappas = []
    for a1, a2 in combinations(pivot.columns, 2):
        valid = pivot[[a1, a2]].dropna()
        if valid.empty:
            kappa = np.nan
        else:
            kappa = cohen_kappa_score(valid[a1], valid[a2])
        kappas.append({"annotator_1": a1, "annotator_2": a2, "label": label, "cohen_kappa": kappa})
    return kappas

# ==============================
# 3. PER-LABEL AGREEMENT
# ==============================

results = []
pairwise_results = []

for label in labels:
    fk = fleiss_kappa_for_label(df, label)
    ka = kripp_alpha_for_label(df, label)
    results.append({"label": label, "Fleiss_kappa": fk, "Kripp_alpha": ka})
    pairwise_results.extend(pairwise_cohen_kappas(df, label))

results_df = pd.DataFrame(results)
pairwise_df = pd.DataFrame(pairwise_results)

print("📊 Per-label group agreement:")
print(results_df, "\n")

# ==============================
# 4. OVERALL AGREEMENT
# ==============================

flat_df = pd.melt(df, id_vars=[item_col, annotator_col], value_vars=labels,
                  var_name='label_type', value_name='label_value')
pivot_all = flat_df.pivot_table(index=[item_col, 'label_type'], columns=annotator_col, values='label_value')

overall_alpha = krippendorff.alpha(pivot_all.to_numpy(), level_of_measurement='nominal')

# Fleiss’ κ only valid if all items have same number of non-missing annotations
counts_all = np.apply_along_axis(lambda x: np.bincount(x[~np.isnan(x)].astype(int), minlength=2),
                                 axis=1, arr=pivot_all.to_numpy())
overall_kappa = fleiss_kappa(counts_all)

print(f"🏁 Overall Fleiss' κ: {overall_kappa:.3f}")
print(f"🏁 Overall Krippendorff’s α: {overall_alpha:.3f}\n")

# ==============================
# 5. VISUALIZE PER-LABEL AGREEMENT
# ==============================

res_melt = results_df.melt(id_vars='label', var_name='metric', value_name='score')
sns.barplot(data=res_melt, x='label', y='score', hue='metric')
plt.title("Inter-Annotator Agreement per Label")
plt.axhline(0.6, color='gray', linestyle='--', label='Substantial agreement threshold')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# 6. PAIRWISE COHEN'S KAPPA HEATMAP
# ==============================

pairwise_avg = (
    pairwise_df.groupby(["annotator_1", "annotator_2"])["cohen_kappa"]
    .mean()
    .reset_index()
)

pivot_pw = pairwise_avg.pivot(index="annotator_1", columns="annotator_2", values="cohen_kappa")

sns.heatmap(pivot_pw, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
plt.title("Average Pairwise Cohen’s κ Between Annotators")
plt.show()

print("🤝 Pairwise Cohen’s κ per label:")
print(pairwise_df.head(12))
