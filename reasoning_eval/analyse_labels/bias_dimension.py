# reasoning_error_by_category.py
#
# Purpose
# -------
# Answers the reviewer's question:
#   "It would be informative to report if reasoning behaviors are particularly
#    prevalent for certain bias dimensions or rather uniformly represented
#    across dimensions."
#
# Here "bias dimension" == BBQ `category` (Age, Gender_identity,
# Race_ethnicity, Religion, SES, Sexual_orientation, Nationality,
# Physical_appearance, Disability_status, Race_x_gender, Race_x_SES).
#
# The script produces, per BiasTrace label:
#   1. Label rate by category (pooled, and faceted by model x prompt_type).
#   2. A per-label "uniformity profile" (min/max/range/CoV across categories)
#      so the reviewer can see at a glance which labels are dimension-
#      specific vs. flat.
#   3. Label rate by category x outcome_class (correct / incorrect_non_stereotype
#      / biased), which is the version most directly tied to the reviewer's
#      Fig G.2 point about biased-outcome proportions differing across
#      categories.
#   4. Chi-square tests (label x category), BH-corrected across labels, so
#      "particularly prevalent" claims are backed by a p-value.
#
# Outputs go to reasoning_eval/analyse_labels/by_category_plots/.

import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests


# -------------------------
# Load multiple JSON annotation files (same as the overall-plots script)
# -------------------------
def load_judge_files(paths, judge_labels):
    rows = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)

        for r in data["results"]:
            path_str = str(path)
            if "simple_prompt" in path_str:
                prompt_type = "simple_prompt"
            elif "full_prompt" in path_str:
                prompt_type = "full_prompt"
            else:
                prompt_type = r.get("prompt_type", "unknown")

            row = {
                "source_file": path,
                "sample_id": r["sample_id"],
                "category": r.get("category"),
                "example_id": r.get("example_id"),
                "model": r.get("model"),
                "prompt_type": prompt_type,
                "judge_model": r.get("judge_model"),
                "judge_prompt": r.get("judge_prompt"),
                "is_correct": r.get("is_correct"),
                "ambiguous": r.get("ambiguous"),
                "stereotype_alignment": r.get("stereotype_alignment"),
                "incorrect_and_stereotype": r.get("incorrect_and_stereotype"),
            }

            judge_out = r.get("judge_output")
            if judge_out is None:
                for k in judge_labels:
                    row[k] = 0
                row["judge_missing"] = True
            else:
                for k in judge_labels:
                    row[k] = judge_out.get(k, 0)
                row["judge_missing"] = False

            rows.append(row)
    return pd.DataFrame(rows)


# -------------------------
# 1. Load data
# -------------------------
INPUT_GLOBS = (
    "outputs/qwen_full_8B_simple_prompt/**/full_annotation/*/llm_eval_bbq_*.json",
    "outputs/qwen_full_8B_full_prompt/full_annotation/*/llm_eval_bbq_*.json",
    "outputs/qwen_full_14B_simple_prompt/**/full_annotation/*/llm_eval_bbq_*.json",
    "outputs/qwen_full_14B_full_prompt/full_annotation/*/llm_eval_bbq_*.json",
    # "outputs/gpt-oss-120b_simple_prompt_medium_reasoning/**/full_annotation/*/llm_eval_bbq_*.json",
    # "outputs/gpt-oss-120b_simple_prompt_low_reasoning/**/full_annotation/*/llm_eval_bbq_*.json",
    # "outputs/gpt-oss-120b_full_prompt_low_reasoning/**/full_annotation/*/llm_eval_bbq_*.json",
    # "outputs/gpt-oss-120b_full_prompt_medium_reasoning/**/full_annotation/*/llm_eval_bbq_*.json",
)

all_paths = []
for pattern in INPUT_GLOBS:
    all_paths.extend(glob.glob(pattern, recursive=True))

if len(all_paths) == 0:
    raise RuntimeError(f"No files found for input patterns: {INPUT_GLOBS}")

error_cols = [
    "group_assumption",
    "bias_acknowledgement",
    "meta_reflection",
    "outside_demo_knowledge",
    "outside_topical_knowledge",
    # "unresolved",
    "overthinking",
    # "missing_logic",
]

df = load_judge_files(all_paths, judge_labels=error_cols)
print(f"Loaded {len(df)} annotated samples from {len(all_paths)} files")

# Types / cleanup
df[error_cols] = df[error_cols].fillna(0).astype(int)
df["is_correct"] = df["is_correct"].fillna(0).astype(int)
df["incorrect_and_stereotype"] = df.get("incorrect_and_stereotype", 0)
df["incorrect_and_stereotype"] = (
    df["incorrect_and_stereotype"].fillna(0).astype(int)
)


def normalize_model_name(model_str):
    if pd.isna(model_str):
        return model_str
    model_str = model_str.lower()
    if "qwen3-14b" in model_str:
        return "Qwen3-14B"
    return model_str


df["model"] = df["model"].apply(normalize_model_name)

# Drop rows with no category — we can't attribute them to a bias dimension
before = len(df)
df = df[df["category"].notna() & (df["category"].astype(str) != "")].copy()
print(f"Dropped {before - len(df)} rows with no category")

# Order categories by frequency for readable plots
category_order = df["category"].value_counts().index.tolist()
df["category"] = pd.Categorical(df["category"], categories=category_order, ordered=True)

# Outcome class (same three-way definition as the overall script)
outcome_order = ["correct", "incorrect_non_stereotype", "biased"]


def classify_outcome(row):
    if row["is_correct"] == 1:
        return "correct"
    if row.get("incorrect_and_stereotype", 0) == 1:
        return "biased"
    return "incorrect_non_stereotype"


df["outcome_class"] = pd.Categorical(
    df.apply(classify_outcome, axis=1), categories=outcome_order, ordered=True,
)

# Ambiguity flag as clean bool
df["ambiguous_bool"] = df["ambiguous"].astype(str).isin(["True", "true", "1"])

# -------------------------
# 2. Output directory
# -------------------------
output_dir = "reasoning_eval/analyse_labels/by_category_plots"
os.makedirs(output_dir, exist_ok=True)


# -------------------------
# 3. Helpers
# -------------------------
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def rates_with_ci(frame, group_cols, label_cols):
    """Long-format rate table with Wilson 95% CIs. group_cols defines cells."""
    rows = []
    grouped = frame.groupby(group_cols, observed=True)
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(sub)
        for lbl in label_cols:
            k = int(sub[lbl].sum())
            lo, hi = wilson_ci(k, n)
            row = dict(zip(group_cols, keys))
            row.update({
                "label": lbl, "n": n, "k": k,
                "rate": k / n if n else np.nan,
                "ci_low": lo, "ci_high": hi,
            })
            rows.append(row)
    return pd.DataFrame(rows)


# -------------------------
# 4. Cell sizes — reviewer needs to see denominators
# -------------------------
cat_counts = df["category"].value_counts().reindex(category_order)
print("\nSamples per category:")
print(cat_counts)

cat_counts.to_csv(os.path.join(output_dir, "category_counts.csv"))

# Denominators broken down by (category x model x prompt_type x outcome)
denom_full = (
    df.groupby(
        ["category", "model", "prompt_type", "outcome_class"], observed=True,
    )
    .size()
    .unstack(fill_value=0)
)
denom_full.to_csv(os.path.join(output_dir, "cell_counts_full.csv"))


# -------------------------
# 5. Overall label rates by category (pooled across model x prompt_type)
# -------------------------
rates_by_cat = rates_with_ci(df, ["category"], error_cols)
rates_by_cat.to_csv(
    os.path.join(output_dir, "label_rates_by_category.csv"), index=False,
)

# 5a. Heatmap: categories x labels, with rate annotations.
pivot = rates_by_cat.pivot(index="category", columns="label", values="rate")
pivot = pivot.reindex(index=category_order, columns=error_cols)

fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(error_cols)), 0.55 * len(category_order) + 2))
sns.heatmap(
    pivot, annot=True, fmt=".2f", cmap="viridis",
    cbar_kws={"label": "Proportion of traces with label"}, ax=ax,
)
ax.set_title("BiasTrace label rates by BBQ category (pooled)")
ax.set_xlabel("BiasTrace label")
ax.set_ylabel("BBQ category")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "label_rates_by_category_heatmap.png"), dpi=150)
plt.close()

# 5b. Grouped bar: one bar cluster per category, coloured by label.
fig, ax = plt.subplots(figsize=(max(12, 1.1 * len(category_order)), 6))
pivot.plot(kind="bar", ax=ax, width=0.85)
ax.set_title("BiasTrace label rates by BBQ category (pooled)")
ax.set_ylabel("Proportion of traces with label")
ax.set_xlabel("BBQ category")
ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "label_rates_by_category_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

# 5c. Per-label bar with error bars, so reviewers can eyeball significance
n_labels = len(error_cols)
ncols_grid = 3
nrows_grid = int(np.ceil(n_labels / ncols_grid))
fig, axes = plt.subplots(
    nrows_grid, ncols_grid, figsize=(6 * ncols_grid, 4 * nrows_grid), sharey=False,
)
axes = np.array(axes).reshape(-1)
for i, lbl in enumerate(error_cols):
    ax = axes[i]
    sub = rates_by_cat[rates_by_cat["label"] == lbl].set_index("category").reindex(category_order)
    yerr = np.vstack([sub["rate"] - sub["ci_low"], sub["ci_high"] - sub["rate"]])
    yerr = np.clip(yerr, 0, None)
    ax.bar(range(len(sub)), sub["rate"], yerr=yerr, capsize=3, color="steelblue")
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub.index, rotation=45, ha="right", fontsize=9)
    ax.set_title(lbl)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, max(0.05, float(sub["ci_high"].max()) * 1.15))
for j in range(len(error_cols), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Per-label rate across BBQ categories (95% Wilson CI)", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "label_rates_by_category_perlabel.png"),
    dpi=150, bbox_inches="tight",
)
plt.close()


# -------------------------
# 6. Uniformity profile per label
#    Directly answers: is this label prevalent for certain dimensions,
#    or uniformly represented across dimensions?
# -------------------------
uniformity_rows = []
for lbl in error_cols:
    rates = pivot[lbl].dropna()
    mean_r = rates.mean()
    std_r = rates.std(ddof=0)
    min_r = rates.min()
    max_r = rates.max()
    argmin_c = rates.idxmin() if len(rates) else None
    argmax_c = rates.idxmax() if len(rates) else None
    cov = (std_r / mean_r) if mean_r > 0 else np.nan
    fold = (max_r / min_r) if min_r > 0 else np.nan
    uniformity_rows.append({
        "label": lbl,
        "mean_rate": mean_r,
        "std_rate": std_r,
        "cov": cov,                     # coefficient of variation (unitless spread)
        "min_rate": min_r,
        "max_rate": max_r,
        "range": max_r - min_r,
        "fold_max_over_min": fold,      # NaN if some category has zero rate
        "argmin_category": argmin_c,
        "argmax_category": argmax_c,
    })
uniformity_df = pd.DataFrame(uniformity_rows).sort_values("cov", ascending=False)
uniformity_df.to_csv(os.path.join(output_dir, "label_uniformity_across_categories.csv"), index=False)

# Plot: CoV per label — high CoV = dimension-specific, low CoV = uniform.
fig, ax = plt.subplots(figsize=(10, 5))
uniformity_df.plot(x="label", y="cov", kind="bar", ax=ax, color="darkorange", legend=False)
ax.set_ylabel("Coefficient of variation across categories\n(std / mean)")
ax.set_title(
    "How dimension-specific is each BiasTrace label?\n"
    "Higher = more concentrated in some BBQ categories; lower = more uniform."
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "label_uniformity_cov.png"), dpi=150)
plt.close()


# -------------------------
# 7. Statistical tests: label x category
#    For each label, chi-square on the (category x label) contingency table.
#    BH-correct p-values across labels.
# -------------------------
stat_rows = []
for lbl in error_cols:
    tab = pd.crosstab(df["category"], df[lbl])
    if tab.shape[1] == 2 and (tab.values.sum(axis=0) > 0).all():
        chi2, p, dof, _ = chi2_contingency(tab.values)
    else:
        chi2, p, dof = np.nan, np.nan, np.nan
    stat_rows.append({
        "label": lbl, "chi2": chi2, "dof": dof, "p": p,
        "n_categories": tab.shape[0],
    })
stat_df = pd.DataFrame(stat_rows)
mask = stat_df["p"].notna()
if mask.any():
    _, q, _, _ = multipletests(stat_df.loc[mask, "p"].values, method="fdr_bh")
    stat_df.loc[mask, "q_bh"] = q
stat_df.to_csv(os.path.join(output_dir, "label_by_category_stats.csv"), index=False)
print("\nchi-square tests (label x category):")
print(stat_df.to_string(index=False))


# -------------------------
# 8. Label rates by category x outcome_class
#    This is the version most directly tied to the reviewer's Fig G.2 point:
#    biased-outcome proportion varies across categories — do reasoning
#    behaviors within biased outcomes also vary across categories?
# -------------------------
rates_cat_out = rates_with_ci(df, ["category", "outcome_class"], error_cols)
rates_cat_out.to_csv(
    os.path.join(output_dir, "label_rates_by_category_x_outcome.csv"), index=False,
)

# One faceted figure per label so the reviewer can page through them.
for lbl in error_cols:
    sub = rates_cat_out[rates_cat_out["label"] == lbl]
    piv = sub.pivot(index="category", columns="outcome_class", values="rate")[outcome_order]
    lo = sub.pivot(index="category", columns="outcome_class", values="ci_low")[outcome_order]
    hi = sub.pivot(index="category", columns="outcome_class", values="ci_high")[outcome_order]
    piv = piv.reindex(category_order)
    lo = lo.reindex(category_order)
    hi = hi.reindex(category_order)

    yerr = np.array([piv.values - lo.values, hi.values - piv.values])
    yerr = np.transpose(yerr, (2, 0, 1))
    yerr = np.clip(yerr, 0, None)

    fig, ax = plt.subplots(figsize=(max(12, 1.1 * len(category_order)), 6))
    piv.plot(kind="bar", ax=ax, width=0.8, yerr=yerr, capsize=2, error_kw={"elinewidth": 0.8})
    ax.set_title(f"{lbl} rate by BBQ category × outcome class")
    ax.set_ylabel("Proportion of traces with label")
    ax.set_xlabel("BBQ category")
    ax.legend(title="Outcome class")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"label_rate_by_category_x_outcome__{lbl}.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()


# -------------------------
# 9. Faceted view: label rates by category, per (model x prompt_type)
#    Sanity check that any dimension-specific pattern isn't just a
#    pooling artefact.
# -------------------------
df_long = df.melt(
    id_vars=["model", "prompt_type", "category"],
    value_vars=error_cols, var_name="label", value_name="present",
)

g = sns.catplot(
    data=df_long,
    x="category", y="present", hue="label",
    col="prompt_type", row="model",
    kind="bar", errorbar=("ci", 95),
    height=4.5, aspect=2.0, sharey=True,
)
g.set_xticklabels(rotation=45, ha="right")
g.set_axis_labels("BBQ category", "Proportion")
g.fig.suptitle(
    "BiasTrace label rates by BBQ category, faceted by model × prompt_type",
    y=1.02, fontsize=13,
)
g.savefig(
    os.path.join(output_dir, "label_rates_by_category_facets_model_prompt.png"),
    dpi=150, bbox_inches="tight",
)
plt.close("all")


# -------------------------
# 10. Ambiguity split — pooled by category x ambiguity
#     Optional but often the reviewer's next question: does the
#     dimension-specificity hold in the ambiguous vs. disambiguated slice?
# -------------------------
rates_cat_amb = rates_with_ci(df, ["category", "ambiguous_bool"], error_cols)
rates_cat_amb.to_csv(
    os.path.join(output_dir, "label_rates_by_category_x_ambiguity.csv"), index=False,
)

fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
for ax, amb in zip(axes, [True, False]):
    sub = rates_cat_amb[rates_cat_amb["ambiguous_bool"] == amb]
    piv = sub.pivot(index="category", columns="label", values="rate").reindex(
        index=category_order, columns=error_cols,
    )
    piv.plot(kind="bar", ax=ax, width=0.85)
    counts = df[df["ambiguous_bool"] == amb]["category"].value_counts().reindex(category_order)
    ax.set_title(f"{'Ambiguous' if amb else 'Disambiguated'} — n per category shown below")
    ax.set_ylabel("Proportion of traces with label" if amb else "")
    ax.set_xlabel("BBQ category")
    ax.set_xticklabels(
        [f"{c}\n(n={int(counts[c])})" for c in category_order],
        rotation=45, ha="right", fontsize=9,
    )
    if amb:
        ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    else:
        ax.get_legend().remove()

plt.suptitle("BiasTrace label rates by BBQ category × ambiguity", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "label_rates_by_category_x_ambiguity.png"),
    dpi=150, bbox_inches="tight",
)
plt.close()


print(f"\nAll outputs written to: {output_dir}")