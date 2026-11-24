# reasoning_error_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import scipy.spatial.distance as ssd
import numpy as np
import os

import glob
import json
from pathlib import Path


# -------------------------
# Load multiple JSON annotation files (like previous script)
# -------------------------
def load_judge_files(paths, judge_labels):
    rows = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        for r in data["results"]:

            # Convert path to string
            path_str = str(path)
            
            # Detect prompt_type from path
            if "simple_prompt" in path_str:
                prompt_type = "simple_prompt"
            elif "full_prompt" in path_str:
                prompt_type = "full_prompt"
            else:
                prompt_type = r.get("prompt_type", "unknown")  # fallback

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
# 1️⃣ Load multiple JSON files instead of CSV
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

# Collect all files
all_paths = []
for pattern in INPUT_GLOBS:
    all_paths.extend(glob.glob(pattern, recursive=True))

if len(all_paths) == 0:
    raise RuntimeError(f"No files found for input patterns: {INPUT_GLOBS}")

error_cols = [
    'group_assumption',
    'bias_acknowledgement',
    'meta_reflection',
    'outside_demo_knowledge',
    'outside_topical_knowledge',
    #'unresolved',
    'overthinking'
    #'missing_logic'
]

df = load_judge_files(all_paths, judge_labels=error_cols)
print(f"Loaded {len(df)} annotated samples from {len(all_paths)} files")
print(df.head())

# Fill NaNs in error labels with 0
df[error_cols] = df[error_cols].fillna(0).astype(int)
df["is_correct"] = df["is_correct"].fillna(0).astype(int)

# Optional: ambiguous as categorical
if "ambiguous" in df.columns:
    df["ambiguous"] = df["ambiguous"].astype("category")

# Normalize model names
def normalize_model_name(model_str):
    if pd.isna(model_str):
        return model_str
    model_str = model_str.lower()
    if "qwen3-14b" in model_str:
        return "Qwen3-14B"
    return model_str

df['model'] = df['model'].apply(normalize_model_name)



# -------------------------
# 2️⃣ Define error columns
# -------------------------

# Normalize model names
def normalize_model_name(model_str):
    if pd.isna(model_str):
        return model_str
    model_str = model_str.lower()
    if "qwen3-14b" in model_str:
        return "Qwen3-14B"
    return model_str

df['model'] = df['model'].apply(normalize_model_name)

# Ensure output folder exists
output_dir = "reasoning_eval/analyse_labels/overall_plots"
os.makedirs(output_dir, exist_ok=True)

# Fill NaNs in error labels with 0
df[error_cols] = df[error_cols].fillna(0)

# -------------------------
# 3️⃣ Error frequencies
# -------------------------
error_freq = df[error_cols].mean().sort_values(ascending=False)
plt.figure(figsize=(12,6))
error_freq.plot(kind='bar')
plt.title("Frequency of reasoning errors", fontsize=16)
plt.ylabel("Proportion of examples with error")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "error_frequency.png"))
plt.close()

# -------------------------
# 4️⃣ Mutual Information with correctness
# -------------------------
mi_df = df[error_cols + ['is_correct']].dropna()
mi_scores = {err: mutual_info_score(mi_df[err], mi_df['is_correct']) for err in error_cols}
mi_series = pd.Series(mi_scores).sort_values(ascending=False)

plt.figure(figsize=(12,6))
mi_series.plot(kind='bar', color='orange')
plt.title("Mutual Information between errors and correctness", fontsize=16)
plt.ylabel("MI (bits)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mutual_information.png"))
plt.close()

# -------------------------
# 5️⃣ Correlation matrix of errors
# -------------------------
corr_matrix = df[error_cols + ['is_correct']].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation between reasoning errors", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "error_correlation_heatmap.png"))
plt.close()

# -------------------------
# 6️⃣ Error rates by model
# -------------------------
model_error_rates = df.groupby('model')[error_cols].mean()
plt.figure(figsize=(14,8))
model_error_rates.plot(kind='bar')
plt.title("Reasoning error rates per model", fontsize=16)
plt.ylabel("Proportion of examples with error")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_error_rates.png"))
plt.close()

# -------------------------
# 7️⃣ Error rates by prompt type
# -------------------------
if 'prompt_type' in df.columns:
    prompt_error_rates = df.groupby('prompt_type')[error_cols].mean()
    plt.figure(figsize=(14,8))
    prompt_error_rates.plot(kind='bar')
    plt.title("Reasoning error rates per prompt type", fontsize=16)
    plt.ylabel("Proportion of examples with error")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prompt_type_error_rates.png"))
    plt.close()

# -------------------------
# 8️⃣ Error co-occurrence scatter (frequency vs MI)
# -------------------------
plt.figure(figsize=(12,8))
plt.scatter(error_freq, mi_series, s=100)
for i, txt in enumerate(error_cols):
    plt.annotate(txt, (error_freq[txt], mi_series[txt]),
                 xytext=(5,5), textcoords='offset points',
                 fontsize=12, ha='left', va='bottom')
plt.xlabel("Error Frequency", fontsize=14)
plt.ylabel("Mutual Information with correctness", fontsize=14)
plt.title("Frequency vs. MI of reasoning errors", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "frequency_vs_mi.png"))
plt.close()

# -------------------------
# 9️⃣ By ambiguity
# -------------------------
if 'ambiguous' in df.columns:
    amb_error_rates = df.groupby('ambiguous')[error_cols].mean()
    plt.figure(figsize=(12,8))
    amb_error_rates.plot(kind='bar')
    plt.title("Reasoning error rates by ambiguity", fontsize=16)
    plt.ylabel("Proportion of examples with error")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_rates_by_ambiguous.png"))
    plt.close()

# -------------------------
# 10️⃣ 4x4 Grid: Model × Prompt Type with is_correct on x-axis
# -------------------------
# Melt dataframe to long format
df_long = df.melt(
    id_vars=['model', 'prompt_type', 'is_correct'],
    value_vars=error_cols,
    var_name='error',
    value_name='presence'
)

# Palette for error types
palette = dict(zip(error_cols, sns.color_palette("Set3", n_colors=len(error_cols))))

# Determine order of columns: all combinations of model × prompt_type
models = sorted(df['model'].unique())
prompts = sorted(df['prompt_type'].unique())
col_order = [(m, p) for m in models for p in prompts]

# Rows = is_correct
is_correct_vals = [True, False]
n_rows = len(is_correct_vals)
n_cols = len(col_order)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), squeeze=False)
fig.suptitle("Error frequencies by Error Type (rows = is_correct, cols = model × prompt_type)", fontsize=18, y=0.95)

for i, correct_val in enumerate(is_correct_vals):
    for j, (model, prompt) in enumerate(col_order):
        ax = axes[i, j]
        subset = df_long[(df_long['model']==model) & (df_long['prompt_type']==prompt) & (df_long['is_correct']==correct_val)]
        if not subset.empty:
            means = subset.groupby('error')['presence'].mean()
            means.plot(kind='bar', ax=ax, color=[palette[e] for e in means.index])
        else:
            ax.set_visible(False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim(0,1)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f"{model} | {prompt}", fontsize=12)
    
    # Add facet label for the row (is_correct)
    fig.text(0.04, 1 - (i + 0.5)/n_rows, f"is_correct={correct_val}", fontsize=14, rotation=90, va='center')

# Create legend handles
handles = [plt.Rectangle((0,0),1,1,color=palette[e]) for e in error_cols]

# Place legend just below the main title
fig.legend(handles, error_cols, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=len(error_cols), frameon=False, title='')

plt.tight_layout(rect=[0.06,0,1,0.9])  # leave space for legend and row labels

# Save figure
output_dir = "reasoning_eval/analyse_labels/overall_plots"
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, "errors_by_model_prompt_rows_is_correct_facet_label_legend_top.png"))
plt.close()
# -------------------------
# 11️⃣ Normalized Mutual Information
# -------------------------
def binary_entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1-1e-12)
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

norm_mi = {}
for err in error_cols:
    p_x = mi_df[err].mean()
    p_y = mi_df['is_correct'].mean()
    H_X = binary_entropy(p_x)
    H_Y = binary_entropy(p_y)
    H_min = min(H_X, H_Y)
    norm_mi[err] = mi_scores[err] / H_min if H_min > 0 else 0

norm_mi_series = pd.Series(norm_mi).sort_values(ascending=False)

plt.figure(figsize=(12,8))
plt.scatter(error_freq, norm_mi_series, s=100, color='dodgerblue')
for err in error_cols:
    plt.text(error_freq[err], norm_mi_series[err], err, fontsize=12, ha='center', va='bottom')
plt.xlabel("Error Frequency", fontsize=14)
plt.ylabel("Normalized Mutual Information (0-1)", fontsize=14)
plt.title("Error Frequency vs Normalized MI", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "frequency_vs_normalized_mi.png"))
plt.close()

# -------------------------
# 12️⃣ Hierarchical clustering + dendrogram
# -------------------------
# Only cluster error_cols, not is_correct
corr_matrix_errors = df[error_cols].corr()

distance_matrix = 1 - corr_matrix_errors
condensed_distance = ssd.squareform(distance_matrix)
Z = linkage(condensed_distance, method='average')
max_distance = 0.5
cluster_labels = fcluster(Z, t=max_distance, criterion='distance')

cluster_df = pd.DataFrame({'error': error_cols, 'cluster': cluster_labels})
print("Error clusters:\n", cluster_df)

plt.figure(figsize=(14,10))
dendrogram(Z, labels=error_cols, leaf_rotation=45, leaf_font_size=12, color_threshold=max_distance)
plt.title("Hierarchical clustering of reasoning errors (co-occurrence)", fontsize=16)
plt.ylabel("Distance (1 - correlation)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "error_type_dendrogram_colored.png"))
plt.close()

# -------------------------
# 13️⃣ PCA of errors
# -------------------------
X = df[error_cols]
X_T = X.T
pca = PCA(n_components=2)
coords = pca.fit_transform(X_T)

plt.figure(figsize=(10,10))
plt.scatter(coords[:,0], coords[:,1], s=150, color='dodgerblue')
for i, err in enumerate(error_cols):
    plt.text(coords[i,0], coords[i,1], err, fontsize=12, ha='center', va='bottom')
plt.title("PCA of reasoning errors (co-occurrence)", fontsize=16)
plt.xlabel("PC1", fontsize=14)
plt.ylabel("PC2", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "error_type_pca.png"))
plt.show()


# -------------------------
# 14️⃣ Three-way outcome classification (addressing R#'s §4.2 point)
# -------------------------
def classify_outcome(row):
    if row['is_correct'] == 1:
        return 'correct'
    if row.get('incorrect_and_stereotype', 0) == 1:
        return 'biased'
    return 'incorrect_non_stereotype'

outcome_order = ['correct', 'incorrect_non_stereotype', 'biased']
df['outcome_class'] = pd.Categorical(
    df.apply(classify_outcome, axis=1),
    categories=outcome_order, ordered=True,
)

# Report cell sizes — reviewers want to see the denominators, especially
# for 'biased', which is ~0.72% of Qwen3 traces.
n_by_outcome = df['outcome_class'].value_counts().reindex(outcome_order)
print("Overall outcome distribution:\n", n_by_outcome)

n_by_model_outcome = (
    df.groupby(['model', 'prompt_type', 'outcome_class'], observed=True)
      .size().unstack(fill_value=0)
)
n_by_model_outcome.to_csv(os.path.join(output_dir, "outcome_class_counts.csv"))




###

# --- Rates by outcome class (this is the plot the reviewer asked for) ---
rates_by_outcome = df.groupby('outcome_class', observed=True)[error_cols].mean()
counts = df['outcome_class'].value_counts().reindex(outcome_order)
rates_by_outcome.to_csv(os.path.join(output_dir, "error_rates_by_outcome_class.csv"))

fig, ax = plt.subplots(figsize=(14, 7))
rates_by_outcome.T.plot(kind='bar', ax=ax, width=0.8)
ax.set_title(
    "BiasTrace label rates by outcome class\n"
    + " | ".join(f"n({k})={int(counts[k])}" for k in outcome_order),
    fontsize=13,
)
ax.set_ylabel("Proportion of traces with label")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.legend(title="Outcome class")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "error_rates_by_outcome_class.png"))
plt.close()

# Same plot faceted by model × prompt_type, so the reviewer can see
# whether patterns hold within each condition (not just pooled).
g = sns.catplot(
    data=df.melt(
        id_vars=['model', 'prompt_type', 'outcome_class'],
        value_vars=error_cols, var_name='error', value_name='present',
    ),
    x='error', y='present', hue='outcome_class', hue_order=outcome_order,
    col='prompt_type', row='model', kind='bar', errorbar=('ci', 95),
    height=4, aspect=1.6, sharey=True,
)
g.set_xticklabels(rotation=45, ha='right')
g.set_axis_labels("", "Proportion")
g.savefig(os.path.join(output_dir, "error_rates_by_outcome_class_facets.png"))
plt.close('all')


###

from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half

records = []
for err in error_cols:
    tab = pd.crosstab(df['outcome_class'], df[err])
    chi2_p = chi2_contingency(tab.values)[1] if tab.shape[1] == 2 else np.nan

    row = {'error': err, 'chi2_p_3way': chi2_p}
    for cls in outcome_order:
        sub = df[df['outcome_class'] == cls][err]
        k, n = int(sub.sum()), int(len(sub))
        lo, hi = wilson_ci(k, n)
        row.update({
            f'{cls}_n': n, f'{cls}_rate': k / n if n else np.nan,
            f'{cls}_ci_low': lo, f'{cls}_ci_high': hi,
        })

    # Pairwise: biased vs. each non-biased class (Fisher's exact for small cells)
    for other in ['correct', 'incorrect_non_stereotype']:
        a = df[df['outcome_class'] == 'biased'][err]
        b = df[df['outcome_class'] == other][err]
        if len(a) and len(b):
            _, p = fisher_exact([
                [int(a.sum()), int((1 - a).sum())],
                [int(b.sum()), int((1 - b).sum())],
            ])
        else:
            p = np.nan
        row[f'biased_vs_{other}_p'] = p
    records.append(row)

stat_df = pd.DataFrame(records)

# BH correction across error types, per family of tests
for col in ['chi2_p_3way', 'biased_vs_correct_p', 'biased_vs_incorrect_non_stereotype_p']:
    mask = stat_df[col].notna()
    if mask.any():
        _, q, _, _ = multipletests(stat_df.loc[mask, col].values, method='fdr_bh')
        stat_df.loc[mask, col.replace('_p', '_q')] = q

stat_df.to_csv(os.path.join(output_dir, "error_rates_by_outcome_class_stats.csv"), index=False)


# -------------------------
# 14b️⃣ Rates by outcome class × ambiguity
# -------------------------

# Normalize ambiguous to a clean boolean
df['ambiguous_bool'] = df['ambiguous'].astype(str).isin(['True', 'true', '1'])

# --- Long-format table: rate + Wilson CI for every cell ---
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half

rows = []
for amb in [False, True]:
    for cls in outcome_order:
        sub = df[(df['ambiguous_bool'] == amb) & (df['outcome_class'] == cls)]
        n = len(sub)
        for err in error_cols:
            k = int(sub[err].sum())
            lo, hi = wilson_ci(k, n)
            rows.append({
                'ambiguous': amb, 'outcome_class': cls, 'error': err,
                'n': n, 'k': k,
                'rate': k / n if n else np.nan,
                'ci_low': lo, 'ci_high': hi,
            })

rates_long = pd.DataFrame(rows)
rates_long.to_csv(os.path.join(output_dir, "error_rates_by_outcome_x_ambiguous.csv"), index=False)

# Print cell sizes so you can see immediately if biased_disambig is too small
print("Cell sizes (outcome × ambiguous):")
print(df.groupby(['ambiguous_bool', 'outcome_class'], observed=True).size().unstack(fill_value=0))

# --- Faceted plot: one panel per ambiguity condition ---
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
for ax, amb in zip(axes, [False, True]):
    sub = rates_long[rates_long['ambiguous'] == amb]
    pivot = sub.pivot(index='error', columns='outcome_class', values='rate')[outcome_order]
    err_low = sub.pivot(index='error', columns='outcome_class', values='ci_low')[outcome_order]
    err_high = sub.pivot(index='error', columns='outcome_class', values='ci_high')[outcome_order]
    yerr = np.array([pivot.values - err_low.values, err_high.values - pivot.values])
    # yerr shape must be (2, n_groups, n_bars) for grouped bar error bars
    yerr = np.transpose(yerr, (2, 0, 1))

    pivot.plot(kind='bar', ax=ax, width=0.8, yerr=yerr, capsize=2, error_kw={'elinewidth': 0.8})
    counts = df[df['ambiguous_bool'] == amb]['outcome_class'].value_counts().reindex(outcome_order)
    ax.set_title(
        f"{'Ambiguous' if amb else 'Disambiguated'}\n"
        + " | ".join(f"n({k})={int(counts[k])}" for k in outcome_order),
        fontsize=12,
    )
    ax.set_ylabel("Proportion of traces with label" if not amb else "")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title="Outcome class", loc='upper left', fontsize=9)

plt.suptitle("BiasTrace label rates by outcome class × context ambiguity", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "error_rates_by_outcome_x_ambiguous.png"), bbox_inches='tight')
plt.close()

# --- Stats within each ambiguity condition ---
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

stat_rows = []
for amb in [False, True]:
    sub_df = df[df['ambiguous_bool'] == amb]
    for err in error_cols:
        tab = pd.crosstab(sub_df['outcome_class'], sub_df[err])
        chi2_p = chi2_contingency(tab.values)[1] if tab.shape[1] == 2 else np.nan
        row = {'ambiguous': amb, 'error': err, 'chi2_p_3way': chi2_p}
        for other in ['correct', 'incorrect_non_stereotype']:
            a = sub_df[sub_df['outcome_class'] == 'biased'][err]
            b = sub_df[sub_df['outcome_class'] == other][err]
            if len(a) and len(b):
                _, p = fisher_exact([
                    [int(a.sum()), int((1 - a).sum())],
                    [int(b.sum()), int((1 - b).sum())],
                ])
            else:
                p = np.nan
            row[f'biased_vs_{other}_p'] = p
        stat_rows.append(row)

stat_amb_df = pd.DataFrame(stat_rows)
# BH correction within each ambiguity condition, per test family
for amb in [False, True]:
    mask_amb = stat_amb_df['ambiguous'] == amb
    for col in ['chi2_p_3way', 'biased_vs_correct_p', 'biased_vs_incorrect_non_stereotype_p']:
        mask = mask_amb & stat_amb_df[col].notna()
        if mask.any():
            _, q, _, _ = multipletests(stat_amb_df.loc[mask, col].values, method='fdr_bh')
            stat_amb_df.loc[mask, col.replace('_p', '_q')] = q

stat_amb_df.to_csv(os.path.join(output_dir, "error_rates_by_outcome_x_ambiguous_stats.csv"), index=False)


# -------------------------
# 14b️⃣ Rates by outcome class × ambiguity × prompt_type
# -------------------------

df['ambiguous_bool'] = df['ambiguous'].astype(str).isin(['True', 'true', '1'])

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half

prompt_types = sorted(df['prompt_type'].dropna().unique())

# --- Long-format table: rate + Wilson CI for every cell ---
rows = []
for amb in [False, True]:
    for pt in prompt_types:
        for cls in outcome_order:
            sub = df[
                (df['ambiguous_bool'] == amb)
                & (df['prompt_type'] == pt)
                & (df['outcome_class'] == cls)
            ]
            n = len(sub)
            for err in error_cols:
                k = int(sub[err].sum())
                lo, hi = wilson_ci(k, n)
                rows.append({
                    'ambiguous': amb, 'prompt_type': pt,
                    'outcome_class': cls, 'error': err,
                    'n': n, 'k': k,
                    'rate': k / n if n else np.nan,
                    'ci_low': lo, 'ci_high': hi,
                })

rates_long = pd.DataFrame(rows)
rates_long.to_csv(
    os.path.join(output_dir, "error_rates_by_outcome_x_ambiguous_x_prompt.csv"),
    index=False,
)

# Print cell sizes — biased × disambiguated × prompt_type is where things get tiny
print("Cell sizes (ambiguous × prompt_type × outcome):")
print(
    df.groupby(['ambiguous_bool', 'prompt_type', 'outcome_class'], observed=True)
      .size().unstack(fill_value=0)
)

# --- 2×2 grid: rows = ambiguity, cols = prompt_type ---
n_rows, n_cols = 2, len(prompt_types)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 6 * n_rows), sharey=True)
# Force axes to always be 2D for indexing
if n_cols == 1:
    axes = axes.reshape(-1, 1)

for i, amb in enumerate([True, False]):   # ambiguous on top row (usually the story)
    for j, pt in enumerate(prompt_types):
        ax = axes[i, j]
        sub = rates_long[
            (rates_long['ambiguous'] == amb)
            & (rates_long['prompt_type'] == pt)
        ]
        pivot = sub.pivot(index='error', columns='outcome_class', values='rate')[outcome_order]
        err_low = sub.pivot(index='error', columns='outcome_class', values='ci_low')[outcome_order]
        err_high = sub.pivot(index='error', columns='outcome_class', values='ci_high')[outcome_order]

        yerr = np.array([pivot.values - err_low.values, err_high.values - pivot.values])
        yerr = np.transpose(yerr, (2, 0, 1))  # (n_series, 2, n_errors)
        yerr = np.clip(yerr, 0, None) 

        pivot.plot(
            kind='bar', ax=ax, width=0.8, yerr=yerr,
            capsize=2, error_kw={'elinewidth': 0.8}, legend=(i == 0 and j == n_cols - 1),
        )
        counts = df[
            (df['ambiguous_bool'] == amb) & (df['prompt_type'] == pt)
        ]['outcome_class'].value_counts().reindex(outcome_order).fillna(0).astype(int)

        ax.set_title(
            f"{'Ambiguous' if amb else 'Disambiguated'} | {pt}\n"
            + " | ".join(f"n({k})={counts[k]}" for k in outcome_order),
            fontsize=11,
        )
        ax.set_ylabel("Proportion of traces with label" if j == 0 else "")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        if ax.get_legend() is not None:
            ax.legend(title="Outcome class", loc='upper left', fontsize=9)

plt.suptitle(
    "BiasTrace label rates by outcome class × context ambiguity × prompt type",
    fontsize=14, y=1.00,
)
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "error_rates_by_outcome_x_ambiguous_x_prompt.png"),
    bbox_inches='tight',
)
plt.close()

# --- Stats within each ambiguity × prompt_type cell ---
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

stat_rows = []
for amb in [False, True]:
    for pt in prompt_types:
        sub_df = df[(df['ambiguous_bool'] == amb) & (df['prompt_type'] == pt)]
        for err in error_cols:
            tab = pd.crosstab(sub_df['outcome_class'], sub_df[err])
            chi2_p = chi2_contingency(tab.values)[1] if tab.shape[1] == 2 else np.nan
            row = {
                'ambiguous': amb, 'prompt_type': pt,
                'error': err, 'chi2_p_3way': chi2_p,
            }
            for other in ['correct', 'incorrect_non_stereotype']:
                a = sub_df[sub_df['outcome_class'] == 'biased'][err]
                b = sub_df[sub_df['outcome_class'] == other][err]
                if len(a) and len(b):
                    _, p = fisher_exact([
                        [int(a.sum()), int((1 - a).sum())],
                        [int(b.sum()), int((1 - b).sum())],
                    ])
                else:
                    p = np.nan
                row[f'biased_vs_{other}_p'] = p
            stat_rows.append(row)

stat_df = pd.DataFrame(stat_rows)

# BH correction within each (ambiguous × prompt_type) cell, per test family
for amb in [False, True]:
    for pt in prompt_types:
        mask_cell = (stat_df['ambiguous'] == amb) & (stat_df['prompt_type'] == pt)
        for col in ['chi2_p_3way', 'biased_vs_correct_p', 'biased_vs_incorrect_non_stereotype_p']:
            mask = mask_cell & stat_df[col].notna()
            if mask.any():
                _, q, _, _ = multipletests(stat_df.loc[mask, col].values, method='fdr_bh')
                stat_df.loc[mask, col.replace('_p', '_q')] = q

stat_df.to_csv(
    os.path.join(output_dir, "error_rates_by_outcome_x_ambiguous_x_prompt_stats.csv"),
    index=False,
)