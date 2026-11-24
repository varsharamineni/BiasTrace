# reasoning_error_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
import scipy.spatial.distance as ssd
import numpy as np
import os
import glob
import json
from pathlib import Path


# -------------------------
# Merged results files (model answers + BBQ answer_info)
# -------------------------
# Annotation files don't say WHICH option the model picked, so we join them
# to the merged results file named in metadata["input_file"].
_merged_cache = {}

def _resolve_input_file(annot_path, meta):
    """Find the merged results file for an annotation file."""
    candidates = []
    if meta.get("input_file"):
        candidates.append(Path(meta["input_file"]))
    stem = meta.get("input_stem")
    if stem:
        for parent in Path(annot_path).parents:           # walk up from the annotation file
            candidates.append(parent / f"{stem}.json")
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Merged results file for {annot_path} not found. Tried: {[str(c) for c in candidates]}"
    )


def _load_merged(path):
    """example_id -> merged record, cached per file."""
    key = str(path)
    if key not in _merged_cache:
        with open(path, "r") as f:
            recs = json.load(f)["results"]
        _merged_cache[key] = {r["example_id"]: r for r in recs}
    return _merged_cache[key]


def predicted_option_index(rec):
    """Index (0/1/2) of the option the model chose, or None if unparsed."""
    ea = str(rec.get("extracted_answer") or "").strip().upper()
    if ea in ("A", "B", "C"):
        return "ABC".index(ea)
    ans = rec.get("normalized_answer") or rec.get("model_answer")
    if ans is not None:
        for i in range(3):
            if str(ans).strip().lower() == str(rec.get(f"ans{i}", "")).strip().lower():
                return i
    return None


def predicted_answer_tag(rec):
    """BBQ group tag of the chosen option ('unknown', 'lowSES', ...), or None."""
    idx = predicted_option_index(rec)
    if idx is None:
        return None
    info = (rec.get("answer_info") or {}).get(f"ans{idx}")
    return str(info[-1]) if info else None


# -------------------------
# Load multiple JSON annotation files
# -------------------------
def load_judge_files(paths, judge_labels):
    rows = []
    n_unmatched = 0
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)

        merged = _load_merged(_resolve_input_file(path, data.get("metadata", {})))

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

            m = merged.get(r.get("example_id"))
            if m is None:
                n_unmatched += 1
                row["answer_tag"] = None
                row["join_correct_mismatch"] = False
            else:
                row["answer_tag"] = predicted_answer_tag(m)
                # Both files record is_correct; they should agree for a correct join
                row["join_correct_mismatch"] = bool(m.get("is_correct")) != bool(r.get("is_correct"))

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

    out = pd.DataFrame(rows)
    n_mismatch = int(out["join_correct_mismatch"].sum())
    print(f"Join to merged results: {len(out) - n_unmatched}/{len(out)} rows matched; "
          f"{n_mismatch} with conflicting is_correct; "
          f"{int(out['answer_tag'].isna().sum())} with no parsed answer option")
    if n_unmatched or n_mismatch:
        print("  !! Check the join: example_id may not be unique within a merged file.")
    return out


def normalize_model_name(model_str):
    if pd.isna(model_str):
        return model_str
    model_str = model_str.lower()
    if "qwen3-14b" in model_str:
        return "Qwen3-14B"
    return model_str


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


# -------------------------
# 1️⃣ Load data
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
    'group_assumption',
    'bias_acknowledgement',
    'meta_reflection',
    'outside_demo_knowledge',
    'outside_topical_knowledge',
    'overthinking',
]

df = load_judge_files(all_paths, judge_labels=error_cols)
print(f"Loaded {len(df)} annotated samples from {len(all_paths)} files")

df[error_cols] = df[error_cols].fillna(0).astype(int)
df["is_correct"] = df["is_correct"].fillna(0).astype(int)
if "ambiguous" in df.columns:
    df["ambiguous"] = df["ambiguous"].astype("category")
df['model'] = df['model'].apply(normalize_model_name)

output_dir = "reasoning_eval/analyse_labels/overall_plots_outcome"
os.makedirs(output_dir, exist_ok=True)


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
plt.ylabel("MI (nats)")      # sklearn's mutual_info_score uses natural log
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
# 8️⃣ Frequency vs MI
# -------------------------
plt.figure(figsize=(12,8))
plt.scatter(error_freq, mi_series[error_freq.index], s=100)
for txt in error_cols:
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
    amb_error_rates = df.groupby('ambiguous', observed=True)[error_cols].mean()
    plt.figure(figsize=(12,8))
    amb_error_rates.plot(kind='bar')
    plt.title("Reasoning error rates by ambiguity", fontsize=16)
    plt.ylabel("Proportion of examples with error")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_rates_by_ambiguous.png"))
    plt.close()

# -------------------------
# 10️⃣ Grid: Model × Prompt Type, rows = is_correct
# -------------------------
df_long = df.melt(
    id_vars=['model', 'prompt_type', 'is_correct'],
    value_vars=error_cols, var_name='error', value_name='presence',
)
palette = dict(zip(error_cols, sns.color_palette("Set3", n_colors=len(error_cols))))
models = sorted(df['model'].unique())
prompts = sorted(df['prompt_type'].unique())
col_order = [(m, p) for m in models for p in prompts]
is_correct_vals = [True, False]
n_rows, n_cols = len(is_correct_vals), len(col_order)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), squeeze=False)
fig.suptitle("Error frequencies by Error Type (rows = is_correct, cols = model × prompt_type)",
             fontsize=18, y=0.95)
for i, correct_val in enumerate(is_correct_vals):
    for j, (model, prompt) in enumerate(col_order):
        ax = axes[i, j]
        subset = df_long[(df_long['model']==model) & (df_long['prompt_type']==prompt)
                         & (df_long['is_correct']==correct_val)]
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
    fig.text(0.04, 1 - (i + 0.5)/n_rows, f"is_correct={correct_val}", fontsize=14,
             rotation=90, va='center')

handles = [plt.Rectangle((0,0),1,1,color=palette[e]) for e in error_cols]
fig.legend(handles, error_cols, loc='upper center', bbox_to_anchor=(0.5, 0.92),
           ncol=len(error_cols), frameon=False, title='')
plt.tight_layout(rect=[0.06,0,1,0.9])
fig.savefig(os.path.join(output_dir, "errors_by_model_prompt_rows_is_correct_facet_label_legend_top.png"))
plt.close()

# -------------------------
# 11️⃣ Normalized Mutual Information
# -------------------------
def binary_entropy(p):
    """Entropy in nats, to match sklearn's mutual_info_score."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1-1e-12)
    return -p*np.log(p) - (1-p)*np.log(1-p)

norm_mi = {}
for err in error_cols:
    H_X = binary_entropy(mi_df[err].mean())
    H_Y = binary_entropy(mi_df['is_correct'].mean())
    H_min = min(H_X, H_Y)
    norm_mi[err] = mi_scores[err] / H_min if H_min > 0 else 0
norm_mi_series = pd.Series(norm_mi).sort_values(ascending=False)

plt.figure(figsize=(12,8))
plt.scatter(error_freq, norm_mi_series[error_freq.index], s=100, color='dodgerblue')
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
corr_matrix_errors = df[error_cols].corr()
distance_matrix = 1 - corr_matrix_errors
condensed_distance = ssd.squareform(distance_matrix, checks=False)
Z = linkage(condensed_distance, method='average')
max_distance = 0.5
cluster_labels = fcluster(Z, t=max_distance, criterion='distance')
print("Error clusters:\n", pd.DataFrame({'error': error_cols, 'cluster': cluster_labels}))

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
coords = PCA(n_components=2).fit_transform(df[error_cols].T)
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
plt.close()


# -------------------------
# 14️⃣ Four-way outcome classification
#     correct | incorrect: unknown | incorrect: non-stereotype | biased
# -------------------------
# An "unknown" answer = the model chose the option BBQ tags as "unknown"
# in answer_info (e.g. "Can't answer", "Not enough info"). The tag comes
# from the merged results file joined in load_judge_files.
df['pred_is_unknown'] = df['answer_tag'].astype(str).str.lower().eq('unknown')

print("\nChosen-option tag 'unknown' by is_correct x ambiguous:")
print(pd.crosstab(df['pred_is_unknown'], [df['is_correct'], df['ambiguous'].astype(str)]))

def is_unknown_answer(row):
    return bool(row['pred_is_unknown'])

def classify_outcome(row):
    if row['is_correct'] == 1:
        return 'correct'
    if row.get('incorrect_and_stereotype', 0) == 1:
        return 'biased'
    if is_unknown_answer(row):
        return 'incorrect_unknown'
    return 'incorrect_non_stereotype'

outcome_order  = ['correct', 'incorrect_unknown', 'incorrect_non_stereotype', 'biased']
OTHER_CLASSES  = [c for c in outcome_order if c != 'biased']
PAIRWISE_COLS  = [f'biased_vs_{c}_p' for c in OTHER_CLASSES]
TEST_COLS      = ['chi2_p_3way'] + PAIRWISE_COLS   # 'chi2_p_3way' = test across all classes

df['outcome_class'] = pd.Categorical(
    df.apply(classify_outcome, axis=1), categories=outcome_order, ordered=True,
)
df['ambiguous_bool'] = df['ambiguous'].astype(str).isin(['True', 'true', '1'])

n_by_outcome = df['outcome_class'].value_counts().reindex(outcome_order)
print("\nOverall outcome distribution:\n", n_by_outcome)

# Sanity checks: in BBQ, "unknown" is the correct answer in ambiguous
# contexts, so incorrect_unknown should only appear in disambiguated ones.
if n_by_outcome['incorrect_unknown'] == 0:
    print("  !! No 'incorrect_unknown' rows found. Check that answer_info tags "
          "the unknown option as 'unknown' in the merged results files.")
n_unk_amb = int(((df['outcome_class'] == 'incorrect_unknown') & df['ambiguous_bool']).sum())
if n_unk_amb > 0:
    print(f"  !! {n_unk_amb} 'incorrect_unknown' rows in AMBIGUOUS contexts. "
          f"In BBQ these should be correct; check the join and the answer parsing.")

n_by_model_outcome = (
    df.groupby(['model', 'prompt_type', 'outcome_class'], observed=False)
      .size().unstack(fill_value=0)
)
n_by_model_outcome.to_csv(os.path.join(output_dir, "outcome_class_counts.csv"))


def chi2_p_across_classes(sub_df, err):
    """Chi-square across outcome classes, dropping classes with no rows."""
    tab = pd.crosstab(sub_df['outcome_class'], sub_df[err])
    tab = tab.loc[tab.sum(axis=1) > 0]
    if tab.shape[0] < 2 or tab.shape[1] != 2:
        return np.nan
    return chi2_contingency(tab.values)[1]


def pairwise_fisher(sub_df, err, row):
    a = sub_df[sub_df['outcome_class'] == 'biased'][err]
    for other in OTHER_CLASSES:
        b = sub_df[sub_df['outcome_class'] == other][err]
        if len(a) and len(b):
            _, p = fisher_exact([
                [int(a.sum()), int((1 - a).sum())],
                [int(b.sum()), int((1 - b).sum())],
            ])
        else:
            p = np.nan
        row[f'biased_vs_{other}_p'] = p


def bh_correct(stat_df, group_masks):
    for mask_group in group_masks:
        for col in TEST_COLS:
            mask = mask_group & stat_df[col].notna()
            if mask.any():
                _, q, _, _ = multipletests(stat_df.loc[mask, col].values, method='fdr_bh')
                stat_df.loc[mask, col.replace('_p', '_q')] = q


def grouped_bar_with_ci(ax, sub_long, counts, legend):
    """Grouped bars (x = behaviour, hue = outcome class), skipping empty classes."""
    present = [c for c in outcome_order if counts.get(c, 0) > 0]
    pivot = sub_long.pivot(index='error', columns='outcome_class', values='rate')[present]
    lo    = sub_long.pivot(index='error', columns='outcome_class', values='ci_low')[present]
    hi    = sub_long.pivot(index='error', columns='outcome_class', values='ci_high')[present]
    yerr = np.array([pivot.values - lo.values, hi.values - pivot.values])
    yerr = np.clip(np.transpose(yerr, (2, 0, 1)), 0, None)   # (n_series, 2, n_errors)
    pivot.plot(kind='bar', ax=ax, width=0.8, yerr=yerr, capsize=2,
               error_kw={'elinewidth': 0.8}, legend=legend,
               color=[CLASS_COLOURS[c] for c in present])


CLASS_COLOURS = dict(zip(outcome_order, sns.color_palette("tab10", len(outcome_order))))


# --- Rates by outcome class (pooled) ---
rates_by_outcome = df.groupby('outcome_class', observed=True)[error_cols].mean()
counts = df['outcome_class'].value_counts().reindex(outcome_order).fillna(0).astype(int)
rates_by_outcome.to_csv(os.path.join(output_dir, "error_rates_by_outcome_class.csv"))

fig, ax = plt.subplots(figsize=(14, 7))
rates_by_outcome.T.plot(kind='bar', ax=ax, width=0.8,
                        color=[CLASS_COLOURS[c] for c in rates_by_outcome.index])
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

# Faceted by model × prompt_type
g = sns.catplot(
    data=df.melt(
        id_vars=['model', 'prompt_type', 'outcome_class'],
        value_vars=error_cols, var_name='error', value_name='present',
    ),
    x='error', y='present', hue='outcome_class', hue_order=outcome_order,
    palette=CLASS_COLOURS,
    col='prompt_type', row='model', kind='bar', errorbar=('ci', 95),
    height=4, aspect=1.6, sharey=True,
)
g.set_xticklabels(rotation=45, ha='right')
g.set_axis_labels("", "Proportion")
g.savefig(os.path.join(output_dir, "error_rates_by_outcome_class_facets.png"))
plt.close('all')

# --- Stats, pooled ---
records = []
for err in error_cols:
    row = {'error': err, 'chi2_p_3way': chi2_p_across_classes(df, err)}
    for cls in outcome_order:
        sub = df[df['outcome_class'] == cls][err]
        k, n = int(sub.sum()), int(len(sub))
        lo, hi = wilson_ci(k, n)
        row.update({
            f'{cls}_n': n, f'{cls}_rate': k / n if n else np.nan,
            f'{cls}_ci_low': lo, f'{cls}_ci_high': hi,
        })
    pairwise_fisher(df, err, row)
    records.append(row)

stat_df = pd.DataFrame(records)
bh_correct(stat_df, [pd.Series(True, index=stat_df.index)])
stat_df.to_csv(os.path.join(output_dir, "error_rates_by_outcome_class_stats.csv"), index=False)


# -------------------------
# 14b️⃣ Rates by outcome class × ambiguity
# -------------------------
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
                'n': n, 'k': k, 'rate': k / n if n else np.nan,
                'ci_low': lo, 'ci_high': hi,
            })
rates_long = pd.DataFrame(rows)
rates_long.to_csv(os.path.join(output_dir, "error_rates_by_outcome_x_ambiguous.csv"), index=False)

print("\nCell sizes (outcome × ambiguous):")
print(df.groupby(['ambiguous_bool', 'outcome_class'], observed=False).size().unstack(fill_value=0))

fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
for ax, amb in zip(axes, [False, True]):
    counts = (df[df['ambiguous_bool'] == amb]['outcome_class']
              .value_counts().reindex(outcome_order).fillna(0).astype(int))
    grouped_bar_with_ci(ax, rates_long[rates_long['ambiguous'] == amb], counts, legend=True)
    ax.set_title(
        f"{'Ambiguous' if amb else 'Disambiguated'}\n"
        + " | ".join(f"n({k})={counts[k]}" for k in outcome_order),
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

stat_rows = []
for amb in [False, True]:
    sub_df = df[df['ambiguous_bool'] == amb]
    for err in error_cols:
        row = {'ambiguous': amb, 'error': err, 'chi2_p_3way': chi2_p_across_classes(sub_df, err)}
        pairwise_fisher(sub_df, err, row)
        stat_rows.append(row)
stat_amb_df = pd.DataFrame(stat_rows)
bh_correct(stat_amb_df, [stat_amb_df['ambiguous'] == amb for amb in [False, True]])
stat_amb_df.to_csv(os.path.join(output_dir, "error_rates_by_outcome_x_ambiguous_stats.csv"), index=False)


# -------------------------
# 14c️⃣ Rates by outcome class × ambiguity × prompt_type
# -------------------------
prompt_types = sorted(df['prompt_type'].dropna().unique())

rows = []
for amb in [False, True]:
    for pt in prompt_types:
        for cls in outcome_order:
            sub = df[(df['ambiguous_bool'] == amb) & (df['prompt_type'] == pt)
                     & (df['outcome_class'] == cls)]
            n = len(sub)
            for err in error_cols:
                k = int(sub[err].sum())
                lo, hi = wilson_ci(k, n)
                rows.append({
                    'ambiguous': amb, 'prompt_type': pt,
                    'outcome_class': cls, 'error': err,
                    'n': n, 'k': k, 'rate': k / n if n else np.nan,
                    'ci_low': lo, 'ci_high': hi,
                })
rates_long = pd.DataFrame(rows)
rates_long.to_csv(os.path.join(output_dir, "error_rates_by_outcome_x_ambiguous_x_prompt.csv"), index=False)

print("\nCell sizes (ambiguous × prompt_type × outcome):")
print(df.groupby(['ambiguous_bool', 'prompt_type', 'outcome_class'], observed=False)
        .size().unstack(fill_value=0))

n_rows, n_cols = 2, len(prompt_types)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 6 * n_rows), sharey=True, squeeze=False)
for i, amb in enumerate([True, False]):
    for j, pt in enumerate(prompt_types):
        ax = axes[i, j]
        counts = (df[(df['ambiguous_bool'] == amb) & (df['prompt_type'] == pt)]['outcome_class']
                  .value_counts().reindex(outcome_order).fillna(0).astype(int))
        sub = rates_long[(rates_long['ambiguous'] == amb) & (rates_long['prompt_type'] == pt)]
        grouped_bar_with_ci(ax, sub, counts, legend=True)
        ax.set_title(
            f"{'Ambiguous' if amb else 'Disambiguated'} | {pt}\n"
            + " | ".join(f"n({k})={counts[k]}" for k in outcome_order),
            fontsize=11,
        )
        ax.set_ylabel("Proportion of traces with label" if j == 0 else "")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title="Outcome class", loc='upper left', fontsize=9)

plt.suptitle("BiasTrace label rates by outcome class × context ambiguity × prompt type",
             fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "error_rates_by_outcome_x_ambiguous_x_prompt.png"), bbox_inches='tight')
plt.close()

stat_rows = []
for amb in [False, True]:
    for pt in prompt_types:
        sub_df = df[(df['ambiguous_bool'] == amb) & (df['prompt_type'] == pt)]
        for err in error_cols:
            row = {'ambiguous': amb, 'prompt_type': pt, 'error': err,
                   'chi2_p_3way': chi2_p_across_classes(sub_df, err)}
            pairwise_fisher(sub_df, err, row)
            stat_rows.append(row)
stat_df = pd.DataFrame(stat_rows)
bh_correct(stat_df, [(stat_df['ambiguous'] == amb) & (stat_df['prompt_type'] == pt)
                     for amb in [False, True] for pt in prompt_types])
stat_df.to_csv(os.path.join(output_dir, "error_rates_by_outcome_x_ambiguous_x_prompt_stats.csv"), index=False)