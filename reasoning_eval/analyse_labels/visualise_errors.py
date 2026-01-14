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
INPUT_GLOBS = [
    "outputs/qwen_full_8B_simple_prompt/**/full_annotation/*/llm_eval_bbq_*.json",
    "outputs/qwen_full_8B_full_prompt/full_annotation/*/llm_eval_bbq_*.json",
]

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
    'unresolved',
    'overthinking',
    'missing_logic'
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
error_cols = [
    'group_assumption',
    'bias_acknowledgement',
    'meta_reflection',
    'outside_demo_knowledge',
    'outside_topical_knowledge',
    'unresolved',
    'overthinking',
    'missing_logic'
]

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
