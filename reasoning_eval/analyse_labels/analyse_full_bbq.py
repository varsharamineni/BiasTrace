"""
End-to-end analysis of LLM-as-a-judge annotations for BBQ reasoning traces.

Outputs:
- flattened annotations CSV
- label prevalence tables
- correctness + stereotype comparisons
- category-wise summaries
- publication-ready figures (PDF)
- logistic regression predicting stereotypical errors
- full run metadata with input file hashes
"""

# ======================
# Imports
# ======================
import json
import glob
import hashlib
import datetime
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive, safe for SSH
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

# ======================
# Config
# ======================
INPUT_GLOBS = (
    "outputs/qwen_full_8B_simple_prompt/**/full_annotation/*/llm_eval_bbq_*.json",
    "outputs/qwen_full_8B_full_prompt/full_annotation/*/llm_eval_bbq_*.json",
)

JUDGE_LABELS = [
    "group_assumption",
    "bias_acknowledgement",
    "meta_reflection",
    "outside_demo_knowledge",
    "outside_topical_knowledge",
    #"unresolved",
    "overthinking",
    #"missing_logic",
]

N_BOOTSTRAP = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUT_DIR = Path(f"reasoning_eval/analyse_labels/bbq_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# Helpers
# ======================
def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_ci(series, n=N_BOOTSTRAP, alpha=0.05):
    vals = series.values
    means = []
    for _ in range(n):
        sample = np.random.choice(vals, size=len(vals), replace=True)
        means.append(sample.mean())
    return (
        np.percentile(means, 100 * alpha / 2),
        np.percentile(means, 100 * (1 - alpha / 2)),
    )


# ======================
# Load + flatten judge annotations
# ======================
def load_judge_files(paths):
    rows = []
    file_hashes = {}

    if len(paths) == 0:
        raise RuntimeError("No files found for input glob patterns")

    for path in paths:
        file_hashes[path] = hash_file(path)

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
                "category": r["category"],
                "example_id": r["example_id"],
                "model": r["model"],
                "prompt_type": prompt_type,
                "judge_model": r["judge_model"],
                "judge_prompt": r["judge_prompt"],
                "is_correct": r["is_correct"],
                "ambiguous": r["ambiguous"],
                "stereotype_alignment": r["stereotype_alignment"],
                "incorrect_and_stereotype": r["incorrect_and_stereotype"],
                "bbq_category": meta.get("bbq_category"),
                "reasoning_prompt_used": meta.get("reasoning_prompt_used"),
            }

            judge_out = r.get("judge_output")
            if judge_out is None:
                for k in JUDGE_LABELS:
                    row[k] = np.nan
                row["judge_missing"] = True
            else:
                for k in JUDGE_LABELS:
                    row[k] = judge_out.get(k, 0)
                row["judge_missing"] = False

            rows.append(row)

    return pd.DataFrame(rows), file_hashes


# ======================
# Collect all files
# ======================
all_paths = []
for pattern in INPUT_GLOBS:
    all_paths.extend(glob.glob(pattern, recursive=True))

if len(all_paths) == 0:
    raise RuntimeError(f"No files found for input patterns: {INPUT_GLOBS}")

df, file_hashes = load_judge_files(all_paths)

print(f"Loaded {len(df)} annotated samples")
print("Total samples:", len(df))
print("Missing judge_output:", df["judge_missing"].sum())
print("Fraction missing:", df["judge_missing"].mean())


# ======================
# 1. Overall label prevalence + CI
# ======================
means = df[JUDGE_LABELS].mean().sort_values(ascending=False)

ci_rows = []
for label in JUDGE_LABELS:
    lo, hi = bootstrap_ci(df[label])
    ci_rows.append({"label": label, "ci_low": lo, "ci_high": hi})

ci_df = pd.DataFrame(ci_rows).set_index("label")

overall_df = pd.concat(
    [means.rename("mean"), ci_df],
    axis=1
)

overall_df.to_csv(OUT_DIR / "table_overall_label_prevalence.csv")

plt.figure()
plt.bar(overall_df.index, overall_df["mean"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Proportion")
plt.title("Judge Label Distribution (All Samples)")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_overall_distribution.pdf")
plt.close()


# ======================
# 2. Correct vs incorrect
# ======================
by_correct = df.groupby("is_correct")[JUDGE_LABELS].mean().T
by_correct.to_csv(OUT_DIR / "table_labels_by_correctness.csv")

by_correct.plot(kind="bar")
plt.ylabel("Proportion")
plt.title("Judge Labels by Answer Correctness")
plt.xticks(rotation=45, ha="right")
plt.legend(title="is_correct")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_correct_vs_incorrect.pdf")
plt.close()


# ======================
# 3. Stereotype failures
# ======================
incorrect = df[~df["is_correct"]]
stereo = df[df["incorrect_and_stereotype"]]

compare = pd.DataFrame({
    "incorrect_all": incorrect[JUDGE_LABELS].mean(),
    "incorrect_and_stereotype": stereo[JUDGE_LABELS].mean(),
})

compare.to_csv(OUT_DIR / "table_stereotype_comparison.csv")

compare.plot(kind="bar")
plt.ylabel("Proportion")
plt.title("Reasoning Failures in Stereotypical Errors")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_stereotype_failures.pdf")
plt.close()


# ======================
# 4. Category-wise heatmap
# ======================
cat_means = df.groupby("category")[JUDGE_LABELS].mean()
cat_means.to_csv(OUT_DIR / "table_labels_by_category.csv")

plt.figure(figsize=(10, 4))
plt.imshow(cat_means.values, aspect="auto")
plt.yticks(range(len(cat_means)), cat_means.index)
plt.xticks(range(len(JUDGE_LABELS)), JUDGE_LABELS, rotation=45, ha="right")
plt.colorbar(label="Proportion")
plt.title("Judge Label Prevalence by BBQ Category")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_category_heatmap.pdf")
plt.close()

# ======================
# 5. Logistic regression with interaction
# ======================
REG_COLS = ["incorrect_and_stereotype", "is_correct", "ambiguous", "model", "category", "prompt_type"] + JUDGE_LABELS

reg_df = df.loc[~df["judge_missing"], REG_COLS].dropna().copy()



print(reg_df["prompt_type"].unique())
print(reg_df["prompt_type"].dtype)

# Type coercion
for col in JUDGE_LABELS:
    reg_df[col] = reg_df[col].astype(int)

reg_df["incorrect"] = (~reg_df["is_correct"].astype(bool)).astype(int)
reg_df["ambiguous"] = reg_df["ambiguous"].astype("category")
reg_df["prompt_type"] = reg_df["prompt_type"].astype("category")
reg_df["model"] = reg_df["model"].astype("category")
reg_df["category"] = reg_df["category"].astype("category")


interaction_terms = " + ".join([f"{label}:C(prompt_type, Treatment(reference=\"simple_prompt\"))" for label in JUDGE_LABELS])

formula = (
    "incorrect ~ "
    + " + ".join(JUDGE_LABELS)
    + " + C(prompt_type, Treatment(reference=\"simple_prompt\"))"
    + " + " + interaction_terms
    + " + C(model)"
    + " + C(category)"
)


logit_model = smf.logit(formula=formula, data=reg_df).fit(
    disp=False,
    cov_type="HC3"
)

with open(OUT_DIR / "logit_incorrect.txt", "w") as f:
    f.write(logit_model.summary().as_text())

with open(OUT_DIR / "logit_sample_size.txt", "w") as f:
    f.write(f"N regression samples: {len(reg_df)}\n")



# ----------------------
# Logistic regression for stereotypical errors
# ----------------------
# Ensure target is int (0/1)
reg_df["incorrect_and_stereotype"] = reg_df["incorrect_and_stereotype"].astype(int)

formula_stereo = (
    "incorrect_and_stereotype ~ "
    + " + ".join(JUDGE_LABELS)
    + " + C(prompt_type, Treatment(reference='simple_prompt'))"
    + " + " + interaction_terms
    + " + C(model)"
    + " + C(category)"
)

logit_stereo_model = smf.logit(formula=formula_stereo, data=reg_df).fit(
    disp=False,
    cov_type="HC3"
)

# Save summary
with open(OUT_DIR / "logit_incorrect_and_stereo.txt", "w") as f:
    f.write(logit_stereo_model.summary().as_text())

print("Logistic regression for stereotypical errors complete.")

# ======================
# Compare is_correct vs incorrect_and_stereotype
# ======================

# Extract coefficients from both models
coefs_correct = pd.DataFrame(logit_model.params, columns=["coef_incorrect"])
coefs_stereo = pd.DataFrame(logit_stereo_model.params, columns=["coef_incorrect_and_stereo"])

# Merge into one DataFrame
coef_compare = coefs_correct.join(coefs_stereo, how="outer")
coef_compare["coef_incorrect_signed"] = coef_compare["coef_incorrect"].fillna(0)
coef_compare["coef_incorrect_and_stereo_signed"] = coef_compare["coef_incorrect_and_stereo"].fillna(0)

# Optional: calculate magnitude difference
coef_compare["diff"] = coef_compare["coef_incorrect_signed"] - coef_compare["coef_incorrect_and_stereo_signed"]
# Save for record
coef_compare.to_csv(OUT_DIR / "logit_coefficients_comparison.csv")

# ======================
# Plot side-by-side
# ======================
plt.figure(figsize=(12, 12))

y = np.arange(len(coef_compare))
height = 0.4

plt.barh(
    y - height/2,
    coef_compare["coef_incorrect_signed"],
    height,
    label="is_incorrect",
    color="steelblue"
)

plt.barh(
    y + height/2,
    coef_compare["coef_incorrect_and_stereo_signed"],
    height,
    label="incorrect_and_stereotype",
    color="indianred"
)

plt.yticks(y, coef_compare.index)
plt.xlabel("Logit Coefficient")
plt.title("Comparison of Logistic Regression Coefficients")
plt.axvline(0, color="black", linewidth=0.8)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_coefficients_comparison.pdf")
plt.close()


# ======================
# 7. Distribution of errors by prompt_type and ambiguous
# ======================
plt.figure(figsize=(8, 5))

# Compute mean correctness by prompt_type and ambiguity
error_dist = (
    reg_df
    .groupby(["prompt_type", "ambiguous"])["is_correct"]
    .mean()
    .unstack()
)

print("Mean correctness by prompt_type and ambiguous:")
print(error_dist)

# Bar plot
error_dist.plot(kind="bar")
plt.ylabel("Mean Correctness")
plt.title("Mean Correctness by Prompt Type and Ambiguous Flag")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Ambiguous")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_correctness_by_prompt_and_ambiguous.pdf")
plt.close()

# Optional: boxplot of individual samples
plt.figure(figsize=(8, 5))
import seaborn as sns
sns.boxplot(
    data=reg_df,
    x="prompt_type",
    y="is_correct",
    hue="ambiguous"
)
plt.ylabel("Correctness (0/1)")
plt.title("Distribution of Correctness by Prompt Type and Ambiguous Flag")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Ambiguous")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_correctness_boxplot_by_prompt_and_ambiguous.pdf")
plt.close()


# ======================
# 6. Run metadata
# ======================
run_metadata = {
    "timestamp": datetime.datetime.now().isoformat(),
    "input_globs": INPUT_GLOBS,
    "n_samples": int(len(df)),
    "judge_labels": JUDGE_LABELS,
    "bootstrap_samples": N_BOOTSTRAP,
    "random_seed": RANDOM_SEED,
    "input_file_hashes": file_hashes,
}

try:
    run_metadata["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()
except Exception:
    run_metadata["git_commit"] = None

with open(OUT_DIR / "run_metadata.json", "w") as f:
    json.dump(run_metadata, f, indent=2)

print(f"Analysis complete. Results saved to:\n{OUT_DIR}")
