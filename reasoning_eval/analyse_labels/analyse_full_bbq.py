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
            row = {
                "source_file": path,
                "sample_id": r["sample_id"],
                "category": r["category"],
                "example_id": r["example_id"],
                "model": r["model"],
                "prompt_type": r["prompt_type"],
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
REG_COLS = ["is_correct", "ambiguous", "model", "category", "prompt_type"] + JUDGE_LABELS

reg_df = df.loc[~df["judge_missing"], REG_COLS].dropna().copy()



print(reg_df["prompt_type"].unique())
print(reg_df["prompt_type"].dtype)

# Type coercion
for col in JUDGE_LABELS:
    reg_df[col] = reg_df[col].astype(int)

reg_df["is_correct"] = reg_df["is_correct"].astype(int)
reg_df["ambiguous"] = reg_df["ambiguous"].astype("category")
reg_df["prompt_type"] = reg_df["prompt_type"].astype("category")
reg_df["model"] = reg_df["model"].astype("category")
reg_df["category"] = reg_df["category"].astype("category")


interaction_terms = " + ".join([f"{label}:C(prompt_type)" for label in JUDGE_LABELS])

formula = (
    "is_correct ~ "
    + " + ".join(JUDGE_LABELS)
    + " + C(prompt_type)"
    + " + " + interaction_terms
    + " + C(model)"
    + " + C(category)"
)


import patsy
y, X = patsy.dmatrices(formula, data=reg_df, return_type="dataframe")
print(X.columns)


logit_model = smf.logit(formula=formula, data=reg_df).fit(
    disp=False,
    cov_type="HC3"
)

with open(OUT_DIR / "logit_stereotype_prediction.txt", "w") as f:
    f.write(logit_model.summary().as_text())

with open(OUT_DIR / "logit_sample_size.txt", "w") as f:
    f.write(f"N regression samples: {len(reg_df)}\n")


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
