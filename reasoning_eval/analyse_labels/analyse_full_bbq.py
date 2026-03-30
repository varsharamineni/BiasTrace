# ======================
# Full BBQ Analysis Script with Net & Isolated Effect Plots
# ======================

import json
import glob
import hashlib
import datetime
import subprocess
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

# ======================
# Config
# ======================
INPUT_GLOBS = (
    "outputs/qwen_full_8B_simple_prompt/**/full_annotation/*/llm_eval_bbq_*.json",
    "outputs/qwen_full_8B_full_prompt/full_annotation/*/llm_eval_bbq_*.json",
    "outputs/qwen_full_14B_simple_prompt/**/full_annotation/*/llm_eval_bbq_*.json",
    "outputs/qwen_full_14B_full_prompt/full_annotation/*/llm_eval_bbq_*.json",
    #"outputs/gpt-oss-120b_simple_prompt_medium_reasoning/**/full_annotation/*/llm_eval_bbq_*.json",
    #"outputs/gpt-oss-120b_simple_prompt_low_reasoning/**/full_annotation/*/llm_eval_bbq_*.json",
    #"outputs/gpt-oss-120b_full_prompt_low_reasoning/**/full_annotation/*/llm_eval_bbq_*.json",
    #"outputs/gpt-oss-120b_full_prompt_medium_reasoning/**/full_annotation/*/llm_eval_bbq_*.json",
)

JUDGE_LABELS = [
    "group_assumption",
    "bias_acknowledgement",
    "meta_reflection",
    "outside_demo_knowledge",
    "outside_topical_knowledge",
    "overthinking",
]

N_BOOTSTRAP = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUT_DIR = Path("reasoning_eval/analyse_labels/bbq_analysis_new")
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

def normalize_model_name(model_name: str) -> str:
    match = re.search(r"qwen\d+-\d+B", model_name, re.IGNORECASE)
    if match:
        canonical = match.group(0).capitalize()
        return f"Qwen/{canonical}"
    return model_name

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
            path_str = str(path)
            prompt_type = ("simple_prompt" if "simple_prompt" in path_str
                           else "full_prompt" if "full_prompt" in path_str
                           else r.get("prompt_type", "unknown"))
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
# Collect all JSON files
# ======================
all_paths = []
for pattern in INPUT_GLOBS:
    all_paths.extend(glob.glob(pattern, recursive=True))

if len(all_paths) == 0:
    raise RuntimeError(f"No files found for input patterns: {INPUT_GLOBS}")

df, file_hashes = load_judge_files(all_paths)
print(f"Loaded {len(df)} samples; missing judge output: {df['judge_missing'].sum()}")

# ======================
# Regression preparation
# ======================
REG_COLS = ["incorrect_and_stereotype", "is_correct", "ambiguous", "model", "category", "prompt_type", "sample_id"] + JUDGE_LABELS
reg_df = df.loc[~df["judge_missing"], REG_COLS].dropna().copy()

for col in JUDGE_LABELS:
    reg_df[col] = reg_df[col].astype(int)

reg_df["incorrect"] = (~reg_df["is_correct"].astype(bool)).astype(int)
reg_df["incorrect_and_stereotype"] = reg_df["incorrect_and_stereotype"].astype(int)
reg_df["ambiguous"] = reg_df["ambiguous"].astype("category")
reg_df["prompt_type"] = reg_df["prompt_type"].astype("category")
reg_df["model"] = reg_df["model"].astype("category").apply(normalize_model_name)
reg_df["category"] = reg_df["category"].astype("category")

# Interaction terms
interaction_prompt = " + ".join([f"{label}:C(prompt_type, Treatment(reference='simple_prompt'))" for label in JUDGE_LABELS])
interaction_ambiguous = " + ".join([f"{label}:C(ambiguous)" for label in JUDGE_LABELS])
interaction_ambig_prompt = "C(ambiguous):C(prompt_type, Treatment(reference='simple_prompt'))"

# Drop any rows with missing values in dep vars
reg_df = reg_df.dropna(subset=["incorrect", "incorrect_and_stereotype"])

# ======================
# Fit logistic regressions
# ======================
def fit_logit(formula, data, alpha=0.01):
    """Fit logistic regression with L1 regularization to mitigate quasi-separation"""
    model = smf.logit(formula=formula, data=data)
    return model.fit_regularized(method="l1", alpha=alpha, disp=False)

formula_base = "incorrect ~ " + " + ".join(JUDGE_LABELS) + \
               " + C(prompt_type, Treatment(reference='simple_prompt'))" + \
               " + " + interaction_prompt + \
               " + C(ambiguous) + " + interaction_ambiguous + \
               " + " + interaction_ambig_prompt + \
               " + C(model) + C(category)"

logit_model = fit_logit(formula_base, reg_df)

formula_stereo = "incorrect_and_stereotype ~ " + " + ".join(JUDGE_LABELS) + \
                 " + C(prompt_type, Treatment(reference='simple_prompt'))" + \
                 " + " + interaction_prompt + \
                 " + C(ambiguous) + " + interaction_ambiguous + \
                 " + " + interaction_ambig_prompt + \
                 " + C(model) + C(category)"

logit_stereo_model = fit_logit(formula_stereo, reg_df)

# ======================
# Plotting functions
# ======================
def plot_net_effects(coef, JUDGE_LABELS, OUT_DIR, filename="fig_net_effects.pdf"):
    rows = []
    question_types = ["simple_prompt", "full_prompt", "non_ambiguous", "ambiguous"]
    for qtype in question_types:
        for err in JUDGE_LABELS:
            main_effect = coef.get(err, 0)
            interaction = 0
            q_main = 0
            if qtype == "full_prompt":
                q_main = coef.get("C(prompt_type, Treatment(reference='simple_prompt'))[T.full_prompt]", 0)
                interaction = coef.get(f"{err}:C(prompt_type, Treatment(reference='simple_prompt'))[T.full_prompt]", 0)
            elif qtype == "ambiguous":
                q_main = coef.get("C(ambiguous)[T.True]", 0)
                interaction = coef.get(f"{err}:C(ambiguous)[T.True]", 0)
            net_effect = main_effect + q_main + interaction
            rows.append({"question_type": qtype, "judge_label": err, "net_logit_effect": net_effect})

    net_effect_df = pd.DataFrame(rows)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=net_effect_df, x="question_type", y="net_logit_effect", hue="judge_label", palette="tab10")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Net Logit Effect on Outcome")
    plt.title("Net Effects of Question Type × Judge Label")
    plt.xticks(rotation=0)
    plt.legend(title="Judge Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename)
    plt.close()

def plot_isolated_effects(coef, JUDGE_LABELS, OUT_DIR, filename="fig_isolated_effects.pdf"):
    rows = []
    for err in JUDGE_LABELS:
        rows.append({"factor": err, "effect_type": "Judge Label", "logit_effect": coef.get(err, 0)})
    qtype_effects = {
        "full_prompt": coef.get("C(prompt_type, Treatment(reference='simple_prompt'))[T.full_prompt]", 0),
        "ambiguous": coef.get("C(ambiguous)[T.True]", 0)
    }
    for q, val in qtype_effects.items():
        rows.append({"factor": q, "effect_type": "Question Type", "logit_effect": val})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="factor", y="logit_effect", hue="effect_type", palette="Set2")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Logit Coefficient (Isolated Effect)")
    plt.title("Isolated Main Effects of Judge Labels and Question Types")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Effect Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename)
    plt.close()


def save_logit_coefficients(model, filename):
    coef_df = pd.DataFrame({
        "coefficient": model.params,
        "std_err": model.bse,
        "z_value": model.tvalues,
        "p_value": model.pvalues
    })
    coef_df.to_csv(OUT_DIR / filename)

# ======================
# Generate plots for all models
# ======================
plot_net_effects(logit_model.params, JUDGE_LABELS, OUT_DIR, "fig_net_effects_incorrect.pdf")
plot_isolated_effects(logit_model.params, JUDGE_LABELS, OUT_DIR, "fig_isolated_effects_incorrect.pdf")

plot_net_effects(logit_stereo_model.params, JUDGE_LABELS, OUT_DIR, "fig_net_effects_stereo.pdf")
plot_isolated_effects(logit_stereo_model.params, JUDGE_LABELS, OUT_DIR, "fig_isolated_effects_stereo.pdf")


# Save coefficients for incorrect model
save_logit_coefficients(logit_model, "table_logit_model_coefficients_incorrect.csv")

# Save coefficients for incorrect_and_stereotype model
save_logit_coefficients(logit_stereo_model, "table_logit_model_coefficients_stereo.csv")

# Logistic regression predicting 'incorrect'
with open(OUT_DIR / "logit_model_summary_incorrect.txt", "w") as f:
    f.write(logit_model.summary().as_text())
print("Saved logit_model summary (incorrect) to txt.")

# Logistic regression predicting 'incorrect_and_stereotype'
with open(OUT_DIR / "logit_model_summary_stereo.txt", "w") as f:
    f.write(logit_stereo_model.summary().as_text())
print("Saved logit_model summary (incorrect_and_stereotype) to txt.")



# Overall frequency of judge labels
label_means = df[JUDGE_LABELS].mean().sort_values(ascending=False)

plt.figure(figsize=(10,6))
plt.bar(label_means.index, label_means.values, color="#1f77b4")
plt.ylabel("Proportion of Samples")
plt.title("Overall Frequency of Judge Error Labels")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_label_frequency_overall.pdf")
plt.close()

import seaborn as sns

# Compute correlation
# Ensure logit_model has been fit already
coef = logit_model.params

# Compute different aggregations
df_corr = df.copy()

# Simple sum of all error labels
df_corr["agg_errors"] = df_corr[JUDGE_LABELS].sum(axis=1)

# Sum excluding bias_acknowledgement
df_corr["agg_errors_minus"] = df_corr[[l for l in JUDGE_LABELS if l != "bias_acknowledgement"]].sum(axis=1)

# Weighted sum using logit_model coefficients
df_corr["weighted_agg_errors"] = sum(df_corr[label] * coef.get(label, 0) for label in JUDGE_LABELS)

# Binary flag if at least one error occurred
df_corr["at_least_one_error"] = (df_corr[JUDGE_LABELS].sum(axis=1) > 0).astype(int)

# Add the outcome columns
df_corr["incorrect"] = (~df_corr["is_correct"]).astype(int)
df_corr["incorrect_and_stereotype"] = df_corr["incorrect_and_stereotype"].astype(int)

# Columns to include in correlation
corr_cols = JUDGE_LABELS + [
    "incorrect",
    "incorrect_and_stereotype",
    "agg_errors",
    "agg_errors_minus",
    "weighted_agg_errors",
    "at_least_one_error"
]

corr_matrix = df_corr[corr_cols].corr()


# Save correlation CSV
corr_matrix.to_csv(OUT_DIR / "table_correlation_matrix.csv")

# Plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix: Judge Labels, Errors, Aggregates")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_correlation_matrix.pdf")
plt.close()


# ======================
# Hold-out Evaluation of Predictive Performance
# ======================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

print("\n📊 Running hold-out evaluation for predictive performance...")

# Define features and target
X_cols = JUDGE_LABELS + ["ambiguous", "prompt_type", "model", "category"]
y_cols = ["incorrect", "incorrect_and_stereotype"]

# One-hot encode categorical vars
X = pd.get_dummies(reg_df[X_cols], drop_first=True)

results_holdout = {}

for y_col in y_cols:
    y = reg_df[y_col]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Fit logistic regression
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nTarget: {y_col}")
    print(f"Hold-out Accuracy: {acc:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print("Confusion Matrix:")
    print(cm)
    print(classification_report(y_test, y_pred))

    results_holdout[y_col] = {
        "accuracy": acc,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),  # convert to list for JSON saving
    }

# Optionally save hold-out metrics
with open(OUT_DIR / "holdout_evaluation.json", "w") as f:
    import json
    json.dump(results_holdout, f, indent=2)

print("✅ Hold-out evaluation metrics saved.")








# ======================
# Save run metadata
# ======================
run_metadata = {
    "timestamp": datetime.datetime.now().isoformat(),
    "input_globs": INPUT_GLOBS,
    "n_samples": int(len(df)),
    "judge_labels": JUDGE_LABELS,
    "random_seed": RANDOM_SEED,
    "input_file_hashes": file_hashes,
}

try:
    run_metadata["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
except Exception:
    run_metadata["git_commit"] = None

with open(OUT_DIR / "run_metadata.json", "w") as f:
    json.dump(run_metadata, f, indent=2)

print(f"Analysis complete. Plots and metadata saved to {OUT_DIR}")


