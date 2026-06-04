# ======================
# Full BBQ Analysis Script with Net & Isolated Effect Plots
# Three model types: L1 regularized, standard MLE (full), MLE (bias_acknowledgement removed)
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

from sklearn.metrics import average_precision_score, roc_auc_score, classification_report, confusion_matrix

# ======================
# Config
# ======================
INPUT_GLOBS = (
    "outputs/qwen_full_8B_simple_prompt/**/full_annotation/*/llm_eval_bbq_*.json",
    "outputs/qwen_full_8B_full_prompt/full_annotation/*/llm_eval_bbq_*.json",
    "outputs/qwen_full_14B_simple_prompt/**/full_annotation/*/llm_eval_bbq_*.json",
    "outputs/qwen_full_14B_full_prompt/full_annotation/*/llm_eval_bbq_*.json",
)

JUDGE_LABELS = [
    "group_assumption",
    "bias_acknowledgement",
    "meta_reflection",
    "outside_demo_knowledge",
    "outside_topical_knowledge",
    "overthinking",
]

# bias_acknowledgement excluded due to separation-driven non-convergence in MLE
JUDGE_LABELS_NO_BIAS = [l for l in JUDGE_LABELS if l != "bias_acknowledgement"]

N_BOOTSTRAP = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUT_DIR = Path("reasoning_eval/analyse_labels/bbq_analysis_new_review")
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

reg_df = reg_df.dropna(subset=["incorrect", "incorrect_and_stereotype"])

# Interaction terms
interaction_prompt = " + ".join([f"{label}:C(prompt_type, Treatment(reference='simple_prompt'))" for label in JUDGE_LABELS])
interaction_ambiguous = " + ".join([f"{label}:C(ambiguous)" for label in JUDGE_LABELS])
interaction_ambig_prompt = "C(ambiguous):C(prompt_type, Treatment(reference='simple_prompt'))"


# ======================
# Outcome prevalence (full dataset)
# ======================

prevalence_incorrect = reg_df["incorrect"].mean()
prevalence_stereo = reg_df["incorrect_and_stereotype"].mean()

print("Prevalence (incorrect):", prevalence_incorrect)
print("Prevalence (stereotype):", prevalence_stereo)

prevalence_stats = {
    "n_total": len(reg_df),
    "prevalence_incorrect": float(prevalence_incorrect),
    "prevalence_incorrect_and_stereotype": float(prevalence_stereo),
}

with open(OUT_DIR / "prevalence.json", "w") as f:
    json.dump(prevalence_stats, f, indent=2)


# ======================
# Model fitting functions
# ======================

def fit_logit_l1(formula, data, alpha=0.01):
    """Fit logistic regression with L1 regularization to mitigate quasi-separation.
    NOTE: p-values from this model are not valid for inference."""
    model = smf.logit(formula=formula, data=data)
    return model.fit_regularized(method="l1", alpha=alpha, disp=False)


def fit_logit_mle(formula, data):
    """Fit standard (unregularized) logistic regression via MLE.
    Provides valid p-values but may be unstable under quasi-separation."""
    model = smf.logit(formula=formula, data=data)
    return model.fit(disp=False, maxiter=200)


# ======================
# Plotting functions (shared by all model types)
# ======================

def plot_net_effects(coef, JUDGE_LABELS, OUT_DIR, filename="fig_net_effects.pdf", title_suffix=""):
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
    plt.title(f"Net Effects of Question Type × Judge Label{title_suffix}")
    plt.xticks(rotation=0)
    plt.legend(title="Judge Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename)
    plt.close()


def plot_isolated_effects(coef, JUDGE_LABELS, OUT_DIR, filename="fig_isolated_effects.pdf", title_suffix=""):
    rows = []
    for err in JUDGE_LABELS:
        rows.append({"factor": err, "effect_type": "Judge Label", "logit_effect": coef.get(err, 0)})
    qtype_effects = {
        "full_prompt": coef.get("C(prompt_type, Treatment(reference='simple_prompt'))[T.full_prompt]", 0),
        "ambiguous": coef.get("C(ambiguous)[T.True]", 0)
    }
    for q, val in qtype_effects.items():
        rows.append({"factor": q, "effect_type": "Question Type", "logit_effect": val})
    df_plot = pd.DataFrame(rows)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_plot, x="factor", y="logit_effect", hue="effect_type", palette="Set2")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Logit Coefficient (Isolated Effect)")
    plt.title(f"Isolated Main Effects of Judge Labels and Question Types{title_suffix}")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Effect Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename)
    plt.close()


def save_statsmodels_coefficients(model, OUT_DIR, filename):
    """Save coefficients for a statsmodels result object (L1 or MLE)."""
    coef_df = pd.DataFrame({
        "coefficient": model.params,
        "std_err": model.bse,
        "z_value": model.tvalues,
        "p_value": model.pvalues,
        "OR_value": np.exp(model.params),
    })
    coef_df.to_csv(OUT_DIR / filename)


def save_firth_coefficients(results_df, OUT_DIR, filename):
    """Save coefficients for a Firth model results DataFrame."""
    results_df["OR_value"] = np.exp(results_df["coefficient"])
    results_df.to_csv(OUT_DIR / filename)


# ======================
# Build formulas
# ======================

def build_formula(outcome, labels):
    interaction_prompt = " + ".join([f"{l}:C(prompt_type, Treatment(reference='simple_prompt'))" for l in labels])
    interaction_ambiguous = " + ".join([f"{l}:C(ambiguous)" for l in labels])
    return (
        f"{outcome} ~ "
        + " + ".join(labels)
        + " + C(prompt_type, Treatment(reference='simple_prompt'))"
        + " + " + interaction_prompt
        + " + C(ambiguous) + " + interaction_ambiguous
        + " + C(ambiguous):C(prompt_type, Treatment(reference='simple_prompt'))"
        + " + C(model) + C(category)"
    )

formula_base        = build_formula("incorrect",                JUDGE_LABELS)
formula_stereo      = build_formula("incorrect_and_stereotype", JUDGE_LABELS)
formula_base_nb     = build_formula("incorrect",                JUDGE_LABELS_NO_BIAS)
formula_stereo_nb   = build_formula("incorrect_and_stereotype", JUDGE_LABELS_NO_BIAS)


# ======================
# Fit all models
# ======================

print("Fitting L1 models...")
logit_l1_incorrect = fit_logit_l1(formula_base, reg_df)
logit_l1_stereo    = fit_logit_l1(formula_stereo, reg_df)

print("Fitting standard MLE models (full)...")
logit_mle_incorrect = fit_logit_mle(formula_base, reg_df)
logit_mle_stereo    = fit_logit_mle(formula_stereo, reg_df)

print("Fitting MLE models without bias_acknowledgement...")
logit_mle_nb_incorrect = fit_logit_mle(formula_base_nb, reg_df)
logit_mle_nb_stereo    = fit_logit_mle(formula_stereo_nb, reg_df)

print("All models fitted.")


# ======================
# Generate plots — one set per model type per outcome
# ======================

MODEL_CONFIGS = [
    # (label, coef_dict, judge_labels_list, outcome_tag)
    ("[L1]",      logit_l1_incorrect.params,      JUDGE_LABELS,         "incorrect"),
    ("[MLE]",     logit_mle_incorrect.params,      JUDGE_LABELS,         "incorrect"),
    ("[MLE_nb]",  logit_mle_nb_incorrect.params,   JUDGE_LABELS_NO_BIAS, "incorrect"),
    ("[L1]",      logit_l1_stereo.params,           JUDGE_LABELS,         "stereo"),
    ("[MLE]",     logit_mle_stereo.params,          JUDGE_LABELS,         "stereo"),
    ("[MLE_nb]",  logit_mle_nb_stereo.params,       JUDGE_LABELS_NO_BIAS, "stereo"),
]

for model_label, coef_dict, labels, outcome in MODEL_CONFIGS:
    tag = model_label.strip("[]").lower()
    plot_net_effects(
        coef_dict, labels, OUT_DIR,
        filename=f"fig_net_effects_{outcome}_{tag}.pdf",
        title_suffix=f" {model_label} — {outcome}",
    )
    plot_isolated_effects(
        coef_dict, labels, OUT_DIR,
        filename=f"fig_isolated_effects_{outcome}_{tag}.pdf",
        title_suffix=f" {model_label} — {outcome}",
    )


# ======================
# Save coefficients
# ======================

# L1
save_statsmodels_coefficients(logit_l1_incorrect,     OUT_DIR, "table_coef_incorrect_l1.csv")
save_statsmodels_coefficients(logit_l1_stereo,        OUT_DIR, "table_coef_stereo_l1.csv")

# MLE (full)
save_statsmodels_coefficients(logit_mle_incorrect,    OUT_DIR, "table_coef_incorrect_mle.csv")
save_statsmodels_coefficients(logit_mle_stereo,       OUT_DIR, "table_coef_stereo_mle.csv")

# MLE without bias_acknowledgement
save_statsmodels_coefficients(logit_mle_nb_incorrect, OUT_DIR, "table_coef_incorrect_mle_nb.csv")
save_statsmodels_coefficients(logit_mle_nb_stereo,    OUT_DIR, "table_coef_stereo_mle_nb.csv")


# ======================
# Save model summaries (text where available)
# ======================

for model, name in [
    (logit_l1_incorrect,     "logit_summary_incorrect_l1.txt"),
    (logit_l1_stereo,        "logit_summary_stereo_l1.txt"),
    (logit_mle_incorrect,    "logit_summary_incorrect_mle.txt"),
    (logit_mle_stereo,       "logit_summary_stereo_mle.txt"),
    (logit_mle_nb_incorrect, "logit_summary_incorrect_mle_nb.txt"),
    (logit_mle_nb_stereo,    "logit_summary_stereo_mle_nb.txt"),
]:
    with open(OUT_DIR / name, "w") as f:
        f.write(model.summary().as_text())

print("Saved all summaries and coefficient tables.")


# ======================
# Overall frequency of judge labels
# ======================

label_means = df[JUDGE_LABELS].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(label_means.index, label_means.values, color="#1f77b4")
plt.ylabel("Proportion of Samples")
plt.title("Overall Frequency of Judge Error Labels")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_label_frequency_overall.pdf")
plt.close()


# ======================
# Correlation matrix
# ======================

coef_for_weighting = logit_mle_incorrect.params  # use MLE for weighting (interpretable)

df_corr = df.copy()
df_corr["agg_errors"] = df_corr[JUDGE_LABELS].sum(axis=1)
df_corr["agg_errors_minus"] = df_corr[[l for l in JUDGE_LABELS if l != "bias_acknowledgement"]].sum(axis=1)
df_corr["weighted_agg_errors"] = sum(
    df_corr[label] * coef_for_weighting.get(label, 0) for label in JUDGE_LABELS
)
df_corr["at_least_one_error"] = (df_corr[JUDGE_LABELS].sum(axis=1) > 0).astype(int)
df_corr["incorrect"] = (~df_corr["is_correct"]).astype(int)
df_corr["incorrect_and_stereotype"] = df_corr["incorrect_and_stereotype"].astype(int)

corr_cols = JUDGE_LABELS + [
    "incorrect", "incorrect_and_stereotype",
    "agg_errors", "agg_errors_minus", "weighted_agg_errors", "at_least_one_error",
]

corr_matrix = df_corr[corr_cols].corr()
corr_matrix.to_csv(OUT_DIR / "table_correlation_matrix.csv")

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix: Judge Labels, Errors, Aggregates")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_correlation_matrix.pdf")
plt.close()


# ======================
# Hold-out evaluation
# ======================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

print("\nRunning hold-out evaluation for predictive performance...")

X_cols = JUDGE_LABELS + ["ambiguous", "prompt_type", "model", "category"]
y_cols = ["incorrect", "incorrect_and_stereotype"]

X = pd.get_dummies(reg_df[X_cols], drop_first=True)

results_holdout = {}

for y_col in y_cols:
    y = reg_df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_train, y_train)

    pos_rate = y.mean()
    test_pos_rate = y_test.mean()
    train_pos_rate = y_train.mean()
    print(f"Positive rate: {pos_rate:.4f}")
    print(f"Train positive rate: {train_pos_rate:.4f}")
    print(f"Test positive rate: {test_pos_rate:.4f}")

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\nTarget: {y_col}")
    print(f"Hold-out Accuracy: {acc:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"PR-AUC: {pr_auc:.3f}")
    print("Confusion Matrix:")
    print(cm)
    print(classification_report(y_test, y_pred))

    results_holdout[y_col] = {
        "accuracy": acc,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
    }

with open(OUT_DIR / "holdout_evaluation.json", "w") as f:
    json.dump(results_holdout, f, indent=2)

print("Hold-out evaluation metrics saved.")



# ======================
# Hold-out evaluation - no bias acknowledgement
# ======================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

print("\nRunning hold-out evaluation for predictive performance...")

X_cols = JUDGE_LABELS_NO_BIAS + ["ambiguous", "prompt_type", "model", "category"]
y_cols = ["incorrect", "incorrect_and_stereotype"]

X = pd.get_dummies(reg_df[X_cols], drop_first=True)

results_holdout = {}

for y_col in y_cols:
    y = reg_df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    pos_rate = y.mean()
    test_pos_rate = y_test.mean()
    train_pos_rate = y_train.mean()
    print(f"Positive rate: {pos_rate:.4f}")
    print(f"Train positive rate: {train_pos_rate:.4f}")
    print(f"Test positive rate: {test_pos_rate:.4f}")

    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\nTarget: {y_col}")
    print(f"Hold-out Accuracy: {acc:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"PR-AUC: {pr_auc:.3f}")
    print("Confusion Matrix:")
    print(cm)
    print(classification_report(y_test, y_pred))

    results_holdout[y_col] = {
        "accuracy": acc,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
    }

with open(OUT_DIR / "holdout_evaluation_nb.json", "w") as f:
    json.dump(results_holdout, f, indent=2)

print("Hold-out evaluation metrics saved.")


# ======================
# Top L1 Predictors with 95% Bootstrap CIs (Forest Plots)
# ======================
from tqdm import tqdm

def bootstrap_l1_coefs(model_formula, data, labels, n_bootstrap=500, alpha=0.05):
    """
    Bootstrap L1 logistic regression coefficients to get empirical 95% CIs.
    Returns a DataFrame with median, lower, upper CI for each predictor.
    """
    coefs = {label: [] for label in labels}

    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping L1 coefficients"):
        sample_df = data.sample(frac=1, replace=True, random_state=None)
        try:
            fit = fit_logit_l1(model_formula, sample_df)
            for label in coefs.keys():
                coefs[label].append(fit.params.get(label, 0))
        except Exception:
            continue  # skip failed iteration

    rows = []
    for label, values in coefs.items():
        arr = np.array(values)
        median = np.median(arr)
        lower = np.percentile(arr, 100 * alpha/2)
        upper = np.percentile(arr, 100 * (1-alpha/2))
        rows.append({"predictor": label, "median": median, "ci_lower": lower, "ci_upper": upper})

    df = pd.DataFrame(rows)
    df = df.sort_values("median", key=abs, ascending=False)  # rank by absolute effect
    return df


def plot_top_l1_forest(bootstrap_df, top_n=10, title="Top L1 Predictors (95% CI)", out_path=None):
    """Plot a forest plot of top L1 predictors with 95% bootstrap CIs."""
    df_top = bootstrap_df.head(top_n).copy()
    df_top = df_top[::-1]  # reverse for plotting top at top

    plt.figure(figsize=(8, 6))
    sns.pointplot(
        x="median", y="predictor", data=df_top,
        join=False, color="blue", ci=None
    )
    # Add horizontal error bars manually
    for i, row in enumerate(df_top.itertuples()):
        plt.plot([row.ci_lower, row.ci_upper], [i, i], color="blue", lw=2)
        plt.scatter(row.median, i, color="blue", s=50)

    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("Logit Coefficient")
    plt.ylabel("Predictor")
    plt.title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    plt.show()


# # ======================
# # Generate bootstrap CIs and forest plots for L1 models
# # ======================
# print("\nBootstrapping top L1 predictors for 'incorrect' outcome...")
# bootstrap_df_incorrect = bootstrap_l1_coefs(
#     formula_base, reg_df, list(logit_l1_incorrect.params.index), n_bootstrap=500
# )

# plot_top_l1_forest(
#     bootstrap_df_incorrect,
#     top_n=10,
#     title="Top 10 L1 Predictors for 'incorrect' Outcome",
#     out_path=OUT_DIR / "forest_top10_l1_incorrect.pdf"
# )

# print("\nBootstrapping top L1 predictors for 'incorrect_and_stereotype' outcome...")
# bootstrap_df_stereo = bootstrap_l1_coefs(
#     formula_stereo, reg_df, list(logit_l1_stereo.params.index), n_bootstrap=500
# )

# plot_top_l1_forest(
#     bootstrap_df_stereo,
#     top_n=10,
#     title="Top 10 L1 Predictors for 'incorrect_and_stereotype' Outcome",
#     out_path=OUT_DIR / "forest_top10_l1_stereo.pdf"
# )



# ======================
# Save run metadata
# ======================

run_metadata = {
    "timestamp": datetime.datetime.now().isoformat(),
    "input_globs": INPUT_GLOBS,
    "n_samples": int(len(df)),
    "judge_labels": JUDGE_LABELS,
    "random_seed": RANDOM_SEED,
    "models_fitted": ["l1", "mle", "mle_no_bias_acknowledgement"],
    "input_file_hashes": file_hashes,
}

try:
    run_metadata["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
except Exception:
    run_metadata["git_commit"] = None

with open(OUT_DIR / "run_metadata.json", "w") as f:
    json.dump(run_metadata, f, indent=2)

print(f"\nAnalysis complete. All plots and tables saved to {OUT_DIR}")
print("\nOutput files per model type:")
print("  L1:     table_coef_*_l1.csv, logit_summary_*_l1.txt  (no valid p-values)")
print("  MLE:    table_coef_*_mle.csv, logit_summary_*_mle.txt  (check convergence warnings)")
print("  MLE_nb: table_coef_*_mle_nb.csv, logit_summary_*_mle_nb.txt  (bias_acknowledgement removed; use for inference)")