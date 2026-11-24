# ==========================================================================
# Reviewer-Response Analyses
# --------------------------------------------------------------------------
# Companion script to the main BBQ analysis. 

#   (R1) Non-independence of observations / non-robust SEs
#        -> Cluster-robust MLE
#        -> GEE with exchangeable working correlation on question
#        -> Bayesian GLMM with random intercept for question   [optional, slow]
#        -> Cluster bootstrap over sample_id                    [optional, slow]
#        -> Leave-one-category-out stability
#        -> Group-aware hold-out (no question leakage)
#
#   (R2) Causal language on associational estimates
#        -> E-values (VanderWeele & Ding, 2017) for each behavior coefficient
#
# Inputs, JUDGE_LABELS, formulas, and data loading mirror the main script
# so this file runs standalone.
# ==========================================================================

import json
import glob
import re
import datetime
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix,
)

from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================================================
# Config (mirrors the main script)
# ==========================================================================
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
JUDGE_LABELS_NO_BIAS = [l for l in JUDGE_LABELS if l != "bias_acknowledgement"]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- Switches for the slow sections -------------------------------------
# The cluster bootstrap (hundreds of full refits) and the Bayesian GLMM
# (one random effect per question) are by far the slowest parts. They are
# off by default; the cluster-robust MLE + GEE already answer R1.
RUN_CLUSTER_BOOTSTRAP = False
RUN_GLMM              = False
RUN_LOCO              = False   # ~1 refit per category; set False to skip
RUN_GEE               = False

N_CLUSTER_BOOTSTRAP   = 500     # only used if RUN_CLUSTER_BOOTSTRAP
N_BOOT_GLMM_SUBSAMPLE = None    # only used if RUN_GLMM (e.g. 40000)

OUT_DIR = Path("reasoning_eval/analyse_labels/bbq_analysis_clustered")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================================
# Data loading (same as main script)
# ==========================================================================
def normalize_model_name(model_name: str) -> str:
    match = re.search(r"qwen\d+-\d+B", model_name, re.IGNORECASE)
    if match:
        return f"Qwen/{match.group(0).capitalize()}"
    return model_name


def load_judge_files(paths):
    rows = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        for r in data["results"]:
            path_str = str(path)
            prompt_type = ("simple_prompt" if "simple_prompt" in path_str
                           else "full_prompt" if "full_prompt" in path_str
                           else r.get("prompt_type", "unknown"))
            row = {
                "sample_id":  r["sample_id"],
                "category":   r["category"],
                "example_id": r["example_id"],
                "context":    r.get("context", ""),
                "question":   r.get("question", ""),
                "model":      r["model"],
                "prompt_type": prompt_type,
                "is_correct":  r["is_correct"],
                "ambiguous":   r["ambiguous"],
                "incorrect_and_stereotype": r["incorrect_and_stereotype"],
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
    return pd.DataFrame(rows)


all_paths = []
for pattern in INPUT_GLOBS:
    all_paths.extend(glob.glob(pattern, recursive=True))
if len(all_paths) == 0:
    raise RuntimeError(f"No files found for input patterns: {INPUT_GLOBS}")

df = load_judge_files(all_paths)
print(f"Loaded {len(df)} samples; missing judge output: {df['judge_missing'].sum()}")


# ==========================================================================
# Regression frame prep (mirrors the main script)
# ==========================================================================
REG_COLS = (
    ["incorrect_and_stereotype", "is_correct", "ambiguous",
     "model", "category", "prompt_type", "sample_id", "example_id",
     "context", "question"]
    + JUDGE_LABELS
)
reg_df = df.loc[~df["judge_missing"], REG_COLS].dropna().copy()

for col in JUDGE_LABELS:
    reg_df[col] = reg_df[col].astype(int)

reg_df["incorrect"] = (~reg_df["is_correct"].astype(bool)).astype(int)
reg_df["incorrect_and_stereotype"] = reg_df["incorrect_and_stereotype"].astype(int)
reg_df["ambiguous"]   = reg_df["ambiguous"].astype("category")
reg_df["prompt_type"] = reg_df["prompt_type"].astype("category")
reg_df["model"]       = reg_df["model"].astype("category").apply(normalize_model_name)
reg_df["category"]    = reg_df["category"].astype("category")

# --- CRITICAL: build the cluster identifier ---
# Each (category, context, question) triple is fed to every model x prompt
# combination in our design, so residuals for rows sharing that triple are
# not independent. Model and prompt are already covariates in the
# regression, so we do NOT include them in the cluster ID.
def _triple_hash(cat, ctx, q):
    key = f"{cat}\x00{ctx}\x00{q}".encode("utf-8")
    return hashlib.md5(key).hexdigest()[:16]

# A coarser cluster: the underlying question template, sharing text across
# ambiguous / disambiguated variants. Used later as a robustness check.
reg_df["template_id"] = [
    _triple_hash(c, "", q)
    for c, q in zip(reg_df["category"], reg_df["question"])
]

n_rows       = len(reg_df)
n_questions  = reg_df["sample_id"].nunique()
avg_per_qid  = n_rows / max(n_questions, 1)

print(f"Regression frame: n_rows = {n_rows}")
print(f"  n_unique_questions (category+context+question) = {n_questions}")
print(f"  average rows per question cluster              = {avg_per_qid:.2f}")

if not (2.0 <= avg_per_qid <= 8.0):
    print(f"  WARNING: avg rows per cluster is {avg_per_qid:.2f}. "
          f"Expected ~4 (2 models x 2 prompts). "
          f"Check that 'context' and 'question' loaded correctly.")


# ==========================================================================
# Formulas
# ==========================================================================
def build_formula(outcome, labels):
    interaction_prompt = " + ".join(
        [f"{l}:C(prompt_type, Treatment(reference='simple_prompt'))" for l in labels]
    )
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

formula_incorrect_nb = build_formula("incorrect",                JUDGE_LABELS_NO_BIAS)
formula_stereo_nb    = build_formula("incorrect_and_stereotype", JUDGE_LABELS_NO_BIAS)


# ==========================================================================
# Helpers
# ==========================================================================
def save_summary(model, name):
    with open(OUT_DIR / name, "w") as f:
        f.write(model.summary().as_text())


def save_coef_table(model, name):
    coef_df = pd.DataFrame({
        "coefficient": model.params,
        "std_err":     model.bse,
        "z_value":     model.tvalues,
        "p_value":     model.pvalues,
        "OR":          np.exp(model.params),
        "ci_lo":       model.conf_int()[0],
        "ci_hi":       model.conf_int()[1],
    })
    coef_df.to_csv(OUT_DIR / name)


# ==========================================================================
# (R1a) Cluster-robust MLE  -- clusters on sample_id
# ==========================================================================
print("\n[R1a] Fitting cluster-robust MLE (cluster = sample_id)...")

def fit_logit_clustered(formula, data, cluster_col="sample_id"):
    model = smf.logit(formula=formula, data=data)
    return model.fit(
        cov_type="cluster",
        cov_kwds={"groups": data[cluster_col].values},
        disp=False,
        maxiter=200,
    )

logit_clu_incorrect = fit_logit_clustered(formula_incorrect_nb, reg_df)
logit_clu_stereo    = fit_logit_clustered(formula_stereo_nb,    reg_df)

save_summary(logit_clu_incorrect, "R1a_logit_summary_incorrect_cluster.txt")
save_summary(logit_clu_stereo,    "R1a_logit_summary_stereo_cluster.txt")
save_coef_table(logit_clu_incorrect, "R1a_coef_incorrect_cluster.csv")
save_coef_table(logit_clu_stereo,    "R1a_coef_stereo_cluster.csv")
# Full cluster-robust covariance matrix: needed to get CIs for sums of
# coefficients (behaviour main effect + interactions) in the per-condition plot.
logit_clu_stereo.cov_params().to_csv(OUT_DIR / "R1a_cov_stereo_cluster.csv")
print("  -> cluster-robust results saved.")


# ==========================================================================
# (R1a-bis) Side-by-side comparison of SEs: naive MLE vs cluster-robust
#           (reuses the sample_id-clustered fits from R1a instead of refitting)
# ==========================================================================
print("\n[R1a-bis] Building naive-vs-cluster comparison tables...")

def fit_logit_naive(formula, data):
    return smf.logit(formula=formula, data=data).fit(disp=False, maxiter=200)

for outcome, formula, clu_qid in [
    ("incorrect", formula_incorrect_nb, logit_clu_incorrect),
    ("stereo",    formula_stereo_nb,    logit_clu_stereo),
]:
    naive     = fit_logit_naive(formula, reg_df)
    clu_tmpl  = fit_logit_clustered(formula, reg_df, cluster_col="template_id")
    comp = pd.DataFrame({
        "coefficient":         naive.params,
        "se_naive":            naive.bse,
        "se_cluster_question": clu_qid.bse,
        "se_cluster_template": clu_tmpl.bse,
        "ratio_qid_naive":     clu_qid.bse  / naive.bse,
        "ratio_tmpl_naive":    clu_tmpl.bse / naive.bse,
        "p_naive":             naive.pvalues,
        "p_cluster_question":  clu_qid.pvalues,
        "p_cluster_template":  clu_tmpl.pvalues,
    })
    comp.to_csv(OUT_DIR / f"R1a_se_comparison_{outcome}.csv")
print("  -> comparison tables saved (see 'ratio_qid_naive' and")
print("     'ratio_tmpl_naive' columns).")


# ==========================================================================
# (R1a-summary) Summary statistics of SE inflation from clustering
# ==========================================================================
print("\n[R1a-summary] Computing SE inflation summary stats...")

summary_rows = []
for outcome in ["incorrect", "stereo"]:
    comp = pd.read_csv(OUT_DIR / f"R1a_se_comparison_{outcome}.csv", index_col=0)
    all_terms = comp.index.tolist()
    behavior_terms = [
        t for t in all_terms
        if any(t == l or t.startswith(l + ":") for l in JUDGE_LABELS_NO_BIAS)
    ]

    for scope_name, terms in [("all_terms", all_terms),
                              ("behavior_terms_only", behavior_terms)]:
        for cluster_kind, col in [("question", "ratio_qid_naive"),
                                  ("template", "ratio_tmpl_naive")]:
            ratios = comp.loc[terms, col].dropna()
            pct_change = (ratios - 1.0) * 100
            summary_rows.append({
                "outcome":        outcome,
                "scope":          scope_name,
                "cluster_level":  cluster_kind,
                "n_terms":        int(len(ratios)),

                "mean_ratio":     float(ratios.mean()),
                "median_ratio":   float(ratios.median()),
                "min_ratio":      float(ratios.min()),
                "max_ratio":      float(ratios.max()),
                "q25_ratio":      float(ratios.quantile(0.25)),
                "q75_ratio":      float(ratios.quantile(0.75)),

                "mean_pct_change":   float(pct_change.mean()),
                "median_pct_change": float(pct_change.median()),
                "min_pct_change":    float(pct_change.min()),
                "max_pct_change":    float(pct_change.max()),
                "q25_pct_change":    float(pct_change.quantile(0.25)),
                "q75_pct_change":    float(pct_change.quantile(0.75)),

                "pct_inflated":      float((ratios > 1.0).mean() * 100),
                "pct_deflated":      float((ratios < 1.0).mean() * 100),
                "n_flipped_sig_at_05": int(
                    ((comp.loc[terms, "p_naive"] < 0.05)
                     != (comp.loc[terms, f"p_cluster_{cluster_kind}"] < 0.05)).sum()
                ),
            })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "R1a_se_inflation_summary.csv", index=False)

print("\n  SE inflation summary (behavior terms only, cluster on question):")
key_rows = summary_df.query(
    "scope == 'behavior_terms_only' and cluster_level == 'question'"
)
for _, r in key_rows.iterrows():
    print(f"    outcome = {r['outcome']:>10s} | "
          f"n = {r['n_terms']:2d} | "
          f"mean ratio = {r['mean_ratio']:.2f} ({r['mean_pct_change']:+.1f}%) | "
          f"median = {r['median_ratio']:.2f} ({r['median_pct_change']:+.1f}%) | "
          f"range = [{r['min_pct_change']:+.1f}%, {r['max_pct_change']:+.1f}%] | "
          f"flipped @ .05 = {r['n_flipped_sig_at_05']}")
print("  -> full table saved to R1a_se_inflation_summary.csv")


# ==========================================================================
# (R1a-diag) Residual diagnostics: quantifying within-cluster dependence
# ==========================================================================
print("\n[R1a-diag] Residual diagnostics for within-cluster correlation...")

def icc_from_residuals(residuals, cluster_ids):
    """One-way random-effects ICC (ANOVA estimator) on residuals."""
    d = pd.DataFrame({"r": np.asarray(residuals), "g": np.asarray(cluster_ids)})
    grouped        = d.groupby("g")["r"]
    cluster_sizes  = grouped.size()
    cluster_means  = grouped.mean()
    overall_mean   = d["r"].mean()
    n_total, n_clu = len(d), len(cluster_means)
    m_bar          = n_total / n_clu

    SSB = (cluster_sizes * (cluster_means - overall_mean) ** 2).sum()
    MSB = SSB / (n_clu - 1)
    d_m = d.merge(cluster_means.rename("gm"), left_on="g", right_index=True)
    SSW = ((d_m["r"] - d_m["gm"]) ** 2).sum()
    MSW = SSW / (n_total - n_clu)
    icc = (MSB - MSW) / (MSB + (m_bar - 1) * MSW)

    return {"icc": float(icc), "n_obs": int(n_total), "n_clusters": int(n_clu),
            "m_bar": float(m_bar), "MSB": float(MSB), "MSW": float(MSW)}


for outcome, model_fit, y_col in [
    ("incorrect", logit_clu_incorrect, "incorrect"),
    ("stereo",    logit_clu_stereo,    "incorrect_and_stereotype"),
]:
    y_hat = model_fit.predict(reg_df)
    y_obs = reg_df[y_col].values
    denom = np.sqrt(np.clip(y_hat * (1 - y_hat), 1e-10, None))
    pearson_resid = (y_obs - y_hat) / denom

    stats = icc_from_residuals(pearson_resid, reg_df["sample_id"])
    icc, m_bar = stats["icc"], stats["m_bar"]

    design_effect          = 1 + (m_bar - 1) * icc
    predicted_se_inflation = np.sqrt(max(design_effect, 0))

    comp = pd.read_csv(OUT_DIR / f"R1a_se_comparison_{outcome}.csv", index_col=0)
    behavior_terms = [t for t in comp.index
                      if any(t == l or t.startswith(l + ":") for l in JUDGE_LABELS_NO_BIAS)]
    observed_median = float(comp.loc[behavior_terms, "ratio_qid_naive"].median())
    observed_mean   = float(comp.loc[behavior_terms, "ratio_qid_naive"].mean())

    result = {
        **stats,
        "design_effect":              float(design_effect),
        "predicted_se_inflation":     float(predicted_se_inflation),
        "observed_median_inflation":  observed_median,
        "observed_mean_inflation":    observed_mean,
        "match_ratio":                float(observed_median / predicted_se_inflation),
    }
    with open(OUT_DIR / f"R1a_diag_icc_{outcome}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Outcome: {outcome}")
    print(f"    ICC of Pearson residuals    : {icc:.4f}")
    print(f"    Cluster size (avg)          : {m_bar:.2f}")
    print(f"    Predicted design effect     : {design_effect:.3f}")
    print(f"    Predicted SE inflation √DE  : {predicted_se_inflation:.3f}")
    print(f"    Observed median (behaviors) : {observed_median:.3f}")
    print(f"    Observed mean   (behaviors) : {observed_mean:.3f}")

    reg_df[f"pearson_resid_{outcome}"] = pearson_resid


# --- Single diagnostic figure: three panels, main outcome ---
outcome   = "stereo"
resid_col = f"pearson_resid_{outcome}"

cluster_means = reg_df.groupby("sample_id")[resid_col].mean()
cluster_sizes = reg_df.groupby("sample_id")[resid_col].size()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel A: cluster-mean residuals vs. null (independence)
sigma_within = np.sqrt(
    reg_df.groupby("sample_id")[resid_col]
          .apply(lambda x: ((x - x.mean()) ** 2).sum()).sum()
    / (len(reg_df) - reg_df["sample_id"].nunique())
)
m_bar   = len(reg_df) / reg_df["sample_id"].nunique()
null_sd = sigma_within / np.sqrt(m_bar)

ax = axes[0]
ax.hist(cluster_means, bins=60, density=True, alpha=0.65,
        color="#4c72b0", label="Observed cluster means")
xs = np.linspace(cluster_means.min(), cluster_means.max(), 300)
ax.plot(xs, norm.pdf(xs, 0, null_sd), color="black", lw=2,
        label=f"Null under independence\nN(0, {null_sd:.2f})")
ax.axvline(0, color="grey", ls="--", lw=0.8)
ax.set_xlabel("Cluster-mean Pearson residual")
ax.set_ylabel("Density")
ax.set_title("A. Cluster means vs. independence null\n(wider than null ⇒ within-cluster dep.)")
ax.legend(fontsize=8)

# Panel B: within-cluster pairwise residual products
full_clusters = cluster_sizes[cluster_sizes == int(round(m_bar))].index
sub = reg_df[reg_df["sample_id"].isin(full_clusters)]
prods = sub.groupby("sample_id")[resid_col].apply(
    lambda x: pd.Series(np.outer(x.values, x.values)[np.triu_indices(len(x), k=1)])
).values

ax = axes[1]
ax.hist(prods, bins=60, density=True, alpha=0.7, color="#dd8452")
ax.axvline(0, color="black", ls="--", lw=0.8)
ax.axvline(np.mean(prods), color="red", lw=2,
           label=f"Mean = {np.mean(prods):.3f}\n(>0 ⇒ positive residual corr.)")
ax.set_xlabel("Within-cluster residual product  rᵢ × rⱼ")
ax.set_ylabel("Density")
ax.set_title("B. Within-cluster residual products\n(mean > 0 ⇒ clustering)")
ax.legend(fontsize=8)

# Panel C: visual "clumping" for 100 sampled clusters
rng = np.random.default_rng(RANDOM_SEED)
sample = rng.choice(full_clusters,
                    size=min(100, len(full_clusters)), replace=False)
sub = reg_df[reg_df["sample_id"].isin(sample)].copy()
order = sub.groupby("sample_id")[resid_col].mean().sort_values().index
sub["rank"] = sub["sample_id"].map({c: i for i, c in enumerate(order)})

ax = axes[2]
ax.scatter(sub["rank"], sub[resid_col], s=12, alpha=0.55, color="#55a868")
ax.axhline(0, color="black", ls="--", lw=0.8)
ax.set_xlabel("Cluster (sorted by mean residual)")
ax.set_ylabel("Pearson residual")
ax.set_title("C. 100 sampled clusters\n(vertical clumps ⇒ within-cluster corr.)")

plt.tight_layout()
plt.savefig(OUT_DIR / f"R1a_diag_residual_plots_{outcome}.pdf")
plt.close()
print("  -> diagnostic plots saved.")


# ==========================================================================
# (R1a-diag2) Predictor prevalence and within- vs between-cluster variation
# ==========================================================================
print("\n[R1a-diag2] Predictor prevalence and within/between decomposition...")

RARE_PREVALENCE_THRESHOLD = 0.05
LOW_WITHIN_VAR_THRESHOLD  = 0.25

comp_stereo = pd.read_csv(OUT_DIR / "R1a_se_comparison_stereo.csv", index_col=0)

rows = []
for term in JUDGE_LABELS_NO_BIAS:
    x = reg_df[term].astype(float)

    prevalence   = float(x.mean())
    var_total    = float(x.var())
    var_between  = float(reg_df.groupby("sample_id")[term].mean().var())
    within_frac  = (1 - var_between / var_total) if var_total > 0 else np.nan

    cluster_activation = reg_df.groupby("sample_id")[term].mean()
    pct_clusters_mixed  = float(((cluster_activation > 0)
                                 & (cluster_activation < 1)).mean() * 100)
    pct_clusters_never  = float((cluster_activation == 0).mean() * 100)
    pct_clusters_always = float((cluster_activation == 1).mean() * 100)

    in_comp = term in comp_stereo.index
    se_ratio     = float(comp_stereo.loc[term, "ratio_qid_naive"]) if in_comp else np.nan
    se_naive_val = float(comp_stereo.loc[term, "se_naive"]) if in_comp else np.nan
    se_clu_val   = float(comp_stereo.loc[term, "se_cluster_question"]) if in_comp else np.nan

    is_rare           = prevalence < RARE_PREVALENCE_THRESHOLD
    is_mostly_between = (not np.isnan(within_frac)) and within_frac < LOW_WITHIN_VAR_THRESHOLD

    rows.append({
        "behavior":                term,
        "prevalence":              prevalence,
        "within_cluster_var_frac": within_frac,
        "pct_clusters_mixed":      pct_clusters_mixed,
        "pct_clusters_never":      pct_clusters_never,
        "pct_clusters_always":     pct_clusters_always,
        "se_naive":                se_naive_val,
        "se_cluster_question":     se_clu_val,
        "se_ratio":                se_ratio,
        "flag_rare":               is_rare,
        "flag_mostly_between":     is_mostly_between,
    })

prev_df = pd.DataFrame(rows).sort_values("prevalence", ascending=False)
prev_df.to_csv(OUT_DIR / "R1a_diag_predictor_prevalence.csv", index=False)

print("\n  Behavior prevalence and within-cluster variation:")
print(f"  {'behavior':<30s} {'prev':>6s} {'within%':>8s} {'mixed%':>8s} "
      f"{'se_ratio':>9s} {'flags':>18s}")
for _, r in prev_df.iterrows():
    flags = []
    if r["flag_rare"]:            flags.append("RARE")
    if r["flag_mostly_between"]:  flags.append("BETWEEN")
    print(f"  {r['behavior']:<30s} "
          f"{r['prevalence']:>6.3f} "
          f"{100*r['within_cluster_var_frac']:>7.1f}% "
          f"{r['pct_clusters_mixed']:>7.1f}% "
          f"{r['se_ratio']:>9.3f} "
          f"{','.join(flags) if flags else '-':>18s}")

n_rare    = int(prev_df["flag_rare"].sum())
n_between = int(prev_df["flag_mostly_between"].sum())
safe      = prev_df[~prev_df["flag_rare"] & ~prev_df["flag_mostly_between"]]
n_safe    = len(safe)

diag_summary = {
    "n_behaviors_total":          len(prev_df),
    "n_behaviors_rare":           n_rare,
    "n_behaviors_mostly_between": n_between,
    "n_behaviors_safe":           n_safe,
    "rare_threshold":             RARE_PREVALENCE_THRESHOLD,
    "within_frac_threshold":      LOW_WITHIN_VAR_THRESHOLD,
    "median_within_frac_safe":    float(safe["within_cluster_var_frac"].median())
                                  if n_safe else None,
    "median_se_ratio_safe":       float(safe["se_ratio"].median())
                                  if n_safe else None,
    "flagged_behaviors":          prev_df.loc[
        prev_df["flag_rare"] | prev_df["flag_mostly_between"], "behavior"
    ].tolist(),
}
with open(OUT_DIR / "R1a_diag_predictor_prevalence_summary.json", "w") as f:
    json.dump(diag_summary, f, indent=2)

if n_rare > 0 or n_between > 0:
    print(f"\n  !! {n_rare} behavior(s) below {RARE_PREVALENCE_THRESHOLD:.0%} prevalence")
    print(f"  !! {n_between} behavior(s) below {LOW_WITHIN_VAR_THRESHOLD:.0%} within-cluster variation")
    print(f"  !! Be cautious about the 'small SE ratio = robustness' claim for those.")
else:
    print(f"\n  All {len(prev_df)} behaviors clear both thresholds — the strong")
    print(f"  version of the rebuttal paragraph (within-cluster identification)")
    print(f"  is defensible for every behavior coefficient.")
print("  -> prevalence/decomposition results saved.")


# ==========================================================================
# (R1b) GEE with exchangeable working correlation on question
# ==========================================================================
if RUN_GEE:
    print("\n[R1b] Fitting GEE (exchangeable working corr. on sample_id)...")

    def fit_gee(formula, data, group_col="sample_id"):
        return smf.gee(
            formula,
            groups=group_col,
            data=data,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()

    try:
        gee_incorrect = fit_gee(formula_incorrect_nb, reg_df)
        gee_stereo    = fit_gee(formula_stereo_nb,    reg_df)
        save_summary(gee_incorrect, "R1b_gee_summary_incorrect.txt")
        save_summary(gee_stereo,    "R1b_gee_summary_stereo.txt")
        save_coef_table(gee_incorrect, "R1b_gee_coef_incorrect.csv")
        save_coef_table(gee_stereo,    "R1b_gee_coef_stereo.csv")
        print("  -> GEE results saved.")
    except Exception as e:
        print(f"  !! GEE fit failed: {e}")
else:
    print("\n[R1b] GEE skipped (RUN_GEE = False).")


# ==========================================================================
# (R1c) Bayesian GLMM with random intercept for question   [SLOW, optional]
# ==========================================================================
if RUN_GLMM:
    print("\n[R1c] Fitting Bayesian GLMM (random intercept for sample_id)...")
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

        glmm_data = reg_df
        if N_BOOT_GLMM_SUBSAMPLE is not None and len(reg_df) > N_BOOT_GLMM_SUBSAMPLE:
            glmm_data = reg_df.sample(N_BOOT_GLMM_SUBSAMPLE, random_state=RANDOM_SEED)
            print(f"  (subsampled to {len(glmm_data)} rows for tractability)")

        vc_formulas = {"question": "0 + C(sample_id)"}

        glmm_stereo = BinomialBayesMixedGLM.from_formula(
            formula_stereo_nb, vc_formulas, glmm_data
        ).fit_vb()

        glmm_df = pd.DataFrame({
            "coefficient":  glmm_stereo.fe_mean,
            "posterior_sd": glmm_stereo.fe_sd,
        }, index=glmm_stereo.model.exog_names)
        glmm_df["OR"]    = np.exp(glmm_df["coefficient"])
        glmm_df["ci_lo"] = np.exp(glmm_df["coefficient"] - 1.96 * glmm_df["posterior_sd"])
        glmm_df["ci_hi"] = np.exp(glmm_df["coefficient"] + 1.96 * glmm_df["posterior_sd"])
        glmm_df.to_csv(OUT_DIR / "R1c_glmm_stereo_coefficients.csv")
        print("  -> GLMM results saved.")
    except Exception as e:
        print(f"  !! GLMM fit failed: {e}")
        print("     (Consider setting N_BOOT_GLMM_SUBSAMPLE or using pymer4 / R's lme4.)")
else:
    print("\n[R1c] Bayesian GLMM skipped (RUN_GLMM = False).")


# ==========================================================================
# (R1d) Cluster bootstrap over sample_id                    [SLOW, optional]
#       Design matrix is built once and row indices are precomputed with
#       groupby, so each iteration only re-runs the Newton solver.
# ==========================================================================
if RUN_CLUSTER_BOOTSTRAP:
    print(f"\n[R1d] Cluster bootstrap ({N_CLUSTER_BOOTSTRAP} iters) over sample_id...")

    def cluster_bootstrap(formula, data, cluster_col, n_boot, seed=RANDOM_SEED):
        rng = np.random.default_rng(seed)
        y_all, X_all = patsy.dmatrices(formula, data, return_type="dataframe")
        y_arr, X_arr = y_all.values.ravel(), X_all.values
        # positional row indices per cluster (single pass, not one scan per cluster)
        groups = data.loc[y_all.index, cluster_col]
        idx_by_cluster = list(groups.reset_index(drop=True)
                                    .groupby(groups.values, sort=False).indices.values())
        n_clu = len(idx_by_cluster)
        coefs = []
        for _ in tqdm(range(n_boot)):
            sampled = rng.integers(0, n_clu, size=n_clu)
            idxs = np.concatenate([idx_by_cluster[c] for c in sampled])
            try:
                fit = sm.Logit(y_arr[idxs], X_arr[idxs]).fit(disp=False, maxiter=100)
                coefs.append(fit.params)
            except Exception:
                continue
        return pd.DataFrame(coefs, columns=X_all.columns)

    boot_stereo = cluster_bootstrap(
        formula_stereo_nb, reg_df, "sample_id", N_CLUSTER_BOOTSTRAP
    )
    boot_summary = pd.DataFrame({
        "coef_median": boot_stereo.median(),
        "coef_mean":   boot_stereo.mean(),
        "boot_se":     boot_stereo.std(),
        "ci_lo_2.5":   boot_stereo.quantile(0.025),
        "ci_hi_97.5":  boot_stereo.quantile(0.975),
        "n_successful_iters": len(boot_stereo),
    })
    boot_summary.to_csv(OUT_DIR / "R1d_cluster_bootstrap_stereo.csv")
    boot_stereo.to_csv(OUT_DIR / "R1d_cluster_bootstrap_stereo_raw.csv", index=False)
    print(f"  -> {len(boot_stereo)}/{N_CLUSTER_BOOTSTRAP} iters succeeded.")
else:
    print("\n[R1d] Cluster bootstrap skipped (RUN_CLUSTER_BOOTSTRAP = False).")


# ==========================================================================
# (R1e) Leave-one-category-out stability
# ==========================================================================
if RUN_LOCO:
    print("\n[R1e] Leave-one-category-out stability check...")

    loco_rows = []
    categories = list(reg_df["category"].cat.categories)
    for cat in tqdm(categories):
        sub = reg_df[reg_df["category"] != cat].copy()
        sub["category"] = sub["category"].cat.remove_unused_categories()
        try:
            fit = fit_logit_clustered(formula_stereo_nb, sub)
            for label in JUDGE_LABELS_NO_BIAS:
                if label in fit.params.index:
                    loco_rows.append({
                        "dropped_category": cat,
                        "behavior":         label,
                        "coef":             fit.params[label],
                        "se_cluster":       fit.bse[label],
                        "p_value":          fit.pvalues[label],
                        "OR":               np.exp(fit.params[label]),
                    })
        except Exception as e:
            print(f"  !! LOCO fit failed for dropped={cat}: {e}")

    loco_df = pd.DataFrame(loco_rows)
    loco_df.to_csv(OUT_DIR / "R1e_loco_stability_stereo.csv", index=False)

    if len(loco_df) > 0:
        loco_summary = loco_df.groupby("behavior")["coef"].agg(
            ["mean", "std", "min", "max"]
        )
        loco_summary["coef_full_sample"] = [
            logit_clu_stereo.params.get(b, np.nan) for b in loco_summary.index
        ]
        loco_summary.to_csv(OUT_DIR / "R1e_loco_summary_stereo.csv")
    print("  -> LOCO stability results saved.")
else:
    print("\n[R1e] LOCO skipped (RUN_LOCO = False).")


# ==========================================================================
# (R1f) Group-aware hold-out (no question leakage)
# ==========================================================================
print("\n[R1f] Group-aware hold-out evaluation (split on sample_id)...")

X_cols = JUDGE_LABELS_NO_BIAS + ["ambiguous", "prompt_type", "model", "category"]
y_cols = ["incorrect", "incorrect_and_stereotype"]

X = pd.get_dummies(reg_df[X_cols], drop_first=True)
groups = reg_df["sample_id"].values

results_grouped = {}
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)

for y_col in y_cols:
    y = reg_df[y_col].values
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_qs = set(reg_df["sample_id"].iloc[train_idx])
    test_qs  = set(reg_df["sample_id"].iloc[test_idx])
    overlap  = train_qs & test_qs
    assert len(overlap) == 0, f"Question leakage: {len(overlap)} shared questions"

    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n  Target: {y_col}")
    print(f"    n_train = {len(y_train)}, n_test = {len(y_test)}")
    print(f"    n_train_questions = {len(train_qs)}, "
          f"n_test_questions = {len(test_qs)}, overlap = {len(overlap)}")
    print(f"    Accuracy: {acc:.3f} | ROC-AUC: {auc:.3f} | PR-AUC: {pr_auc:.3f}")

    results_grouped[y_col] = {
        "n_train":           int(len(y_train)),
        "n_test":            int(len(y_test)),
        "n_train_questions": int(len(train_qs)),
        "n_test_questions":  int(len(test_qs)),
        "question_overlap":  int(len(overlap)),
        "accuracy":          float(acc),
        "roc_auc":           float(auc),
        "pr_auc":            float(pr_auc),
        "confusion_matrix":  cm.tolist(),
    }

with open(OUT_DIR / "R1f_grouped_holdout_evaluation.json", "w") as f:
    json.dump(results_grouped, f, indent=2)
print("  -> grouped hold-out results saved.")


# ==========================================================================
# (R1g) Grouped 5-fold cross-validation (no question leakage)
#       A single split with ~170 positives gives a noisy PR-AUC, so we
#       report mean +/- SD over folds. StratifiedGroupKFold keeps every
#       sample_id in one fold AND spreads the rare positives evenly, so
#       each fold's base rate is close to the overall one.
# ==========================================================================
N_CV_FOLDS = 5
print(f"\n[R1g] Grouped {N_CV_FOLDS}-fold CV (groups = sample_id)...")

cv_fold_rows = []
cv_summary   = {}
for y_col in y_cols:
    y = reg_df[y_col].values
    sgkf = StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True,
                                random_state=RANDOM_SEED)
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        overlap = (set(groups[train_idx]) & set(groups[test_idx]))
        assert len(overlap) == 0, f"Question leakage in fold {fold}"

        clf = LogisticRegression(max_iter=1000, solver="liblinear")
        clf.fit(X.iloc[train_idx], y[train_idx])
        y_proba = clf.predict_proba(X.iloc[test_idx])[:, 1]
        y_test  = y[test_idx]

        base_rate = float(y_test.mean())          # = PR-AUC of a random classifier
        pr_auc    = float(average_precision_score(y_test, y_proba))
        cv_fold_rows.append({
            "outcome":     y_col,
            "fold":        fold,
            "n_test":      int(len(test_idx)),
            "n_positives": int(y_test.sum()),
            "base_rate":   base_rate,
            "roc_auc":     float(roc_auc_score(y_test, y_proba)),
            "pr_auc":      pr_auc,
            "pr_auc_lift": pr_auc / base_rate,    # "x times higher than random"
        })

    f_df = pd.DataFrame([r for r in cv_fold_rows if r["outcome"] == y_col])
    cv_summary[y_col] = {
        "n_folds":          N_CV_FOLDS,
        "positives_per_fold": f_df["n_positives"].tolist(),
        **{f"{m}_{s}": float(getattr(f_df[m], s)())
           for m in ["roc_auc", "pr_auc", "pr_auc_lift", "base_rate"]
           for s in ["mean", "std"]},
    }
    s = cv_summary[y_col]
    print(f"\n  Target: {y_col}")
    print(f"    positives per fold : {s['positives_per_fold']}")
    print(f"    ROC-AUC            : {s['roc_auc_mean']:.3f} ± {s['roc_auc_std']:.3f}")
    print(f"    PR-AUC             : {s['pr_auc_mean']:.3f} ± {s['pr_auc_std']:.3f}")
    print(f"    PR-AUC / base rate : {s['pr_auc_lift_mean']:.1f}x ± {s['pr_auc_lift_std']:.1f}"
          f"  (base rate {100*s['base_rate_mean']:.2f}%)")

pd.DataFrame(cv_fold_rows).to_csv(OUT_DIR / "R1g_grouped_cv_folds.csv", index=False)
with open(OUT_DIR / "R1g_grouped_cv_summary.json", "w") as f:
    json.dump(cv_summary, f, indent=2)
print("  -> grouped CV results saved.")


# ==========================================================================
# (R2) E-values for the causal-language concern
# ==========================================================================
print("\n[R2] Computing E-values for behavior coefficients...")

def e_value_from_rr(rr):
    """E-value formula from VanderWeele & Ding (2017), Annals of Internal Medicine."""
    rr = max(rr, 1.0 / rr)
    return rr + np.sqrt(rr * (rr - 1.0))

def or_to_rr(or_val, prevalence):
    """Approximate risk ratio from odds ratio for a given outcome prevalence."""
    return or_val / ((1.0 - prevalence) + prevalence * or_val)

def e_value_from_or(or_val, prevalence):
    return e_value_from_rr(or_to_rr(or_val, prevalence))

evalue_rows = []
for outcome, model in [
    ("incorrect",                logit_clu_incorrect),
    ("incorrect_and_stereotype", logit_clu_stereo),
]:
    prev = reg_df[outcome].mean()
    params = model.params
    conf   = model.conf_int()
    for term in params.index:
        #if not any(term.startswith(l) or term == l for l in JUDGE_LABELS_NO_BIAS):
        #    continue
        or_pt = np.exp(params[term])
        or_lo = np.exp(conf.loc[term, 0])
        or_hi = np.exp(conf.loc[term, 1])
        or_ci_bound = or_lo if or_pt > 1 else or_hi
        evalue_rows.append({
            "outcome":       outcome,
            "term":          term,
            "OR":            or_pt,
            "OR_ci_lo":      or_lo,
            "OR_ci_hi":      or_hi,
            "prevalence":    prev,
            "E_value_point": e_value_from_or(or_pt,       prev),
            "E_value_CI":    e_value_from_or(or_ci_bound, prev),
        })

evalue_df = pd.DataFrame(evalue_rows)
evalue_df.to_csv(OUT_DIR / "R2_evalues.csv", index=False)
print("  -> E-values saved. Interpretation guide:")
print("     E-value = minimum RR an unmeasured confounder would need to have")
print("     with BOTH the behavior AND the outcome to fully explain the")
print("     observed association away.")


# ==========================================================================
# Quick-look plot: SE inflation from clustering
# ==========================================================================
print("\n[plot] Making SE-inflation figure...")

try:
    comp = pd.read_csv(OUT_DIR / "R1a_se_comparison_stereo.csv", index_col=0)
    behavior_terms = [
        t for t in comp.index
        if any(t == l or t.startswith(l + ":") for l in JUDGE_LABELS_NO_BIAS)
    ]
    sub = comp.loc[behavior_terms].sort_values("ratio_qid_naive", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(sub))))
    y = np.arange(len(sub))
    ax.barh(y - 0.2, sub["ratio_qid_naive"],  height=0.4,
            color="#4c72b0", label="Cluster: question (cat+ctx+q)")
    ax.barh(y + 0.2, sub["ratio_tmpl_naive"], height=0.4,
            color="#dd8452", label="Cluster: template (cat+q)")
    ax.axvline(1.0, color="black", lw=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(sub.index, fontsize=8)
    ax.set_xlabel("SE(cluster-robust) / SE(naive)")
    ax.set_title("SE inflation from clustering on BBQ question")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "R1a_se_inflation_stereo.pdf")
    plt.close()
except Exception as e:
    print(f"  !! plotting failed: {e}")


# ==========================================================================
# Run metadata
# ==========================================================================
analyses_run = ["R1a_cluster_robust_mle", "R1a_se_comparison"]
if RUN_GEE:               analyses_run.append("R1b_gee_exchangeable")
if RUN_GLMM:              analyses_run.append("R1c_bayesian_glmm")
if RUN_CLUSTER_BOOTSTRAP: analyses_run.append("R1d_cluster_bootstrap")
if RUN_LOCO:              analyses_run.append("R1e_leave_one_category_out")
analyses_run += ["R1f_grouped_holdout", "R1g_grouped_cv", "R2_evalues"]

run_metadata = {
    "timestamp":           datetime.datetime.now().isoformat(),
    "n_samples":           int(len(reg_df)),
    "n_unique_questions":  int(reg_df["sample_id"].nunique()),
    "random_seed":         RANDOM_SEED,
    "n_cluster_bootstrap": N_CLUSTER_BOOTSTRAP if RUN_CLUSTER_BOOTSTRAP else 0,
    "glmm_subsample":      N_BOOT_GLMM_SUBSAMPLE if RUN_GLMM else None,
    "analyses":            analyses_run,
    "output_directory":    str(OUT_DIR),
}
with open(OUT_DIR / "run_metadata.json", "w") as f:
    json.dump(run_metadata, f, indent=2)

print(f"\n{'=' * 60}")
print(f"Reviewer-response analyses complete.")
print(f"All outputs in: {OUT_DIR}")
print(f"{'=' * 60}")
print("\nSuggested rebuttal citations:")
print("  R1: R1a_coef_stereo_cluster.csv          -- main cluster-robust table")
print("      R1a_se_comparison_stereo.csv         -- naive vs cluster SEs")
if RUN_GEE:
    print("      R1b_gee_coef_stereo.csv              -- GEE agreement")
if RUN_GLMM:
    print("      R1c_glmm_stereo_coefficients.csv     -- GLMM agreement")
if RUN_CLUSTER_BOOTSTRAP:
    print("      R1d_cluster_bootstrap_stereo.csv     -- non-parametric CIs")
if RUN_LOCO:
    print("      R1e_loco_summary_stereo.csv          -- category robustness")
print("      R1f_grouped_holdout_evaluation.json  -- no-leakage prediction")
print("      R1g_grouped_cv_summary.json          -- grouped 5-fold CV, mean ± SD")
print("  R2: R2_evalues.csv                       -- E-values per behavior")