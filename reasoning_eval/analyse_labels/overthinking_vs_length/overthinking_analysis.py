# ======================
# Overthinking vs Reasoning Length Analysis
# Addresses reviewer critique: "overthinking may be a proxy for CoT length"
#
# Argument: overthinking captures different signal to purely length.
#
# Three models — identical structure to main analysis build_formula(),
# MLE without bias_acknowledgement:
#   M1 — JUDGE_LABELS_NO_BIAS (replicates main paper)
#   M2 — JUDGE_LABELS_NO_BIAS with overthinking replaced by log_reasoning_tokens
#   M3 — JUDGE_LABELS_NO_BIAS with log_reasoning_tokens added alongside overthinking
#
# Separate section characterises the overthinking–length relationship.
# ======================

import json
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

# ======================
# Config
# ======================

BASE_DIRS = [
    "outputs/qwen_full_8B_simple_prompt/20250827_163953",
    "outputs/qwen_full_8B_full_prompt",
    "outputs/qwen_full_14B_simple_prompt/20250828_215719",
    "outputs/qwen_full_14B_full_prompt",
   # "outputs/gpt-oss-120b_simple_prompt_low_reasoning/20251216_114545",
   # "outputs/gpt-oss-120b_simple_prompt_medium_reasoning/20251217_110543",
   # "outputs/gpt-oss-120b_full_prompt_low_reasoning/20251218_140849",
   # "outputs/gpt-oss-120b_full_prompt_low_reasoning/20251225_204037",
   # "outputs/gpt-oss-120b_full_prompt_medium_reasoning/20251218_113157",
   # "outputs/gpt-oss-120b_full_prompt_medium_reasoning/20251225_224835",
   # "outputs/gpt-oss-120b_full_prompt_medium_reasoning/20251226_123752",
]

JUDGE_LABELS = [
    "group_assumption",
    "bias_acknowledgement",
    "meta_reflection",
    "outside_demo_knowledge",
    "outside_topical_knowledge",
    "overthinking"
]

# Matches main script: bias_acknowledgement removed for separation stability
JUDGE_LABELS_NO_BIAS = [l for l in JUDGE_LABELS if l != "bias_acknowledgement"]

# M2: replace overthinking with log_reasoning_tokens
LABELS_LENGTH_ONLY = [l for l in JUDGE_LABELS_NO_BIAS if l != "overthinking"] + ["log_reasoning_tokens"]

# M3: add log_reasoning_tokens alongside overthinking
LABELS_BOTH = JUDGE_LABELS_NO_BIAS + ["log_reasoning_tokens"]

OUT_DIR = Path("reasoning_eval/analyse_labels/overthinking_vs_length")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ======================
# build_formula — exact copy from main analysis script
# ======================

def build_formula(outcome, labels):
    interaction_prompt    = " + ".join([f"{l}:C(prompt_type, Treatment(reference='simple_prompt'))" for l in labels])
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

def build_formula_with_length_ctrl(outcome, labels):
    """
    log_reasoning_tokens enters as a simple covariate (no interactions),
    alongside prompt_type, model, category.
    Judge labels get the full interaction structure as before.
    """
    interaction_prompt    = " + ".join([
        f"{l}:C(prompt_type, Treatment(reference='simple_prompt'))" 
        for l in labels
    ])
    interaction_ambiguous = " + ".join([
        f"{l}:C(ambiguous)" 
        for l in labels
    ])
    return (
        f"{outcome} ~ "
        + " + ".join(labels)
        + " + log_reasoning_tokens"          # <-- length as simple covariate
        + " + log_reasoning_tokens:C(prompt_type, Treatment(reference='simple_prompt'))"  # add this
        + " + log_reasoning_tokens:C(ambiguous)"      
        + " + C(prompt_type, Treatment(reference='simple_prompt'))"
        + " + " + interaction_prompt
        + " + C(ambiguous) + " + interaction_ambiguous
        + " + C(ambiguous):C(prompt_type, Treatment(reference='simple_prompt'))"
        + " + C(model) + C(category)"
    )


def fit_logit_mle(formula, data):
    return smf.logit(formula=formula, data=data).fit(disp=False, maxiter=200)


# ======================
# Helpers
# ======================

def parse_model_prompt(base_dir):
    parts = base_dir.rstrip("/").split(os.sep)
    try:
        model_prompt_folder = parts[parts.index("outputs") + 1]
    except ValueError:
        model_prompt_folder = parts[-1]
    tokens = model_prompt_folder.split("_")
    if model_prompt_folder.startswith("qwen_full_8B"):
        model_type, rest = "Qwen3-8B", "_".join(tokens[3:])
    elif model_prompt_folder.startswith("qwen_full_14B"):
        model_type, rest = "Qwen3-14B", "_".join(tokens[3:])
    elif model_prompt_folder.startswith("gpt-oss-120b"):
        model_type, rest = "GPT-OSS-120B", "_".join(tokens[1:])
    else:
        model_type, rest = tokens[0], "_".join(tokens[1:])
    reasoning_level = None
    for level in ("low", "medium", "high"):
        if f"{level}_reasoning" in rest:
            reasoning_level = level
            rest = rest.replace(f"_{level}_reasoning", "")
    prompt_type = "simple_prompt" if "simple" in rest else "full_prompt" if "full" in rest else "unknown"
    return model_type, prompt_type, reasoning_level


def approx_tokens(text: str) -> float:
    return len(text) / 4.0


# ======================
# Loaders
# ======================

def load_result_files(base_dirs):
    rows = []
    for base_dir in base_dirs:
        model_type, prompt_type, reasoning_level = parse_model_prompt(base_dir)
        for file_path in glob.glob(os.path.join(base_dir, "bbq_*_results_merged.json")):
            data = json.load(open(file_path))
            for item in data.get("results", []):
                reasoning_text = item.get("model_reasoning") or ""
                rows.append({
                    "sample_id":        f"{item['category']}_{item['example_id']}",
                    "model":            model_type,
                    "prompt_type":      prompt_type,
                    "reasoning_level":  reasoning_level,
                    "reasoning_tokens": approx_tokens(reasoning_text),
                })
    df = pd.DataFrame(rows)
    print(f"Result files: {len(df)} rows")
    if df.empty:
        raise RuntimeError("No result files found. Check BASE_DIRS.")
    return df


def load_judge_files(base_dirs):
    rows = []
    for base_dir in base_dirs:
        model_type, prompt_type, reasoning_level = parse_model_prompt(base_dir)
        pattern = os.path.join(base_dir, "full_annotation", "*", "llm_eval_bbq_*.json")
        for file_path in glob.glob(pattern, recursive=True):
            data = json.load(open(file_path))
            for r in data.get("results", []):
                judge_out = r.get("judge_output")
                row = {
                    "sample_id":                f"{r['category']}_{r['example_id']}",
                    "category":                 r.get("category"),
                    "example_id":               r.get("example_id"),
                    "model":                    model_type,
                    "prompt_type":              prompt_type,
                    "reasoning_level":          reasoning_level,
                    "is_correct":               r.get("is_correct"),
                    "ambiguous":                r.get("ambiguous"),
                    "incorrect_and_stereotype": r.get("incorrect_and_stereotype"),
                    "judge_missing":            judge_out is None,
                }
                for label in JUDGE_LABELS:
                    row[label] = judge_out.get(label, 0) if judge_out else np.nan
                rows.append(row)
    df = pd.DataFrame(rows)
    print(f"Judge files:  {len(df)} rows")
    if df.empty:
        raise RuntimeError("No judge files found. Check BASE_DIRS.")
    return df


# ======================
# Load & merge
# ======================

print("Loading result files...")
df_results = load_result_files(BASE_DIRS)

print("Loading judge files...")
df_judge = load_judge_files(BASE_DIRS)

MERGE_KEYS = ["sample_id", "model", "prompt_type", "reasoning_level"]

df = df_judge.merge(
    df_results[MERGE_KEYS + ["reasoning_tokens"]],
    on=MERGE_KEYS,
    how="left",
)

print(f"After merge:              {len(df)} rows")
print(f"Missing reasoning length: {df['reasoning_tokens'].isna().sum()}")

df = df[~df["judge_missing"]].dropna(subset=JUDGE_LABELS + ["reasoning_tokens"])
print(f"After dropping missing:   {len(df)} rows")

if len(df) == 0:
    raise RuntimeError(
        "0 rows after merge.\n"
        f"  Judge sample_ids:  {df_judge['sample_id'].head().tolist()}\n"
        f"  Result sample_ids: {df_results['sample_id'].head().tolist()}"
    )

# Coercions
for col in JUDGE_LABELS:
    df[col] = df[col].astype(int)
df["incorrect_and_stereotype"] = df["incorrect_and_stereotype"].astype(int)
df["incorrect"]       = (~df["is_correct"].astype(bool)).astype(int)
df["ambiguous"]       = df["ambiguous"].astype("category")
df["prompt_type"]     = df["prompt_type"].astype("category")
df["model"]           = df["model"].astype("category")
df["category"]        = df["category"].astype("category")
df["reasoning_level"] = df["reasoning_level"].astype("category")

# Log-transform length — added as a column so build_formula() can reference it by name
df["log_reasoning_tokens"] = np.log1p(df["reasoning_tokens"])


# ======================
# Fit three models
# ======================

# M1: replicates main paper (no length control)
formula_m1 = build_formula("incorrect_and_stereotype", JUDGE_LABELS_NO_BIAS)

# M1b: main paper + length as simple covariate (new primary model)
formula_m1b = build_formula_with_length_ctrl("incorrect_and_stereotype", JUDGE_LABELS_NO_BIAS)

# M2: length replaces overthinking (no interactions for length)
formula_m2 = build_formula_with_length_ctrl(
    "incorrect_and_stereotype", 
    [l for l in JUDGE_LABELS_NO_BIAS if l != "overthinking"]
)

print("\nFitting M1 (overthinking, no length control — replicates paper)...")
m1 = fit_logit_mle(formula_m1, df)

print("Fitting M1b (overthinking + length as covariate)...")
m1b = fit_logit_mle(formula_m1b, df)

print("Fitting M2 (length as covariate, no overthinking)...")
m2 = fit_logit_mle(formula_m2, df)

print("All models fitted.")

# Save model summaries — m3 removed
for model, name in [
    (m1,  "m1_overthinking_only"),
    (m1b, "m1b_overthinking_length_ctrl"),
    (m2,  "m2_length_ctrl_no_overthinking"),
]:
    with open(OUT_DIR / f"logit_summary_{name}.txt", "w") as f:
        f.write(model.summary().as_text())


# ======================
# Comparison table
# ======================

def extract_coef(model, key):
    coef = model.params.get(key, np.nan)
    se   = model.bse.get(key, np.nan)
    return {
        "key":         key,
        "coef":        coef,
        "se":          se,
        "p_value":     model.pvalues.get(key, np.nan),
        "OR":          np.exp(coef),
        "OR_CI_lower": np.exp(coef - 1.96 * se),
        "OR_CI_upper": np.exp(coef + 1.96 * se),
    }

comp = pd.DataFrame([
    {"model": "M1: overthinking, no length ctrl",    **extract_coef(m1,  "overthinking")},
    {"model": "M1b: overthinking + length ctrl",     **extract_coef(m1b, "overthinking")},
    {"model": "M1b: length covariate",               **extract_coef(m1b, "log_reasoning_tokens")},
    {"model": "M2: no overthinking, length ctrl",    **extract_coef(m2,  "log_reasoning_tokens")},
])

# Also pull the key interaction terms from M1b
comp_interactions = pd.DataFrame([
    {"model": "M1b: overthinking × full_prompt",
     **extract_coef(m1b, "overthinking:C(prompt_type, Treatment(reference='simple_prompt'))[T.full_prompt]")},
    {"model": "M1b: length × full_prompt",
     **extract_coef(m1b, "log_reasoning_tokens:C(prompt_type, Treatment(reference='simple_prompt'))[T.full_prompt]")},
    {"model": "M1b: overthinking × ambiguous",
     **extract_coef(m1b, "overthinking:C(ambiguous)[T.True]")},
])

print("\n--- Focal predictor coefficients ---")
print(comp[["model", "key", "OR", "OR_CI_lower", "OR_CI_upper", "p_value"]].to_string(index=False))
comp.to_csv(OUT_DIR / "table_model_comparison.csv", index=False)

print("\n--- Key interaction terms (M1b) ---")
print(comp_interactions[["model", "key", "OR", "OR_CI_lower", "OR_CI_upper", "p_value"]].to_string(index=False))
comp_interactions.to_csv(OUT_DIR / "table_m1b_interactions.csv", index=False)



# ======================
# Overthinking–length relationship
# ======================

pb_r, pb_p = stats.pointbiserialr(df["overthinking"], df["reasoning_tokens"])
sp_r, sp_p = stats.spearmanr(df["overthinking"],      df["reasoning_tokens"])
print(f"\nPoint-biserial r = {pb_r:.3f}  p = {pb_p:.3e}")
print(f"Spearman r       = {sp_r:.3f}  p = {sp_p:.3e}")

pd.DataFrame({
    "metric":  ["point_biserial", "spearman"],
    "r":       [pb_r, sp_r],
    "p_value": [pb_p, sp_p],
}).to_csv(OUT_DIR / "table_overthinking_length_correlation.csv", index=False)

# Overthinking rate by length decile
df["length_decile"] = pd.qcut(df["reasoning_tokens"], q=10, labels=False, duplicates="drop")
decile_stats = (
    df.groupby("length_decile", observed=True)
    .agg(
        overthinking_rate=("overthinking", "mean"),
        median_tokens=("reasoning_tokens", "median"),
        n=("overthinking", "count"),
    )
    .reset_index()
)
decile_stats.to_csv(OUT_DIR / "table_overthinking_rate_by_length_decile.csv", index=False)

# Median length by overthinking × model
med_table = df.groupby(["model", "overthinking"])["reasoning_tokens"].median().unstack()
print("\n--- Median tokens by overthinking × model ---")
print(med_table.round(1))
med_table.to_csv(OUT_DIR / "table_median_length_by_overthinking_model.csv")


# ======================
# Plots
# ======================

# --- Plot 1: Forest plot — M1, M2, M3 focal predictors ---
fig, ax = plt.subplots(figsize=(10, 4.5))

plot_rows = [
    ("M1: overthinking, no length ctrl",  "overthinking",         m1,  "#4E79A7", 4),
    ("M1b: overthinking + length ctrl",   "overthinking",         m1b, "#4E79A7", 3),
    ("M1b: length covariate",             "log_reasoning_tokens", m1b, "#F28E2B", 2),
    ("M2: length only (no overthinking)", "log_reasoning_tokens", m2,  "#F28E2B", 1),
    # Key interactions from M1b
    ("M1b: overthinking × full_prompt",
     "overthinking:C(prompt_type, Treatment(reference='simple_prompt'))[T.full_prompt]",
     m1b, "#59A14F", 0),
]

for label, key, model, color, y in plot_rows:
    row = extract_coef(model, key)
    ax.plot([row["OR_CI_lower"], row["OR_CI_upper"]], [y, y],
            color=color, lw=3, solid_capstyle="round")
    ax.scatter(row["OR"], y, color=color, s=100, zorder=5)
    ax.text(
        row["OR_CI_upper"] * 1.02, y,
        f"OR={row['OR']:.2f} [{row['OR_CI_lower']:.2f}–{row['OR_CI_upper']:.2f}]  p={row['p_value']:.3f}",
        va="center", fontsize=8,
    )

ax.axvline(1.0, color="black", lw=0.8, linestyle="--")
ax.set_xscale("log")
ax.set_yticks([y for _, _, _, _, y in plot_rows])
ax.set_yticklabels([label for label, _, _, _, _ in plot_rows], fontsize=9)
ax.set_xlabel("Odds Ratio (log scale)")
ax.set_title(
    "Overthinking vs Length as Predictors of Stereotyped Errors\n"
    "M1: overthinking only  |  M2: length only  |  M3: both"
)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_model_comparison_forest.pdf")
plt.close()
print("Saved: fig_model_comparison_forest.pdf")


# --- Plot 2: Overthinking rate by length decile ---
smoothed = lowess(decile_stats["overthinking_rate"], decile_stats["median_tokens"], frac=0.6)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(
    range(len(decile_stats)), decile_stats["overthinking_rate"],
    color="steelblue", alpha=0.6, label="Overthinking rate",
)
ax2 = ax.twinx()
ax2.plot(range(len(decile_stats)), decile_stats["median_tokens"],
         color="grey", lw=1.5, linestyle="--", label="Median tokens")
ax2.set_ylabel("Median reasoning tokens")
ax.set_xticks(range(len(decile_stats)))
ax.set_xticklabels([f"{int(v)}" for v in decile_stats["median_tokens"]], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Reasoning length decile (median tokens)")
ax.set_ylabel("Overthinking label rate")
ax.set_title(
    f"Overthinking Rate by Reasoning Length Decile\n"
    f"point-biserial r = {pb_r:.3f}  |  Spearman r = {sp_r:.3f}"
)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_overthinking_rate_by_length_decile.pdf")
plt.close()
print("Saved: fig_overthinking_rate_by_length_decile.pdf")


# --- Plot 3: Violin — length by overthinking, faceted by model ---
g = sns.FacetGrid(df, col="model", col_wrap=3, height=3.5, sharey=False)
g.map_dataframe(sns.violinplot, x="overthinking", y="reasoning_tokens",
                palette="Set2", inner="box")
g.set_axis_labels("Overthinking (0/1)", "Reasoning tokens (log scale)")
g.set_titles("{col_name}")
for ax in g.axes.flat:
    ax.set_yscale("log")
g.figure.suptitle(
    f"Reasoning Length by Overthinking Label per Model\n"
    f"point-biserial r = {pb_r:.3f}",
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_length_by_overthinking_per_model.pdf")
plt.close()
print("Saved: fig_length_by_overthinking_per_model.pdf")


# ======================
# Descriptive stats: overthinking × prompt_type × ambiguous
# These are the exact cells the model interaction terms are capturing
# ======================

print("\n--- Mean reasoning tokens by overthinking  ---")
mean_len = (
    df.groupby(["overthinking"], observed=True)["reasoning_tokens"]
    .agg(mean_tokens="mean", median_tokens="median", n="count")
    .reset_index()
)
print(mean_len.to_string(index=False))
mean_len.to_csv(OUT_DIR / "table_mean_length_overthinking.csv", index=False)

print("\n--- Mean reasoning tokens by prompt_type  ---")
mean_len = (
    df.groupby(["prompt_type"], observed=True)["reasoning_tokens"]
    .agg(mean_tokens="mean", median_tokens="median", n="count")
    .reset_index()
)
print(mean_len.to_string(index=False))
mean_len.to_csv(OUT_DIR / "table_mean_length_prompt.csv", index=False)

print("\n--- Mean reasoning tokens by ambigious  ---")
mean_len = (
    df.groupby(["ambiguous"], observed=True)["reasoning_tokens"]
    .agg(mean_tokens="mean", median_tokens="median", n="count")
    .reset_index()
)
print(mean_len.to_string(index=False))
mean_len.to_csv(OUT_DIR / "table_mean_length_ambiguous.csv", index=False)

print("\n--- Mean reasoning tokens by prompt_type x overthinking  ---")
mean_len = (
    df.groupby(["prompt_type", "overthinking"], observed=True)["reasoning_tokens"]
    .agg(mean_tokens="mean", median_tokens="median", n="count")
    .reset_index()
)
print(mean_len.to_string(index=False))
mean_len.to_csv(OUT_DIR / "table_mean_length_prompt_x_overthinking.csv", index=False)

print("\n--- Mean reasoning tokens by prompt_type × ambiguous ---")
mean_len = (
    df.groupby(["prompt_type", "ambiguous"], observed=True)["reasoning_tokens"]
    .agg(mean_tokens="mean", median_tokens="median", n="count")
    .reset_index()
)
print(mean_len.to_string(index=False))
mean_len.to_csv(OUT_DIR / "table_mean_length_prompt_x_ambiguous.csv", index=False)
 
print("\n--- Mean reasoning tokens by prompt_type × ambiguous × overthinking ---")
mean_len_ot = (
    df.groupby(["prompt_type", "ambiguous", "overthinking"], observed=True)["reasoning_tokens"]
    .agg(mean_tokens="mean", median_tokens="median", n="count")
    .reset_index()
)
print(mean_len_ot.to_string(index=False))
mean_len_ot.to_csv(OUT_DIR / "table_mean_length_prompt_x_ambiguous_x_overthinking.csv", index=False)

 
print("\n--- Overthinking rate by prompt_type × ambiguous ---")
ot_rate = (
    df.groupby(["prompt_type", "ambiguous"], observed=True)["overthinking"]
    .agg(overthinking_rate="mean", n="count")
    .reset_index()
)
print(ot_rate.to_string(index=False))
ot_rate.to_csv(OUT_DIR / "table_overthinking_rate_prompt_x_ambiguous.csv", index=False)
 
print("\n--- Median reasoning tokens by prompt_type × ambiguous × overthinking ---")
len_cross = (
    df.groupby(["prompt_type", "ambiguous", "overthinking"], observed=True)["reasoning_tokens"]
    .agg(median_tokens="median", n="count")
    .reset_index()
)
print(len_cross.to_string(index=False))
len_cross.to_csv(OUT_DIR / "table_length_prompt_x_ambiguous_x_overthinking.csv", index=False)
 
print("\n--- Stereotyped error rate by prompt_type × ambiguous × overthinking ---")
stereo_cross = (
    df.groupby(["prompt_type", "ambiguous", "overthinking"], observed=True)["incorrect_and_stereotype"]
    .agg(stereo_rate="mean", n="count")
    .reset_index()
)
print(stereo_cross.to_string(index=False))
stereo_cross.to_csv(OUT_DIR / "table_stereo_rate_prompt_x_ambiguous_x_overthinking.csv", index=False)
 
# Combined summary table: all three stats in one view
combined = len_cross.merge(stereo_cross, on=["prompt_type", "ambiguous", "overthinking"])
combined = combined.rename(columns={"n_x": "n"}).drop(columns=["n_y"])
combined = combined.merge(
    ot_rate[["prompt_type", "ambiguous", "overthinking_rate"]],
    on=["prompt_type", "ambiguous"],
    how="left",
)
print("\n--- Combined: length + stereo rate by prompt_type × ambiguous × overthinking ---")
print(combined.to_string(index=False))
combined.to_csv(OUT_DIR / "table_combined_prompt_x_ambiguous_x_overthinking.csv", index=False)


# ======================
# Summary
# ======================

or_m1    = comp.loc[(comp["model"] == "M1: overthinking, no length ctrl")  & (comp["key"] == "overthinking"),         "OR"].values[0]
or_m1b   = comp.loc[(comp["model"] == "M1b: overthinking + length ctrl")   & (comp["key"] == "overthinking"),         "OR"].values[0]
or_m1b_l = comp.loc[(comp["model"] == "M1b: length covariate")             & (comp["key"] == "log_reasoning_tokens"), "OR"].values[0]
or_m2    = comp.loc[(comp["model"] == "M2: no overthinking, length ctrl")  & (comp["key"] == "log_reasoning_tokens"), "OR"].values[0]

summary = {
    "n_total":                              int(len(df)),
    "pointbiserial_r_overthinking_length":  float(pb_r),
    "spearman_r_overthinking_length":       float(sp_r),
    "OR_m1_overthinking":                   float(or_m1),
    "OR_m1b_overthinking_with_length_ctrl": float(or_m1b),
    "OR_m1b_length_covariate":              float(or_m1b_l),
    "OR_m2_length_only":                    float(or_m2),
    "OR_pct_change_overthinking_m1_to_m1b": float((or_m1b - or_m1) / or_m1 * 100),
}

with open(OUT_DIR / "summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Summary ===")
for k, v in summary.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

print(f"\nAll outputs saved to: {OUT_DIR}")