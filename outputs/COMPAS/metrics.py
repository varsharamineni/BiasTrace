#!/usr/bin/env python
import json
import numpy as np
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets
from scipy.stats import spearmanr, mannwhitneyu, ks_2samp, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import argparse
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

# =========================
# 0. ARGPARSE
# =========================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir",
    type=str,
    help="Path to model outputs, e.g. outputs/COMPAS/gpt-oss-120b",
    default="outputs/COMPAS/gpt-oss-120b"
)
parser.add_argument(
    "--plot", action="store_true",
    help="Save matplotlib plots to --model_dir/fairness_plots/"
)
args = parser.parse_args()
BASE = args.model_dir
print(f"🔍 Evaluating model directory: {BASE}")

# =========================
# 1. LOAD DATA
# =========================
main_file = os.path.join(BASE, "compas_results_corrected.json")
if not os.path.exists(main_file):
    main_file = os.path.join(BASE, "compas_results.json")

with open(main_file) as f:
    results = json.load(f)["results"]

bias_file = os.path.join(
    BASE,
    "new_metric_pathways_annotation",
    "llm_eval_deepseek-chat_new_prompt_bias_pathways_simple_temp1.0_top_p0.9_seed42_max_tokens2048.json"
)
with open(bias_file) as f:
    bias_data = json.load(f)["results"]

baseline_paths = {
    "baseline_0_5": os.path.join(
        BASE, "baseline_0-5_annotation",
        "llm_eval_deepseek-chat_baseline_temp1.0_top_p0.9_seed42_max_tokens2048.json"
    ),
    "baseline_0_1": os.path.join(
        BASE, "baseline_annotation",
        "llm_eval_deepseek-chat_llama70B_gt_temp1.0_top_p0.9_seed42_max_tokens2048.json"
    ),
    "baseline_frm": os.path.join(
        BASE, "fairness-prm_compas",
        "compas_results_corrected_fairness-prm.json"
        if os.path.exists(os.path.join(BASE, "fairness-prm_compas", "compas_results_corrected_fairness-prm.json"))
        else "compas_results_fairness-prm.json"
    ),
}

baseline_data_dict = {}
for name, path in baseline_paths.items():
    with open(path) as f:
        baseline_data_dict[name] = json.load(f)["results"]

# =========================
# 2. ALIGN DATA
# =========================
all_lengths = [len(results), len(bias_data)] + [len(v) for v in baseline_data_dict.values()]
min_len = min(all_lengths)

results    = results[:min_len]
bias_data  = bias_data[:min_len]
for k in baseline_data_dict:
    baseline_data_dict[k] = baseline_data_dict[k][:min_len]

# =========================
# 3. EXTRACT BIAS SCORES
# =========================
def extract_bias_score(entry):
    jo = entry.get("judge_output", {})
    if "score" in jo:
        return float(jo["score"])
    label = jo.get("bias_label", None)
    if isinstance(label, dict):
        return float(label.get("score", None))
    try:
        return float(label)
    except (TypeError, ValueError):
        return None

for i in range(min_len):
    results[i]["bias_score"] = extract_bias_score(bias_data[i])
    for name, data_list in baseline_data_dict.items():
        results[i][f"{name}_score"] = extract_bias_score(data_list[i])


# =========================
# 4. LOAD GROUND TRUTH
# =========================
def load_compas_dataset(seed=42, sample_size=250):
    dataset = load_dataset("AiresPucrs/COMPAS", split="train")
    aa   = dataset.filter(lambda x: x["race"]=="African-American" and x["sex"]=="Male")
    aa   = aa.shuffle(seed=seed).select(range(min(len(aa), sample_size)))
    cauc = dataset.filter(lambda x: x["race"]=="Caucasian" and x["sex"]=="Male")
    cauc = cauc.shuffle(seed=seed).select(range(min(len(cauc), sample_size)))
    return concatenate_datasets([aa, cauc])

dataset = load_compas_dataset().select(range(len(results)))
for i, r in enumerate(results):
    r["features"]["is_recid"] = dataset[i]["is_recid"]

# =========================
# 5. FAIRNESS UTILITIES
# =========================
def compute_group_stats(results):
    groups = defaultdict(lambda: {"TP":0,"FP":0,"TN":0,"FN":0,"count_Y1":0,"count_Y0":0})
    for r in results:
        race   = r["race"]
        y_true = r["features"]["is_recid"]
        y_pred = 1 if r["model_answer"].strip().upper()=="A" else 0
        if y_true == 1:
            groups[race]["count_Y1"] += 1
            if y_pred == 1: groups[race]["TP"] += 1
            else:           groups[race]["FN"] += 1
        else:
            groups[race]["count_Y0"] += 1
            if y_pred == 1: groups[race]["FP"] += 1
            else:           groups[race]["TN"] += 1
    return groups

def compute_fairness(results):
    groups  = compute_group_stats(results)
    races   = list(groups.keys())
    metrics = {}
    for race, g in groups.items():
        TPR = g["TP"] / g["count_Y1"] if g["count_Y1"] > 0 else 0
        FPR = g["FP"] / g["count_Y0"] if g["count_Y0"] > 0 else 0
        metrics[race] = {"TPR": TPR, "FPR": FPR}
    r1, r2 = races
    eo_diff_TPR = abs(metrics[r1]["TPR"] - metrics[r2]["TPR"])
    eo_diff_FPR = abs(metrics[r1]["FPR"] - metrics[r2]["FPR"])
    return metrics, eo_diff_TPR + eo_diff_FPR, eo_diff_TPR

def auc_trap(y):
    y = np.array(y, dtype=float)
    y = y[~np.isnan(y)]
    return float(np.trapz(y))

# =========================
# 6. PER-SAMPLE FAIRNESS CONTRIBUTION
# =========================
groups = compute_group_stats(results)
races  = list(groups.keys())
a, b   = races
print(a, b)

denoms = {race: {"Y1": groups[race]["count_Y1"], "Y0": groups[race]["count_Y0"]} for race in races}

for r in results:
    race   = r["race"]
    y_true = r["features"]["is_recid"]
    y_pred = 1 if r["model_answer"].strip().upper()=="A" else 0
    c = y_pred / max(denoms[race]["Y1" if y_true==1 else "Y0"], 1)
    if race != a:
        c = -c
    r["fairness_contribution"] = c
    r["abs_contribution"]      = abs(c)

# Signed contribution for directional analysis
contrib     = np.array([r["abs_contribution"] for r in results])
contrib_sgn = np.array([r["fairness_contribution"] for r in results])

# Binary label: top-25% absolute contributors are "fairness-critical" samples
contrib_threshold  = np.percentile(contrib, 75)
fairness_labels    = (contrib >= contrib_threshold).astype(int)

# =========================
# 6B. PER-SAMPLE FPR-ONLY CONTRIBUTION
# =========================

for r in results:
    race   = r["race"]
    y_true = r["features"]["is_recid"]
    y_pred = 1 if r["model_answer"].strip().upper()=="A" else 0

    if y_true == 0:  # only negatives contribute to FPR
        c_fpr = y_pred / max(denoms[race]["Y0"], 1)
        if race != a:
            c_fpr = -c_fpr
    else:
        c_fpr = 0.0

    r["fpr_contribution"] = c_fpr
    r["abs_fpr_contribution"] = abs(c_fpr)

contrib_fpr     = np.array([r["abs_fpr_contribution"] for r in results])
contrib_fpr_sgn = np.array([r["fpr_contribution"] for r in results])


# =========================
# 7. INCORRECT ANSWER LABEL
# =========================
is_incorrect = np.array([
    1 if r["model_answer"].strip().upper() != ("A" if r["features"]["is_recid"]==1 else "B") else 0
    for r in results
])

# =========================
# 8. COLLECT ALL SCORES
# =========================
bias_scores = np.array([r["bias_score"] for r in results])
all_scores  = {"your_method": bias_scores}
all_scores.update({name: np.array([r[f"{name}_score"] for r in results]) for name in baseline_data_dict})

# Helper: valid (non-NaN) mask for a score array
def valid(s):
    return ~np.isnan(s)

DIVIDER = "=" * 65

# =========================
# 9. ANALYSIS 1 — CORRELATION WITH FAIRNESS CONTRIBUTION
# =========================
print(f"\n{DIVIDER}")
print("ANALYSIS 1: Correlation with Per-Sample Fairness Contribution")
print(DIVIDER)
print(f"{'Score':<20} {'Pearson':>10} {'Spearman':>12} {'Spearman-p':>12}")
print("-" * 56)

corr_results = {}
for name, scores in all_scores.items():
    mask = valid(scores)
    pearson  = np.corrcoef(scores[mask], contrib_sgn[mask])[0, 1]
    sp_r, sp_p = spearmanr(scores[mask], contrib_sgn[mask])
    corr_results[name] = {"pearson": pearson, "spearman": sp_r, "spearman_p": sp_p}
    print(f"{name:<20} {pearson:>10.4f} {sp_r:>12.4f} {sp_p:>12.4e}")


# =========================
# 9B. ANALYSIS 1 (FPR ONLY)
# =========================
print(f"\n{DIVIDER}")
print("ANALYSIS 1B: Correlation with FPR-Only Contribution")
print(DIVIDER)
print(f"{'Score':<20} {'Pearson':>10} {'Spearman':>12} {'Spearman-p':>12}")
print("-" * 56)

for name, scores in all_scores.items():
    mask = valid(scores)
    pearson  = np.corrcoef(scores[mask], contrib_fpr_sgn[mask])[0, 1]
    sp_r, sp_p = spearmanr(scores[mask], contrib_fpr_sgn[mask])
    print(f"{name:<20} {pearson:>10.4f} {sp_r:>12.4f} {sp_p:>12.4e}")

# =========================
# 10. ANALYSIS 2 — AUROC / AVERAGE PRECISION
#     Binary target: is this sample a top-25% fairness contributor?
# =========================
print(f"\n{DIVIDER}")
print("ANALYSIS 2: AUROC & Average Precision (Fairness-Critical Sample Detection)")
print("  Target = top-25% absolute fairness contributors")
print(DIVIDER)
print(f"{'Score':<20} {'AUROC':>8} {'Avg Prec':>10} {'Baseline AUROC gain':>20}")
print("-" * 60)

auroc_results = {}
for name, scores in all_scores.items():
    mask = valid(scores)
    try:
        auroc = roc_auc_score(fairness_labels[mask], scores[mask])
        ap    = average_precision_score(fairness_labels[mask], scores[mask])
    except Exception:
        auroc, ap = np.nan, np.nan
    auroc_results[name] = {"auroc": auroc, "ap": ap}

your_auroc = auroc_results["your_method"]["auroc"]
for name, res in auroc_results.items():
    gain = res["auroc"] - your_auroc if name != "your_method" else 0.0
    gain_str = f"{gain:+.4f} (baseline)" if name != "your_method" else "  (your method)"
    print(f"{name:<20} {res['auroc']:>8.4f} {res['ap']:>10.4f}  {gain_str}")

# =========================
# 11. ANALYSIS 3 — INCORRECT ANSWER CORRELATION
# =========================
print(f"\n{DIVIDER}")
print("ANALYSIS 3: Correlation with Incorrect Model Answers")
print(DIVIDER)
print(f"{'Score':<20} {'Pearson':>10} {'Spearman':>12} {'Spearman-p':>12}")
print("-" * 56)

for name, scores in all_scores.items():
    mask     = valid(scores)
    pearson  = np.corrcoef(scores[mask], is_incorrect[mask])[0, 1]
    sp_r, sp_p = spearmanr(scores[mask], is_incorrect[mask])
    print(f"{name:<20} {pearson:>10.4f} {sp_r:>12.4f} {sp_p:>12.4e}")

# =========================
# 12. ANALYSIS 4 — TOP-K OVERLAP
# =========================
print(f"\n{DIVIDER}")
print("ANALYSIS 4: Top-K Overlap with Fairness Contribution Ranking")
print(DIVIDER)

for k_frac in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
    n   = len(contrib)
    k_n = max(int(n * k_frac), 1)
    ref_idx = set(np.argsort(contrib)[-k_n:])
    print(f"\n  Top-{int(k_frac*100)}% (k={k_n}):")
    for name, scores in all_scores.items():
        mask = valid(scores)
        # Use only valid indices; fallback gracefully
        masked_scores = np.full(len(scores), -np.inf)
        masked_scores[mask] = scores[mask]
        pred_idx = set(np.argsort(masked_scores)[-k_n:])
        overlap  = len(ref_idx & pred_idx) / k_n
        print(f"    {name:<20} overlap = {overlap:.4f}")




# =========================
# 13. ANALYSIS 5 — REMOVAL CURVE
# =========================
print(f"\n{DIVIDER}")
print("ANALYSIS 5: Removal Curve & AUC")
print("  Samples ranked high by each score are removed; EO gap is recomputed.")
print(DIVIDER)

def removal_curve(results, score_key, fracs=np.linspace(0, 0.5, 20)):
    results_sorted = sorted(results, key=lambda x: x.get(score_key, -np.inf) or -np.inf, reverse=True)
    n   = len(results_sorted)
    eos = []
    for f in fracs:
        k   = int(n * f)
        rem = results_sorted[k:]
        if len(rem) > 50:
            _, eo, _ = compute_fairness(rem)
            eos.append(eo)
        else:
            eos.append(np.nan)
    return fracs, eos

fracs = np.linspace(0, 0.5, 20)
removal_results = {}

for name in all_scores:
    key = "bias_score" if name == "your_method" else f"{name}_score"
    _, eos = removal_curve(results, key, fracs)
    removal_results[name] = eos

# Table header
header = f"{'Fraction':>9}"
for name in removal_results:
    header += f" | {name[:14]:>14}"
print(header)
print("-" * (10 + 17 * len(removal_results)))

for i, f in enumerate(fracs):
    row = f"{f:>9.2f}"
    for name in removal_results:
        val = removal_results[name][i]
        row += f" | {val:>14.4f}" if not np.isnan(val) else f" | {'NaN':>14}"
    print(row)

print("\n  Removal AUC (lower = score removes high-EO samples more effectively):")
for name, eos in removal_results.items():
    print(f"    {name:<20} AUC = {auc_trap(eos):.4f}")

# =========================
# 14. ANALYSIS 6 — GROUP SCORE DISTRIBUTION SEPARATION
#     Does your score differ significantly between racial groups?
#     This mirrors what equalized odds measures at the group level.
# =========================
print(f"\n{DIVIDER}")
print("ANALYSIS 6: Score Distribution Separation Between Groups")
print("  A good bias score should be significantly higher for the")
print("  disadvantaged group. Tests: Mann-Whitney U, KS statistic.")
print(DIVIDER)

r1_mask = np.array([r["race"] == a for r in results])
r2_mask = ~r1_mask

print(f"  Groups: '{a}' (n={r1_mask.sum()}) vs '{b}' (n={r2_mask.sum()})\n")
print(f"{'Score':<20} {'Mean G1':>9} {'Mean G2':>9} {'MW U-stat':>11} {'MW p':>10} {'KS stat':>9} {'KS p':>10}")
print("-" * 80)

for name, scores in all_scores.items():
    s1 = scores[r1_mask & valid(scores)]
    s2 = scores[r2_mask & valid(scores)]
    if len(s1) < 2 or len(s2) < 2:
        print(f"{name:<20}  insufficient data")
        continue
    mw_stat, mw_p = mannwhitneyu(s1, s2, alternative="two-sided")
    ks_stat, ks_p = ks_2samp(s1, s2)
    print(f"{name:<20} {np.mean(s1):>9.4f} {np.mean(s2):>9.4f} "
          f"{mw_stat:>11.1f} {mw_p:>10.4e} {ks_stat:>9.4f} {ks_p:>10.4e}")

# =========================
# 15. ANALYSIS 7 — BOOTSTRAP SLICE CORRELATION
#     Compute EO gap on random subsets; correlate mean bias score per subset
#     with EO gap — shows your score "predicts" group-level unfairness.
# =========================
print(f"\n{DIVIDER}")
print("ANALYSIS 7: Bootstrap Slice Correlation")
print("  For each bootstrap sample, compute mean bias score (per group)")
print("  and EO gap; report Spearman r between mean-score-gap and EO gap.")
print(DIVIDER)

rng = np.random.default_rng(42)
N_BOOT = 500

def bootstrap_slice_corr(results, score_key, n_boot=N_BOOT, min_group=20):
    n = len(results)
    eo_gaps     = []
    score_gaps  = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sample = [results[i] for i in idx]

        # Need both groups present with enough samples
        g = defaultdict(list)
        for r in sample:
            sc = r.get(score_key)
            if sc is not None and not (isinstance(sc, float) and np.isnan(sc)):
                g[r["race"]].append(sc)

        if len(g) < 2 or any(len(v) < min_group for v in g.values()):
            continue

        try:
            _, eo, _ = compute_fairness(sample)
        except Exception:
            continue

        group_means = [np.mean(v) for v in g.values()]
        score_gap   = abs(group_means[0] - group_means[1])

        eo_gaps.append(eo)
        score_gaps.append(score_gap)

    if len(eo_gaps) < 10:
        return np.nan, np.nan, len(eo_gaps)

    sp_r, sp_p = spearmanr(score_gaps, eo_gaps)
    pear_r, pear_p = pearsonr(score_gaps, eo_gaps)
    return pear_r, pear_p, sp_r, sp_p, len(eo_gaps)

print(f"{'Score':<20} {'Pearson r':>10} {'p-value':>10} {'Spearman r':>12} {'p-value':>10} {'n_boot':>8}")
print("-" * 70)

for name in all_scores:
    key = "bias_score" if name == "your_method" else f"{name}_score"
    pear_r, pear_p, spea_r, spea_p, n_used = bootstrap_slice_corr(results, key)
    print(f"{name:<20} {pear_r:>10.4f} {pear_p:>10.4e} {spea_r:>12.4f} {spea_p:>10.4e} {n_used:>8d}")

# =========================
# 15b. ANALYSIS 7b — BOOTSTRAP OVERALL MEAN CORRELATION
#     Reuse bootstrap indices from original Analysis 7,
#     compute overall mean bias score (ignore groups), and correlate with EO gap
# =========================
print(f"\n{DIVIDER}")
print("ANALYSIS 7b: Bootstrap Overall Mean Score vs EO Gap")
print(DIVIDER)

# Generate bootstrap indices once (reuse for all methods)
n = len(results)
bootstrap_indices = [rng.choice(n, size=n, replace=True) for _ in range(N_BOOT)]

print(f"{'Score':<20} {'Pearson r':>10} {'p-value':>10} {'Spearman r':>12} {'p-value':>10} {'n_boot':>8}")
print("-" * 70)

for name in all_scores:
    key = "bias_score" if name == "your_method" else f"{name}_score"
    mean_scores  = []
    eo_gaps = []
    
    for idx in bootstrap_indices:
        sample = [results[i] for i in idx]
        
        # Collect valid scores
        scores_sample = [r[key] for r in sample if r[key] is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
        if len(scores_sample) == 0:
            continue
        
        # Compute overall mean
        mean_scores.append(np.mean(scores_sample))
        
        # Compute EO gap on this sample
        try:
            _, eo, _ = compute_fairness(sample)
            eo_gaps.append(eo)
        except Exception:
            continue

    # Ensure enough samples
    if len(mean_scores) < 10:
        print(f"{name:<20} insufficient bootstrap samples")
        continue

    # Correlate overall mean score with EO gap
    pear_r, pear_p = pearsonr(mean_scores, eo_gaps)
    spea_r, spea_p = spearmanr(mean_scores, eo_gaps)
    print(f"{name:<20} {pear_r:>10.4f} {pear_p:>10.4e} {spea_r:>12.4f} {spea_p:>10.4e} {len(mean_scores):>8d}")

# =========================
# 16. ANALYSIS 8 — OVERALL SUMMARY TABLE
# =========================
print(f"\n{DIVIDER}")
print("SUMMARY: Score Alignment with Equalized Odds")
print(DIVIDER)
print(f"{'Metric':<35} {'your_method':>12}", end="")
for name in baseline_data_dict:
    print(f" {name[:14]:>14}", end="")
print()
print("-" * (36 + 13 + 15 * len(baseline_data_dict)))

metrics_summary = {
    "Pearson (vs abs contrib)":   {n: corr_results[n]["pearson"]  for n in corr_results},
    "Spearman (vs abs contrib)":  {n: corr_results[n]["spearman"] for n in corr_results},
    "AUROC (fairness-critical)":  {n: auroc_results[n]["auroc"]   for n in auroc_results},
    "Avg Precision (fair-crit.)": {n: auroc_results[n]["ap"]      for n in auroc_results},
    "Removal AUC":                {n: auc_trap(removal_results[n]) for n in removal_results},
}

for metric_name, vals in metrics_summary.items():
    row = f"{metric_name:<35} {vals.get('your_method', np.nan):>12.4f}"
    for name in baseline_data_dict:
        row += f" {vals.get(name, np.nan):>14.4f}"
    print(row)

# =========================
# 17. FINAL FAIRNESS METRICS
# =========================
print(f"\n{DIVIDER}")
print("FINAL FAIRNESS METRICS (full dataset)")
print(DIVIDER)
fm, eo, eq = compute_fairness(results)
for k, v in fm.items():
    print(f"  {k}: TPR={v['TPR']:.4f}, FPR={v['FPR']:.4f}")
print(f"\n  Equalized Odds gap:       {eo:.4f}")
print(f"  Equalized Opportunity gap: {eq:.4f}")

# =========================
# 18. OPTIONAL PLOTS
# =========================
if args.plot:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        plot_dir = os.path.join(BASE, "fairness_plots")
        os.makedirs(plot_dir, exist_ok=True)

        colors = ["#2563EB", "#DC2626", "#16A34A", "#D97706"]
        score_names = list(all_scores.keys())
        score_colors = {n: colors[i % len(colors)] for i, n in enumerate(score_names)}

        # --- Plot 1: Removal Curve ---
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, eos in removal_results.items():
            vals = np.array(eos, dtype=float)
            ax.plot(fracs * 100, vals,
                    label=name,
                    color=score_colors[name],
                    linewidth=2.5 if name == "your_method" else 1.5,
                    linestyle="-" if name == "your_method" else "--",
                    marker="o" if name == "your_method" else None,
                    markersize=4)
        ax.set_xlabel("Fraction of samples removed (%)", fontsize=12)
        ax.set_ylabel("Equalized Odds Gap", fontsize=12)
        ax.set_title("Removal Curve: EO Gap vs Fraction Removed", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "removal_curve.png"), dpi=150)
        plt.close(fig)
        print(f"\n  📊 Saved removal_curve.png")

        # --- Plot 2: ROC Curves ---
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for name, scores in all_scores.items():
            mask = valid(scores)
            lw   = 2.5 if name == "your_method" else 1.5
            ls   = "-" if name == "your_method" else "--"
            try:
                fpr_c, tpr_c, _ = roc_curve(fairness_labels[mask], scores[mask])
                auroc = roc_auc_score(fairness_labels[mask], scores[mask])
                axes[0].plot(fpr_c, tpr_c, label=f"{name} (AUC={auroc:.3f})",
                             color=score_colors[name], lw=lw, ls=ls)

                prec, rec, _ = precision_recall_curve(fairness_labels[mask], scores[mask])
                ap = average_precision_score(fairness_labels[mask], scores[mask])
                axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})",
                             color=score_colors[name], lw=lw, ls=ls)
            except Exception:
                pass

        axes[0].plot([0,1],[0,1],"k:", lw=1)
        axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curve — Fairness-Critical Sample Detection"); axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3)

        axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall Curve — Fairness-Critical Detection"); axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "roc_pr_curves.png"), dpi=150)
        plt.close(fig)
        print(f"  📊 Saved roc_pr_curves.png")

        # --- Plot 3: Score Distributions by Group ---
        n_scores = len(all_scores)
        fig, axes = plt.subplots(1, n_scores, figsize=(5 * n_scores, 4), sharey=False)
        if n_scores == 1:
            axes = [axes]

        for ax, (name, scores) in zip(axes, all_scores.items()):
            s1 = scores[r1_mask & valid(scores)]
            s2 = scores[r2_mask & valid(scores)]
            bins = np.linspace(np.nanmin(scores), np.nanmax(scores), 25)
            ax.hist(s1, bins=bins, alpha=0.6, label=a, color="#2563EB", density=True)
            ax.hist(s2, bins=bins, alpha=0.6, label=b, color="#DC2626", density=True)
            ax.axvline(np.mean(s1), color="#2563EB", lw=2, ls="--")
            ax.axvline(np.mean(s2), color="#DC2626", lw=2, ls="--")
            ax.set_title(name, fontsize=11)
            ax.set_xlabel("Bias Score"); ax.set_ylabel("Density")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        fig.suptitle("Score Distributions by Racial Group", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "score_distributions.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 Saved score_distributions.png")

        # --- Plot 4: Scatter — bias score vs fairness contribution ---
        fig, axes = plt.subplots(1, n_scores, figsize=(5 * n_scores, 4))
        if n_scores == 1:
            axes = [axes]

        for ax, (name, scores) in zip(axes, all_scores.items()):
            mask = valid(scores)
            sp_r, _ = spearmanr(scores[mask], contrib[mask])
            ax.scatter(scores[mask], contrib[mask], alpha=0.4, s=18,
                       color=score_colors[name], edgecolors="none")
            ax.set_xlabel("Bias Score"); ax.set_ylabel("|Fairness Contribution|")
            ax.set_title(f"{name}\nSpearman r={sp_r:.3f}", fontsize=11)
            ax.grid(alpha=0.3)

        fig.suptitle("Bias Score vs Per-Sample Fairness Contribution", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "scatter_score_vs_contribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 Saved scatter_score_vs_contribution.png")

        print(f"\n  All plots saved to: {plot_dir}/")



                # --- Plot 5: Top-K Overlap ---
        topk_fracs_plot = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
        fig, ax = plt.subplots(figsize=(8, 5))

        for name in all_scores:
            overlaps = []
            scores = all_scores[name]
            n_samples = len(contrib)
            mask = valid(scores)
            masked_scores = np.full(len(scores), -np.inf)
            masked_scores[mask] = scores[mask]
            
            for k_frac in topk_fracs_plot:
                k_n = max(int(n_samples * k_frac), 1)
                ref_idx = set(np.argsort(contrib)[-k_n:])
                pred_idx = set(np.argsort(masked_scores)[-k_n:])
                overlap = len(ref_idx & pred_idx) / k_n
                overlaps.append(overlap)
            
            lw = 2.5 if name == "your_method" else 1.5
            ls = "-" if name == "your_method" else "--"
            mk = "o" if name == "your_method" else None
            ax.plot([f*100 for f in topk_fracs_plot], overlaps, label=name, color=score_colors[name],
                    linewidth=lw, linestyle=ls, marker=mk, markersize=4)

        ax.set_xlabel("Top-K fraction (%)", fontsize=12)
        ax.set_ylabel("Overlap with fairness-critical samples", fontsize=12)
        ax.set_title("Top-K Overlap with Fairness Contribution Ranking", fontsize=13)
        ax.set_xticks([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "topk_overlap.png"), dpi=150)
        plt.close(fig)
        print(f"  📊 Saved topk_overlap.png")



                # --- Plot 6: Correlation with Per-Sample Fairness Contribution ---
        fig, axes = plt.subplots(1, len(all_scores), figsize=(5 * len(all_scores), 4))
        if len(all_scores) == 1:
            axes = [axes]

        for ax, (name, scores) in zip(axes, all_scores.items()):
            mask = valid(scores)
            x = scores[mask]
            y = contrib[mask]
            
            sp_r, _ = spearmanr(x, y)
            pear_r, _ = pearsonr(x, y)
            
            ax.scatter(x, y, alpha=0.4, s=20, color=score_colors[name], edgecolors="none")
            
            # Optional: regression line
            coef = np.polyfit(x, y, 1)
            ax.plot(x, np.polyval(coef, x), color="#000000", lw=1.5, linestyle="--")
            
            ax.set_xlabel("Bias Score")
            ax.set_ylabel("|Fairness Contribution|")
            ax.set_title(f"{name}\nSpearman r={sp_r:.3f}, Pearson r={pear_r:.3f}", fontsize=11)
            ax.grid(alpha=0.3)

        fig.suptitle("Correlation: Bias Score vs Per-Sample Fairness Contribution", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "correlation_score_vs_contribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 Saved correlation_score_vs_contribution.png")

    except ImportError:
        print("\n  ⚠️  matplotlib not installed — skipping plots. Run: pip install matplotlib")