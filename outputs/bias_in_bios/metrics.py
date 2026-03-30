#!/usr/bin/env python
import json
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, mannwhitneyu, ks_2samp
import argparse
import os

# =========================
# 0. ARGPARSE
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="outputs/bias_in_bios/gpt-oss-120b")
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()
BASE = args.model_dir
print(f"🔍 Evaluating model directory: {BASE}")

# =========================
# 1. LOAD DATA
# =========================
main_file = os.path.join(BASE, "bias_in_bios_results.json")
if not os.path.exists(main_file):
    raise FileNotFoundError(f"Main results file not found: {main_file}")

with open(main_file) as f:
    results = json.load(f)["results"]

# Path to the “new metric / bias pathways” evaluation
bias_file = os.path.join(
    BASE,
    "new_metric_pathways_annotation",
    "biasbios_deepseek-chat_new_prompt_bias_pathways_simple.json"
)
with open(bias_file) as f:
    bias_data = json.load(f)["results"]

# Baseline annotations, including fairness-prm variant
baseline_paths = {
    "baseline": os.path.join(
        BASE,
        "baseline_annotation",
        "biasbios_deepseek-chat_llama70B_gt.json"
    ),
    "baseline_frm": os.path.join(
        BASE,
        "fairness-prm_biasbios",
        "bias_in_bios_results_fairness-prm.json"
    ),
    "baseline05": os.path.join(
        BASE,
        "baseline_0-5_annotation",
        "biasbios_deepseek-chat_baseline.json"
    )
}

# Load baseline data
baseline_data_dict = {}
for name, path in baseline_paths.items():
    with open(path) as f:
        baseline_data_dict[name] = json.load(f)["results"]

# =========================
# 2. ALIGN DATA LENGTHS
# =========================
all_lengths = [len(results), len(bias_data)] + [len(v) for v in baseline_data_dict.values()]
min_len = min(all_lengths)

results   = results[:min_len]
bias_data = bias_data[:min_len]
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
# 4. INCORRECT ANSWER LABEL
# =========================
for r in results:
    # Binary label for correctness: 1 = incorrect, 0 = correct
    r["_incorrect"] = 0 if r.get("predicted_profession") == r.get("true_profession") else 1

is_incorrect = np.array([r["_incorrect"] for r in results])

# =========================
# 5. FAIRNESS CONTRIBUTION (gender-based)
# =========================
groups = defaultdict(lambda: {"count_Y1":0, "count_Y0":0})

# Count per gender and correctness
for r in results:
    gender = r.get("gender", 0)
    y_true = r["_incorrect"]
    groups[gender]["count_Y1" if y_true else "count_Y0"] += 1

# Compute signed contribution
for r in results:
    gender = r.get("gender", 0)
    y_true = r["_incorrect"]
    c = (1 if y_true else 0) / max(groups[gender]["count_Y1" if y_true else "count_Y0"], 1)
    if gender != 0:  # flip sign for gender=1
        c = -c
    r["fairness_contribution"] = c
    r["abs_contribution"] = abs(c)

contrib     = np.array([r["abs_contribution"] for r in results])
contrib_sgn = np.array([r["fairness_contribution"] for r in results])
contrib_threshold = np.percentile(contrib, 75)
fairness_labels = (contrib >= contrib_threshold).astype(int)

# =========================
# 6. CORRELATION ANALYSIS
# =========================
bias_scores = np.array([r["bias_score"] for r in results])
mask = ~np.isnan(bias_scores)

sp_r, sp_p = spearmanr(bias_scores[mask], contrib[mask])
print(f"\nSpearman correlation between bias scores and per-sample fairness contribution: r={sp_r:.4f}, p={sp_p:.4e}")

sp_r2, sp_p2 = spearmanr(bias_scores[mask], is_incorrect[mask])
print(f"Spearman correlation between bias scores and incorrect answers: r={sp_r2:.4f}, p={sp_p2:.4e}")

# =========================
# 7. GROUP SCORE SEPARATION
# =========================
g0_mask = np.array([r.get("gender",0)==0 for r in results]) & mask
g1_mask = np.array([r.get("gender",0)==1 for r in results]) & mask
s0 = bias_scores[g0_mask]
s1 = bias_scores[g1_mask]

if len(s0) > 1 and len(s1) > 1:
    mw_stat, mw_p = mannwhitneyu(s0, s1, alternative="two-sided")
    ks_stat, ks_p = ks_2samp(s0, s1)
    print(f"\nBias score distribution by gender:")
    print(f"  Gender 0 mean: {np.mean(s0):.4f}, Gender 1 mean: {np.mean(s1):.4f}")
    print(f"  Mann-Whitney U: stat={mw_stat:.1f}, p={mw_p:.4e}")
    print(f"  KS test: stat={ks_stat:.4f}, p={ks_p:.4e}")