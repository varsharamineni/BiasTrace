import json
import numpy as np
from collections import defaultdict

# =========================
# 1. LOAD DATA
# =========================

with open("outputs/COMPAS/new/compas_results.json") as f:
    results = json.load(f)

with open("outputs/COMPAS/new/llm_judge/llm_eval_deepseek-chat_COMPAS_new_prompt_explain_temp1.0_top_p0.9_seed42_max_tokens2048.json") as f:
    bias_data = json.load(f)

with open("outputs/COMPAS/new/llm_judge_baseline05/llm_eval_deepseek-chat_COMPAS_baseline_temp1.0_top_p0.9_seed42_max_tokens2048.json") as f:
    baseline_data = json.load(f)

# =========================
# 2. UNWRAP RESULTS
# =========================

bias_data = bias_data["results"]
baseline_data = baseline_data["results"]

print("Lengths:")
print("results:", len(results))
print("bias_data:", len(bias_data))
print("baseline_data:", len(baseline_data))

# =========================
# 3. ALIGN DATA (SAFE)
# =========================

min_len = min(len(results), len(bias_data), len(baseline_data))
print(f"Truncating to {min_len} samples")

results = results[:min_len]

# =========================
# 4. EXTRACT SCORE (robust)
# =========================

def extract_bias_score(entry):
    """
    Extract numeric bias score from judge_output in multiple formats:
    1. {"bias_label": 1}
    2. {"bias_label": {"score": 1}}
    3. {"bias_label": {"explanation": "...", "score": 1}}
    4. {"score": 3}
    Returns 0.0 if no numeric score is found.
    """
    jo = entry.get("judge_output", {})

    # Case 4: top-level "score" key
    if "score" in jo:
        return float(jo["score"])

    # Case 1/2/3: "bias_label" key
    label = jo.get("bias_label", 0)
    if isinstance(label, dict):
        # try to extract "score" from dict
        return float(label.get("score", 0))
    else:
        # assume it's already numeric
        try:
            return float(label)
        except (TypeError, ValueError):
            return 0.0

for i in range(min_len):
    results[i]["bias_score"] = extract_bias_score(bias_data[i])
    results[i]["baseline_score"] = 1 if extract_bias_score(baseline_data[i]) > 0 else 0

# =========================
# 5. FAIRNESS METRICS
# =========================

def compute_group_stats(results):
    groups = defaultdict(lambda: {
        "TP":0, "FP":0, "TN":0, "FN":0,
        "count_Y1":0, "count_Y0":0
    })

    for r in results:
        race = r["race"]
        y_true = r["features"].get("is_recid", 0)
        y_pred = 1 if r["model_answer"].strip().upper() == "A" else 0

        if y_true == 1:
            groups[race]["count_Y1"] += 1
            if y_pred == 1:
                groups[race]["TP"] += 1
            else:
                groups[race]["FN"] += 1
        else:
            groups[race]["count_Y0"] += 1
            if y_pred == 1:
                groups[race]["FP"] += 1
            else:
                groups[race]["TN"] += 1

    return groups

def compute_fairness(results):
    groups = compute_group_stats(results)

    fairness_metrics = {}
    for race, g in groups.items():
        TPR = g["TP"] / g["count_Y1"] if g["count_Y1"] > 0 else 0
        FPR = g["FP"] / g["count_Y0"] if g["count_Y0"] > 0 else 0
        fairness_metrics[race] = {"TPR": TPR, "FPR": FPR}

    races = list(fairness_metrics.keys())
    if len(races) == 2:
        r1, r2 = races
        eo_diff_TPR = abs(fairness_metrics[r1]["TPR"] - fairness_metrics[r2]["TPR"])
        eo_diff_FPR = abs(fairness_metrics[r1]["FPR"] - fairness_metrics[r2]["FPR"])
        eq_opp = eo_diff_TPR
        eo = eo_diff_TPR + eo_diff_FPR
    else:
        eo_diff_TPR = eo_diff_FPR = eq_opp = eo = None

    return fairness_metrics, eo, eq_opp

# =========================
# 6. PER-SAMPLE CONTRIBUTION
# =========================

groups = compute_group_stats(results)
races = list(groups.keys())

if len(races) != 2:
    raise ValueError("Script assumes exactly 2 groups")

a, b = races[0], races[1]
denoms = {race: {"Y1": groups[race]["count_Y1"], "Y0": groups[race]["count_Y0"]} for race in races}

for r in results:
    race = r["race"]
    y_true = r["features"].get("is_recid", 0)
    y_pred = 1 if r["model_answer"].strip().upper() == "A" else 0

    if y_true == 1:
        c = y_pred / max(denoms[race]["Y1"], 1)
        if race != a: c = -c
    else:
        c = y_pred / max(denoms[race]["Y0"], 1)
        if race != a: c = -c

    r["fairness_contribution"] = c
    r["abs_contribution"] = abs(c)

# =========================
# 7. CORRELATION + OVERLAP
# =========================

bias = np.array([r["bias_score"] for r in results])
baseline = np.array([r["baseline_score"] for r in results])
contrib = np.array([r["abs_contribution"] for r in results])

print("\n=== Correlation with fairness contribution ===")
print("Your method:", np.corrcoef(bias, contrib)[0,1])
print("Baseline:", np.corrcoef(baseline, contrib)[0,1])

def topk_overlap(scores1, scores2, k=0.1):
    n = len(scores1)
    k_n = max(int(n*k), 1)
    idx1 = np.argsort(scores1)[-k_n:]
    idx2 = np.argsort(scores2)[-k_n:]
    return len(set(idx1) & set(idx2)) / k_n

print("\n=== Top-K overlap (10%) ===")
print("Your method:", topk_overlap(bias, contrib))
print("Baseline:", topk_overlap(baseline, contrib))

# =========================
# 8. REMOVAL EXPERIMENT
# =========================

def compute_eo(results):
    _, eo, _ = compute_fairness(results)
    return eo

def removal_curve(results, score_key, steps=20, min_size=50):
    results_sorted = sorted(results, key=lambda x: x[score_key], reverse=True)
    n = len(results)
    fractions = np.linspace(0, 0.5, steps)
    eo_values = []
    for frac in fractions:
        k = int(n*frac)
        remaining = results_sorted[k:]
        if len(remaining) < min_size:
            eo_values.append(np.nan)
        else:
            eo_values.append(compute_eo(remaining))
    return fractions, eo_values

fractions, eo_yours = removal_curve(results, "bias_score")
_, eo_baseline = removal_curve(results, "baseline_score")
rand_results = [{**r, "rand": np.random.rand()} for r in results]
_, eo_random = removal_curve(rand_results, "rand")

# Print results
for i in range(len(fractions)):
    print(f"Remove {fractions[i]:.2f} | Yours: {eo_yours[i]:.4f} | Baseline: {eo_baseline[i]:.4f} | Random: {eo_random[i]:.4f}")

def area_under_curve(y):
    y = np.array(y)
    y = y[~np.isnan(y)]
    return np.trapz(y)

print("\n=== Removal AUC (lower is better) ===")
print("Your method:", area_under_curve(eo_yours))
print("Baseline:", area_under_curve(eo_baseline))
print("Random:", area_under_curve(eo_random))

# =========================
# 9. FINAL FAIRNESS
# =========================

fairness_metrics, eo, eq_opp = compute_fairness(results)

print("\n=== Per-group metrics ===")
for race, m in fairness_metrics.items():
    print(race, m)

print("\n=== Overall fairness ===")
print("Equalized Odds:", eo)
print("Equalized Opportunity:", eq_opp)