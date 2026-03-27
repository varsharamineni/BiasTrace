import json
from collections import defaultdict

# Load results
with open("outputs/COMPAS/new/compas_results.json") as f:
    data = json.load(f)

results = data

# Prepare group metrics
groups = defaultdict(lambda: {"TP":0, "FP":0, "TN":0, "FN":0, "count_Y1":0, "count_Y0":0})

for r in results:
    race = r["race"]

    # TRUE label
    y_true = r["features"].get("is_recid", 0)  # 1 = recidivated

    # PREDICTION from model_answer (A=High risk -> 1, B=Low risk -> 0)
    answer = r["model_answer"].strip().upper()
    y_pred = 1 if answer == "A" else 0

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

# Compute TPR and FPR per group
fairness_metrics = {}
for race, g in groups.items():
    TPR = g["TP"] / g["count_Y1"] if g["count_Y1"] > 0 else 0
    FPR = g["FP"] / g["count_Y0"] if g["count_Y0"] > 0 else 0
    fairness_metrics[race] = {"TPR": TPR, "FPR": FPR}

# Equalized Odds + Equalized Opportunity
races = list(fairness_metrics.keys())
if len(races) == 2:
    eo_diff_TPR = abs(fairness_metrics[races[0]]["TPR"] - fairness_metrics[races[1]]["TPR"])
    eo_diff_FPR = abs(fairness_metrics[races[0]]["FPR"] - fairness_metrics[races[1]]["FPR"])
    eq_opp_diff = eo_diff_TPR
else:
    eo_diff_TPR = eo_diff_FPR = eq_opp_diff = None

print("\nPer-group metrics:")
for race, m in fairness_metrics.items():
    print(race, m)

print("\nEqualized Odds difference:")
print("TPR diff:", eo_diff_TPR)
print("FPR diff:", eo_diff_FPR)

print("\nEqualized Opportunity difference (TPR only):")
print(eq_opp_diff)