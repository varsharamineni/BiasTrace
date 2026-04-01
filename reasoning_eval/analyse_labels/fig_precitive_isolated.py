import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# ======================
# Load data (reuse yours)
# ======================

COEF_CSV = Path("reasoning_eval/analyse_labels/bbq_analysis_new/table_coef_stereo_mle_nb.csv")
coef_df = pd.read_csv(COEF_CSV, index_col=0)

coef  = coef_df["coefficient"].to_dict()
pvals = coef_df["p_value"].to_dict()

# ======================
# Labels
# ======================

JUDGE_LABELS = [
    "group_assumption",
    "meta_reflection",
    "outside_demo_knowledge",
    "outside_topical_knowledge",
    "overthinking",
]

LABEL_NAMES = {
    "group_assumption":          "Group assumption",
    "meta_reflection":           "Meta-reflection",
    "outside_demo_knowledge":    "Outside demo. knowledge",
    "outside_topical_knowledge": "Outside topical knowledge",
    "overthinking":              "Overthinking",
    "full_prompt":               "Guided prompt",
    "ambiguous":                 "Ambiguous context",
}

PROMPT_KEY    = "C(prompt_type, Treatment(reference='simple_prompt'))[T.full_prompt]"
AMBIGUOUS_KEY = "C(ambiguous)[T.True]"

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# ======================
# Build dataframe
# ======================

rows = []

# reasoning behaviours
for label in JUDGE_LABELS:
    rows.append({
        "label": LABEL_NAMES[label],
        "coef":  coef.get(label, 0),
        "p":     pvals.get(label, 1.0),
        "group": "Reasoning behaviour",
        "stars": sig_stars(pvals.get(label, 1.0)),
    })

# conditions
rows.append({
    "label": LABEL_NAMES["full_prompt"],
    "coef":  coef.get(PROMPT_KEY, 0),
    "p":     pvals.get(PROMPT_KEY, 1.0),
    "group": "Condition",
    "stars": sig_stars(pvals.get(PROMPT_KEY, 1.0)),
})

rows.append({
    "label": LABEL_NAMES["ambiguous"],
    "coef":  coef.get(AMBIGUOUS_KEY, 0),
    "p":     pvals.get(AMBIGUOUS_KEY, 1.0),
    "group": "Condition",
    "stars": sig_stars(pvals.get(AMBIGUOUS_KEY, 1.0)),
})

df = pd.DataFrame(rows)

# ======================
# Plot
# ======================

plt.figure(figsize=(7, 4.5))

palette = sns.color_palette("tab10", n_colors=5)
colour_map = dict(zip(
    [LABEL_NAMES[l] for l in JUDGE_LABELS],
    palette
))

condition_colours = {
    "Guided prompt": "#888888",
    "Ambiguous context": "#bbbbbb"
}

colors = [
    colour_map[row["label"]] if row["group"] == "Reasoning behaviour"
    else condition_colours[row["label"]]
    for _, row in df.iterrows()
]

bars = plt.bar(df["label"], df["coef"], color=colors, edgecolor="white")

# stars
for bar, (_, row) in zip(bars, df.iterrows()):
    if not row["stars"]:
        continue
    x = bar.get_x() + bar.get_width()/2
    y = row["coef"]
    offset = 0.1 if y >= 0 else -0.1
    plt.text(x, y + offset, row["stars"], ha="center",
             va="bottom" if y >= 0 else "top", fontsize=9)

plt.axhline(0, color="black", linewidth=0.8)

plt.ylabel("Logit coefficient")
plt.xticks(rotation=30, ha="right")
plt.title("Isolated Effects of Reasoning Behaviours")

sns.despine()
plt.tight_layout()

plt.savefig("isolated_effects.pdf")
plt.savefig("isolated_effects.png", dpi=300)

print("Saved isolated_effects.pdf/png")