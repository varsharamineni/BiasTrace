from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# Config (same as main script)
# ======================

COEF_CSV = Path("reasoning_eval/analyse_labels/bbq_analysis_new/table_coef_stereo_mle_nb.csv")
OUT_DIR  = Path("reasoning_eval/analyse_labels/bbq_analysis_new/publication_plot")
OUT_DIR.mkdir(parents=True, exist_ok=True)

JUDGE_LABELS_NO_BIAS = [
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

# ======================
# Significance helper
# ======================

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# ======================
# Load coefficients
# ======================

coef_df = pd.read_csv(COEF_CSV, index_col=0)
coef    = coef_df["coefficient"].to_dict()
pvals   = coef_df["p_value"].to_dict()

# ======================
# Build isolated effects
# ======================

items = JUDGE_LABELS_NO_BIAS + ["full_prompt", "ambiguous"]

rows = []
for item in items:
    if item in JUDGE_LABELS_NO_BIAS:
        c = coef.get(item, 0)
        p = pvals.get(item, 1.0)
        group = "Reasoning behaviour"
    elif item == "full_prompt":
        c = coef.get(PROMPT_KEY, 0)
        p = pvals.get(PROMPT_KEY, 1.0)
        group = "Condition"
    else:
        c = coef.get(AMBIGUOUS_KEY, 0)
        p = pvals.get(AMBIGUOUS_KEY, 1.0)
        group = "Condition"

    rows.append({
        "label": LABEL_NAMES[item],
        "coef":  c,
        "stars": sig_stars(p),
        "group": group,
    })

df = pd.DataFrame(rows)

# ======================
# Colours (same palette)
# ======================

behaviour_labels  = [LABEL_NAMES[l] for l in JUDGE_LABELS_NO_BIAS]
palette           = sns.color_palette("tab10", n_colors=len(behaviour_labels))
colour_map        = dict(zip(behaviour_labels, palette))
condition_colours = {"Guided prompt": "#888888", "Ambiguous context": "#bbbbbb"}

bar_colours = [
    colour_map[row["label"]] if row["group"] == "Reasoning behaviour"
    else condition_colours[row["label"]]
    for _, row in df.iterrows()
]

# ======================
# Plot
# ======================

plt.figure(figsize=(7, 5.5))

bars = plt.bar(
    df["label"], df["coef"],
    color=bar_colours,
    edgecolor="white",
    linewidth=0.6,
    width=0.6,
)

# Stars
STAR_PAD = 0.10
for bar, (_, row) in zip(bars, df.iterrows()):
    if not row["stars"]:
        continue
    x = bar.get_x() + bar.get_width() / 2
    val = row["coef"]
    y = val + STAR_PAD if val >= 0 else val - STAR_PAD
    va = "bottom" if val >= 0 else "top"
    plt.text(x, y, row["stars"], ha="center", va=va, fontsize=9)

# Styling
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("Logit coefficient")
plt.title("Effects of Reasoning Behaviours on Biased Outcomes", fontsize=11)
plt.xticks(rotation=35, ha="right")

# Divider between behaviours and conditions
n_behaviours = len(JUDGE_LABELS_NO_BIAS)
plt.axvline(n_behaviours - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

sns.despine()
plt.tight_layout()

# ======================
# Save
# ======================

out_pdf = OUT_DIR / "fig_isolated_effects.pdf"
out_png = OUT_DIR / "fig_isolated_effects.png"

plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=300, bbox_inches="tight")

print(f"Saved:\n  {out_pdf}\n  {out_png}")