"""
Camera-ready version of fig_net_effects_stereo_mle_nb.
Reads saved coefficient CSV — no model re-fitting needed.

Usage:
    python plot_camera_ready_net_effects.py

Output:
    fig_net_effects_stereo_mle_nb_camera_ready.pdf
    fig_net_effects_stereo_mle_nb_camera_ready.png  (300 dpi)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ======================
# Config
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

# Human-readable label names
LABEL_NAMES = {
    "group_assumption":        "Group assumption",
    "meta_reflection":         "Meta-reflection",
    "outside_demo_knowledge":  "Outside demo. knowledge",
    "outside_topical_knowledge": "Outside topical knowledge",
    "overthinking":            "Overthinking",
}

QUESTION_TYPES  = ["simple_prompt", "full_prompt", "non_ambiguous", "ambiguous"]
QTYPE_NAMES     = {
    "simple_prompt":  "Simple prompt\n(unambiguous)",
    "full_prompt":    "Guided prompt\n(unambiguous)",
    "non_ambiguous":  "Simple prompt\n(disambiguous)",
    "ambiguous":      "Simple prompt\n(ambiguous)",
}

# ======================
# Significance helper
# ======================

def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


# ======================
# Load coefficients
# ======================

coef_df = pd.read_csv(COEF_CSV, index_col=0)
coef    = coef_df["coefficient"].to_dict()
pvals   = coef_df["p_value"].to_dict()


# ======================
# Build net effects + significance
# ======================

rows = []
for qtype in QUESTION_TYPES:
    for label in JUDGE_LABELS_NO_BIAS:
        main_key        = label
        main_effect     = coef.get(main_key, 0)
        main_p          = pvals.get(main_key, 1.0)

        interaction_val = 0.0
        interaction_p   = 1.0
        q_main          = 0.0
        q_main_p        = 1.0

        if qtype == "full_prompt":
            q_main_key      = "C(prompt_type, Treatment(reference='simple_prompt'))[T.full_prompt]"
            int_key         = f"{label}:C(prompt_type, Treatment(reference='simple_prompt'))[T.full_prompt]"
            q_main          = coef.get(q_main_key, 0)
            q_main_p        = pvals.get(q_main_key, 1.0)
            interaction_val = coef.get(int_key, 0)
            interaction_p   = pvals.get(int_key, 1.0)

        elif qtype == "ambiguous":
            q_main_key      = "C(ambiguous)[T.True]"
            int_key         = f"{label}:C(ambiguous)[T.True]"
            q_main          = coef.get(q_main_key, 0)
            q_main_p        = pvals.get(q_main_key, 1.0)
            interaction_val = coef.get(int_key, 0)
            interaction_p   = pvals.get(int_key, 1.0)

        net_effect = main_effect + q_main + interaction_val

        # Stars: show stars for the interaction term specifically when qtype
        # is full_prompt or ambiguous; for baseline qtypes show main effect stars
        if qtype in ("full_prompt", "ambiguous"):
            stars = sig_stars(interaction_p)
        else:
            stars = sig_stars(main_p)

        rows.append({
            "question_type":  qtype,
            "qtype_label":    QTYPE_NAMES[qtype],
            "judge_label":    label,
            "label_name":     LABEL_NAMES[label],
            "net_logit_effect": net_effect,
            "stars":          stars,
        })

plot_df = pd.DataFrame(rows)

# ======================
# Plot
# ======================

palette = sns.color_palette("tab10", n_colors=len(JUDGE_LABELS_NO_BIAS))
label_order = [LABEL_NAMES[l] for l in JUDGE_LABELS_NO_BIAS]
qtype_order = [QTYPE_NAMES[q] for q in QUESTION_TYPES]

fig, ax = plt.subplots(figsize=(12, 5.5))

sns.barplot(
    data=plot_df,
    x="qtype_label",
    y="net_logit_effect",
    hue="label_name",
    hue_order=label_order,
    order=qtype_order,
    palette=palette,
    ax=ax,
)

ax.axhline(0, color="black", linewidth=0.8, zorder=0)

# --- Add significance stars above/below each bar ---
n_labels   = len(JUDGE_LABELS_NO_BIAS)
n_qtypes   = len(QUESTION_TYPES)
bar_width  = 0.8 / n_labels          # seaborn default total group width = 0.8
star_pad   = 0.12                     # vertical offset above/below bar tip

for qidx, qtype in enumerate(QUESTION_TYPES):
    qtype_label = QTYPE_NAMES[qtype]
    for lidx, label in enumerate(JUDGE_LABELS_NO_BIAS):
        row = plot_df[(plot_df["question_type"] == qtype) & (plot_df["judge_label"] == label)].iloc[0]
        stars = row["stars"]
        if not stars:
            continue

        val = row["net_logit_effect"]
        # x position: group centre + bar offset
        group_centre = qidx
        bar_offset   = (lidx - (n_labels - 1) / 2) * bar_width
        x_pos        = group_centre + bar_offset

        y_pos = val + star_pad if val >= 0 else val - star_pad
        va    = "bottom" if val >= 0 else "top"

        ax.text(x_pos, y_pos, stars, ha="center", va=va, fontsize=8, color="black")

# --- Formatting ---
ax.set_xlabel("")
ax.set_ylabel("Net logit effect on biased output", fontsize=11)
ax.set_xticklabels(qtype_order, fontsize=10)
ax.tick_params(axis="y", labelsize=9)

legend = ax.get_legend()
legend.set_title("Reasoning behaviour", prop={"size": 10})
for text in legend.get_texts():
    text.set_fontsize(9)
legend.set_bbox_to_anchor((1.01, 1))
legend.set_loc("upper left")
legend.get_frame().set_linewidth(0.5)

# Significance legend
star_note = ax.text(
    0.01, 0.02,
    "* p < .05   ** p < .01   *** p < .001   (stars on interaction terms for guided/ambiguous columns)",
    transform=ax.transAxes,
    fontsize=7.5,
    color="dimgray",
    va="bottom",
)

sns.despine(ax=ax)
fig.tight_layout()

# ======================
# Save
# ======================

out_pdf = OUT_DIR / "fig_net_effects_stereo_mle_nb_camera_ready.pdf"
out_png = OUT_DIR / "fig_net_effects_stereo_mle_nb_camera_ready.png"

fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")

print(f"Saved:\n  {out_pdf}\n  {out_png}")