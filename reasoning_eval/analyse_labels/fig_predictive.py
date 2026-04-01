"""
Camera-ready two-panel figure:
  Left:  Isolated effects (main coefficients for each reasoning behaviour,
         guided prompt, and ambiguous context)
  Right: Net effects across all four conditions

Significance shown via stars:
  - Isolated panel: star on bar if that term's p-value is significant
  - Net effects panel: star if main effect OR relevant interaction is significant
                       superscript i if interaction term only

Usage:
    python plot_camera_ready_net_effects.py

Output:
    publication_plot/fig_net_effects_stereo_mle_nb_camera_ready.pdf
    publication_plot/fig_net_effects_stereo_mle_nb_camera_ready.png  (300 dpi)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

LABEL_NAMES = {
    "group_assumption":          "Group assumption",
    "meta_reflection":           "Meta-reflection",
    "outside_demo_knowledge":    "Outside demo. knowledge",
    "outside_topical_knowledge": "Outside topical knowledge",
    "overthinking":              "Overthinking",
    "full_prompt":               "Guided prompt",
    "ambiguous":                 "Ambiguous context",
}

QUESTION_TYPES = ["simple_disambig", "guided_disambig", "simple_ambig", "guided_ambig"]
QTYPE_NAMES = {
    "simple_disambig":  "Simple\n(disambiguous)",
    "guided_disambig":  "Guided\n(disambiguous)",
    "simple_ambig":     "Simple\n(ambiguous)",
    "guided_ambig":     "Guided\n(ambiguous)",
}

PROMPT_KEY    = "C(prompt_type, Treatment(reference='simple_prompt'))[T.full_prompt]"
AMBIGUOUS_KEY = "C(ambiguous)[T.True]"

def int_key_prompt(label):
    return f"{label}:C(prompt_type, Treatment(reference='simple_prompt'))[T.full_prompt]"

def int_key_ambig(label):
    return f"{label}:C(ambiguous)[T.True]"

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
# Build isolated effects data
# ======================

isolated_items = JUDGE_LABELS_NO_BIAS + ["full_prompt", "ambiguous"]

isolated_rows = []
for item in isolated_items:
    if item in JUDGE_LABELS_NO_BIAS:
        c = coef.get(item, 0)
        p = pvals.get(item, 1.0)
        group = "Reasoning behaviour"
    elif item == "full_prompt":
        c = coef.get(PROMPT_KEY, 0)
        p = pvals.get(PROMPT_KEY, 1.0)
        group = "Condition"
    elif item == "ambiguous":
        c = coef.get(AMBIGUOUS_KEY, 0)
        p = pvals.get(AMBIGUOUS_KEY, 1.0)
        group = "Condition"

    isolated_rows.append({
        "item":   item,
        "label":  LABEL_NAMES[item],
        "coef":   c,
        "p":      p,
        "stars":  sig_stars(p),
        "group":  group,
    })

iso_df = pd.DataFrame(isolated_rows)

# ======================
# Build net effects data
# ======================

net_rows = []
for qtype in QUESTION_TYPES:
    for label in JUDGE_LABELS_NO_BIAS:
        main_c = coef.get(label, 0)
        main_p = pvals.get(label, 1.0)

        prompt_c  = coef.get(PROMPT_KEY, 0)
        ambig_c   = coef.get(AMBIGUOUS_KEY, 0)
        int_p_c   = coef.get(int_key_prompt(label), 0)
        int_p_p   = pvals.get(int_key_prompt(label), 1.0)
        int_a_c   = coef.get(int_key_ambig(label), 0)
        int_a_p   = pvals.get(int_key_ambig(label), 1.0)

        if qtype == "simple_disambig":
            net   = main_c
            int_p_used, int_a_used = 1.0, 1.0
        elif qtype == "guided_disambig":
            net   = main_c + prompt_c + int_p_c
            int_p_used, int_a_used = int_p_p, 1.0
        elif qtype == "simple_ambig":
            net   = main_c + ambig_c + int_a_c
            int_p_used, int_a_used = 1.0, int_a_p
        elif qtype == "guided_ambig":
            net   = main_c + prompt_c + ambig_c + int_p_c + int_a_c
            int_p_used, int_a_used = int_p_p, int_a_p

        # Significance: any relevant term sig?
        main_sig  = main_p < 0.05
        int_sig   = min(int_p_used, int_a_used) < 0.05

        # if main_sig and int_sig:
        #     stars    = sig_stars(min(main_p, int_p_used, int_a_used))
        #     sig_note = "both"
        # elif main_sig:
        #     stars    = sig_stars(main_p)
        #     sig_note = "main"
        if int_sig:
            stars    = sig_stars(min(int_p_used, int_a_used))
            sig_note = "interaction"
        else:
            stars    = ""
            sig_note = "none"

        net_rows.append({
            "question_type": qtype,
            "qtype_label":   QTYPE_NAMES[qtype],
            "judge_label":   label,
            "label_name":    LABEL_NAMES[label],
            "net":           net,
            "stars":         stars,
            "sig_note":      sig_note,
        })

net_df = pd.DataFrame(net_rows)

# ======================
# Shared colour palette
# ======================

behaviour_labels  = [LABEL_NAMES[l] for l in JUDGE_LABELS_NO_BIAS]
palette           = sns.color_palette("tab10", n_colors=len(behaviour_labels))
colour_map        = dict(zip(behaviour_labels, palette))
condition_colours = {"Guided prompt": "#888888", "Ambiguous context": "#bbbbbb"}

# ======================
# Plot
# ======================

STAR_PAD = 0.10
STAR_FS  = 8
LABEL_FS = 9
AXIS_FS  = 10

fig, (ax_iso, ax_net) = plt.subplots(
    1, 2,
    figsize=(16, 5.5),
    gridspec_kw={"width_ratios": [1, 2.2]},
)

# ── Left panel: isolated effects ──────────────────────────────────────────────

bar_colours_iso = [
    colour_map[row["label"]] if row["group"] == "Reasoning behaviour"
    else condition_colours[row["label"]]
    for _, row in iso_df.iterrows()
]

bars_iso = ax_iso.bar(
    iso_df["label"], iso_df["coef"],
    color=bar_colours_iso, edgecolor="white", linewidth=0.5, width=0.6,
)

for bar, (_, row) in zip(bars_iso, iso_df.iterrows()):
    if not row["stars"]:
        continue
    x  = bar.get_x() + bar.get_width() / 2
    val = row["coef"]
    y  = val + STAR_PAD if val >= 0 else val - STAR_PAD
    va = "bottom" if val >= 0 else "top"
    ax_iso.text(x, y, row["stars"], ha="center", va=va, fontsize=STAR_FS)

ax_iso.axhline(0, color="black", linewidth=0.8, zorder=0)
ax_iso.set_ylabel("Logit coefficient", fontsize=AXIS_FS)
ax_iso.set_xlabel("")
ax_iso.set_title("(a) Isolated effects", fontsize=AXIS_FS + 1, loc="left", fontweight="bold")
ax_iso.set_xticklabels(iso_df["label"], rotation=35, ha="right", fontsize=LABEL_FS)
ax_iso.tick_params(axis="y", labelsize=LABEL_FS - 1)

# Dashed divider between reasoning behaviours and condition bars
n_behaviours = len(JUDGE_LABELS_NO_BIAS)
ax_iso.axvline(n_behaviours - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
ymin, ymax = ax_iso.get_ylim()
# ax_iso.text(
#     n_behaviours - 0.38, ymin + 0.05 * (ymax - ymin),
#     "Conditions →", fontsize=7.5, color="gray", va="bottom",
# )

sns.despine(ax=ax_iso)

# ── Right panel: net effects ───────────────────────────────────────────────────

qtype_order = [QTYPE_NAMES[q] for q in QUESTION_TYPES]
label_order = [LABEL_NAMES[l] for l in JUDGE_LABELS_NO_BIAS]
n_labels    = len(label_order)
bar_width   = 0.8 / n_labels

sns.barplot(
    data=net_df,
    x="qtype_label", y="net",
    hue="label_name",
    hue_order=label_order,
    order=qtype_order,
    palette=colour_map,
    ax=ax_net,
)

for qidx, qtype in enumerate(QUESTION_TYPES):
    for lidx, label in enumerate(JUDGE_LABELS_NO_BIAS):
        row = net_df[
            (net_df["question_type"] == qtype) &
            (net_df["judge_label"]   == label)
        ].iloc[0]

        if not row["stars"]:
            continue

        val          = row["net"]
        x_pos        = qidx + (lidx - (n_labels - 1) / 2) * bar_width
        y_pos        = val + STAR_PAD if val >= 0 else val - STAR_PAD
        va           = "bottom" if val >= 0 else "top"

        is_int_only = row["sig_note"] == "interaction"

        # ax_net.text(
        #     x_pos, y_pos, row["stars"],
        #     ha="center", va=va,
        #     fontsize=STAR_FS,
        #     fontweight="bold" if is_int_only else "normal",
        # )

        # if is_int_only:
        #     int_label_y = (y_pos + 0.22) if val >= 0 else (y_pos - 0.22)
        #     ax_net.text(
        #         x_pos, int_label_y, "int.",
        #         ha="center", va=va,
        #         fontsize=6.5, color="dimgray", style="italic",
        #     )
        ax_net.text(
            x_pos, y_pos, row["stars"],
            ha="center", va=va,
            fontsize=STAR_FS,
)

ax_net.axhline(0, color="black", linewidth=0.8, zorder=0)
ax_net.set_ylabel("Net logit effect on biased output", fontsize=AXIS_FS)
ax_net.set_xlabel("")
ax_net.set_title("(b) Net effects by condition", fontsize=AXIS_FS + 1, loc="left", fontweight="bold")
ax_net.set_xticklabels(qtype_order, fontsize=LABEL_FS)
ax_net.tick_params(axis="y", labelsize=LABEL_FS - 1)

legend = ax_net.get_legend()
legend.set_title("Reasoning behaviour", prop={"size": LABEL_FS})
for t in legend.get_texts():
    t.set_fontsize(LABEL_FS - 1)
legend.set_bbox_to_anchor((1.01, 1))
legend.set_loc("upper left")
legend.get_frame().set_linewidth(0.5)

sns.despine(ax=ax_net)

# ── Shared footnote ────────────────────────────────────────────────────────────

# fig.text(
#     0.01, -0.05,
#     "* p < .05   ** p < .01   *** p < .001\n"
#     "Panel (b): stars indicate significant interaction terms only.",
#     fontsize=7.5, color="black", va="top",
# )
# fig.tight_layout()

# ======================
# Save
# ======================

out_pdf = OUT_DIR / "fig_net_effects_stereo_mle_nb_camera_ready.pdf"
out_png = OUT_DIR / "fig_net_effects_stereo_mle_nb_camera_ready.png"

fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")

print(f"Saved:\n  {out_pdf}\n  {out_png}")