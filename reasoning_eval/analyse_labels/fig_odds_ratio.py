"""
Odds-ratio figures for the BBQ logistic regression, one file per style.

  A  bars_log        Diverging bars from OR = 1, log x-axis
  B  forest          Point + 95% CI, log x-axis
  C  lollipop        Stem from 1, point + CI, log x-axis
  D  bars_fill_sig   Bars, significance by fill (no stars), log x-axis
  E  forest_table    Forest plot + "OR [95% CI]" text column
  F  per_condition   OR of each behaviour in each prompt x context cell
  G  bars_linear     Bars from 0 on a linear 0-8 axis (closest to the original)

Inputs come from the reviewer-response script:
  R1a_coef_stereo_cluster.csv   (coefficients, p-values, CIs on log-odds scale)
  R1a_cov_stereo_cluster.csv    (cluster-robust covariance; only needed for F)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, MultipleLocator
from scipy.stats import norm
import seaborn as sns

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"

# ======================
# Config
# ======================
RESULTS_DIR = Path("reasoning_eval/analyse_labels/bbq_analysis_clustered")
COEF_CSV = RESULTS_DIR / "R1a_coef_stereo_cluster.csv"
COV_CSV  = RESULTS_DIR / "R1a_cov_stereo_cluster.csv"
OUT_DIR  = RESULTS_DIR / "publication_plot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LINEAR_XMAX = 8          # x-axis upper limit for style G
FONT        = 14         # tick / label size (matches the paper figure)

JUDGE_LABELS_NO_BIAS = [
    "group_assumption",
    "meta_reflection",
    "outside_demo_knowledge",
    "outside_topical_knowledge",
    "overthinking",
]
LABEL_NAMES = {
    "group_assumption":          "Group Assumption",
    "meta_reflection":           "Meta Reflection",
    "outside_demo_knowledge":    "Outside Demo. knowledge",
    "outside_topical_knowledge": "Outside Topical knowledge",
    "overthinking":              "Overthinking",
    "full_prompt":               "Guided prompt",
    "ambiguous":                 "Ambiguous context",
}
PROMPT_KEY    = "C(prompt_type, Treatment(reference='simple_prompt'))[T.full_prompt]"
AMBIGUOUS_KEY = "C(ambiguous)[T.True]"
TERM_KEYS = {**{l: l for l in JUDGE_LABELS_NO_BIAS},
             "full_prompt": PROMPT_KEY, "ambiguous": AMBIGUOUS_KEY}

behaviour_names   = [LABEL_NAMES[l] for l in JUDGE_LABELS_NO_BIAS]
COLOUR_MAP        = dict(zip(behaviour_names, sns.color_palette("tab10", len(behaviour_names))))
CONDITION_COLOURS = {"Guided prompt": "#666666", "Ambiguous context": "#999999"}


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ======================
# Data
# ======================
coef_df = pd.read_csv(COEF_CSV, index_col=0)

rows = []
for item, term in TERM_KEYS.items():
    if term not in coef_df.index:
        raise KeyError(f"Term '{term}' not in {COEF_CSV}. Available: {list(coef_df.index)}")
    r = coef_df.loc[term]
    label = LABEL_NAMES[item]
    rows.append({
        "label": label,
        "or":    np.exp(r["coefficient"]),
        "or_lo": np.exp(r["ci_lo"]),        # ci_lo / ci_hi are on the log-odds scale
        "or_hi": np.exp(r["ci_hi"]),
        "p":     r["p_value"],
        "stars": sig_stars(r["p_value"]),
        "group": "Reasoning behaviour" if item in JUDGE_LABELS_NO_BIAS else "Condition",
        "colour": COLOUR_MAP.get(label, CONDITION_COLOURS.get(label)),
    })
df = pd.DataFrame(rows)

# Behaviours by |log OR| (so OR 0.5 and 2 rank equally), then the two conditions
beh  = df[df["group"] == "Reasoning behaviour"]
beh  = beh.loc[np.log(beh["or"]).abs().sort_values(ascending=False).index]
df_plot = pd.concat([beh, df[df["group"] == "Condition"]], ignore_index=True)
N_BEH = len(beh)


# ======================
# Shared helpers
# ======================
def log_axis(ax, d, pad_lo=1.35, pad_hi=1.35):
    ax.set_xscale("log")
    lo = min(d["or_lo"].min(), 1) / pad_lo
    hi = max(d["or_hi"].max(), 1) * pad_hi
    ax.set_xlim(lo, hi)
    ticks = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
    ax.xaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Odds ratio (log scale)", size=FONT, weight="medium")


def y_rows(ax, d, divider=True):
    y = np.arange(len(d))
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_ylim(len(d) - 0.5, -0.5)                 # first row at the top
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="both", labelsize=FONT)
    if divider:
        ax.axhline(N_BEH - 0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
    return y


def add_stars_log(ax, d, y):
    for yi, (_, r) in zip(y, d.iterrows()):
        if not r["stars"]:
            continue
        if r["or"] >= 1:
            ax.text(r["or_hi"] * 1.06, yi, r["stars"], va="center", ha="left", fontsize=12)
        else:
            ax.text(r["or_lo"] / 1.06, yi, r["stars"], va="center", ha="right", fontsize=12)


def ci_err(d):
    return [d["or"] - d["or_lo"], d["or_hi"] - d["or"]]


def save(fig, name):
    sns.despine(fig=fig, left=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


# ======================
# A. Diverging bars from OR = 1, log axis
# ======================
def style_a_bars_log():
    fig, ax = plt.subplots(figsize=(9, 5))
    log_axis(ax, df_plot)
    y = y_rows(ax, df_plot)
    ax.barh(y, df_plot["or"] - 1, left=1, height=0.6,
            color=df_plot["colour"], edgecolor="white", linewidth=0.6)
    ax.errorbar(df_plot["or"], y, xerr=ci_err(df_plot), fmt="none",
                ecolor="black", elinewidth=1, capsize=3)
    ax.axvline(1, color="black", lw=0.8)
    add_stars_log(ax, df_plot, y)
    save(fig, "fig_or_A_bars_log")


# ======================
# B. Forest plot
# ======================
def _forest(ax):
    log_axis(ax, df_plot)
    y = y_rows(ax, df_plot)
    for yi, (_, r) in zip(y, df_plot.iterrows()):
        ax.plot([r["or_lo"], r["or_hi"]], [yi, yi], color=r["colour"], lw=2.2,
                solid_capstyle="round")
        ax.plot(r["or"], yi, "o", color=r["colour"], ms=9, mec="white", mew=0.8, zorder=3)
    ax.axvline(1, color="black", lw=0.8, ls="--")
    return y


def style_b_forest():
    fig, ax = plt.subplots(figsize=(9, 5))
    y = _forest(ax)
    add_stars_log(ax, df_plot, y)
    save(fig, "fig_or_B_forest")


# ======================
# C. Lollipop
# ======================
def style_c_lollipop():
    fig, ax = plt.subplots(figsize=(9, 5))
    log_axis(ax, df_plot)
    y = y_rows(ax, df_plot)
    ax.hlines(y, 1, df_plot["or"], color=df_plot["colour"], lw=3, alpha=0.5)
    ax.errorbar(df_plot["or"], y, xerr=ci_err(df_plot), fmt="none",
                ecolor="black", elinewidth=0.9, capsize=2.5)
    ax.scatter(df_plot["or"], y, s=90, color=df_plot["colour"], edgecolor="white", zorder=3)
    ax.axvline(1, color="black", lw=0.8)
    add_stars_log(ax, df_plot, y)
    save(fig, "fig_or_C_lollipop")


# ======================
# D. Bars, significance by fill (solid = p < .05, faded + hatched = n.s.)
# ======================
def style_d_bars_fill_sig():
    fig, ax = plt.subplots(figsize=(9, 5))
    log_axis(ax, df_plot, pad_lo=1.15, pad_hi=1.15)
    y = y_rows(ax, df_plot)
    for yi, (_, r) in zip(y, df_plot.iterrows()):
        sig = r["p"] < 0.05
        ax.barh(yi, r["or"] - 1, left=1, height=0.6, color=r["colour"],
                alpha=1.0 if sig else 0.3, hatch=None if sig else "///",
                edgecolor="white" if sig else r["colour"], linewidth=0.8)
    ax.errorbar(df_plot["or"], y, xerr=ci_err(df_plot), fmt="none",
                ecolor="black", elinewidth=1, capsize=3)
    ax.axvline(1, color="black", lw=0.8)
    save(fig, "fig_or_D_bars_fill_sig")


# ======================
# E. Forest plot + "OR [95% CI]" column
# ======================
def style_e_forest_table():
    fig, ax = plt.subplots(figsize=(12, 5))
    y = _forest(ax)
    trans = ax.get_yaxis_transform()               # x in axes coords, y in data coords
    ax.text(1.03, -0.85, "OR [95% CI]", transform=trans, ha="left", va="center",
            fontsize=12, weight="bold")
    for yi, (_, r) in zip(y, df_plot.iterrows()):
        txt = f"{r['or']:.2f} [{r['or_lo']:.2f}, {r['or_hi']:.2f}] {r['stars']}"
        ax.text(1.03, yi, txt, transform=trans, ha="left", va="center",
                fontsize=12, family="DejaVu Sans Mono")
    save(fig, "fig_or_E_forest_table")


# ======================
# F. Per prompt x context condition
#    Behaviour OR in each cell = exp(main + relevant interactions).
#    Filled marker = p < .05, open marker = n.s.
# ======================
def _find_term(index, label, cat_key):
    """Interaction term made of exactly {label, cat_key}, in either order."""
    for t in index:
        if set(t.split(":")) == {label, cat_key}:
            return t
    raise KeyError(f"No interaction term for {label} x {cat_key}")


def per_condition_table():
    cov = pd.read_csv(COV_CSV, index_col=0)
    out = []
    for l in JUDGE_LABELS_NO_BIAS:
        t_p = _find_term(coef_df.index, l, PROMPT_KEY)
        t_a = _find_term(coef_df.index, l, AMBIGUOUS_KEY)
        cells = {                       # baseline = simple prompt, disambiguated
            "Simple, disambig.": [l],
            "Simple, ambig.":    [l, t_a],
            "Guided, disambig.": [l, t_p],
            "Guided, ambig.":    [l, t_p, t_a],
        }
        for cond, terms in cells.items():
            b  = coef_df.loc[terms, "coefficient"].sum()
            se = np.sqrt(cov.loc[terms, terms].values.sum())
            out.append({
                "label": LABEL_NAMES[l], "condition": cond,
                "or": np.exp(b), "or_lo": np.exp(b - 1.96 * se), "or_hi": np.exp(b + 1.96 * se),
                "p": 2 * norm.sf(abs(b / se)),
            })
    return pd.DataFrame(out)


def style_f_per_condition():
    if not COV_CSV.exists():
        print(f"  skipped F: {COV_CSV} not found (rerun the reviewer-response script)")
        return
    cond_df = per_condition_table()
    cond_df.to_csv(OUT_DIR / "per_condition_odds_ratios.csv", index=False)
    labels  = list(beh["label"])                    # same order as the other figures
    conds   = list(cond_df["condition"].unique())
    offsets = np.linspace(-0.27, 0.27, len(conds))
    markers = ["o", "s", "^", "D"]
    cols    = sns.color_palette("colorblind", len(conds))

    fig, ax = plt.subplots(figsize=(9, 6.5))
    log_axis(ax, cond_df)
    for k, c in enumerate(conds):
        d = cond_df[cond_df["condition"] == c].set_index("label").loc[labels]
        y = np.arange(len(labels)) + offsets[k]
        ax.errorbar(d["or"], y, xerr=[d["or"] - d["or_lo"], d["or_hi"] - d["or"]],
                    fmt="none", ecolor=cols[k], elinewidth=1.2)
        sig = (d["p"] < 0.05).values
        ax.scatter(d["or"][sig],  y[sig],  marker=markers[k], s=45, color=cols[k], zorder=3, label=c)
        ax.scatter(d["or"][~sig], y[~sig], marker=markers[k], s=45, facecolor="white",
                   edgecolor=cols[k], zorder=3)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="both", labelsize=FONT)
    ax.axvline(1, color="black", lw=0.8, ls="--")
    ax.legend(fontsize=11, frameon=False, loc="best", title="Prompt, context",
              title_fontsize=11)
    save(fig, "fig_or_F_per_condition")


# ======================
# G. Bars from 0 on a linear axis (closest to the original figure)
# ======================
def style_g_bars_linear():
    xmax = LINEAR_XMAX
    if df_plot["or_hi"].max() > xmax:
        xmax = int(np.ceil(df_plot["or_hi"].max() * 1.1))
        print(f"  note: CI exceeds {LINEAR_XMAX}; style G x-axis extended to {xmax}")

    fig, ax = plt.subplots(figsize=(9, 5))
    y = y_rows(ax, df_plot)
    ax.barh(y, df_plot["or"], height=0.6, color=df_plot["colour"],
            edgecolor="white", linewidth=0.6)
    ax.errorbar(df_plot["or"], y, xerr=ci_err(df_plot), fmt="none",
                ecolor="black", elinewidth=1, capsize=3)
    ax.axvline(1, color="black", lw=0.8, ls="--")   # OR = 1: no association
    for yi, (_, r) in zip(y, df_plot.iterrows()):
        if r["stars"]:
            ax.text(max(r["or_hi"], 1) + 0.1, yi, r["stars"], va="center", ha="left", fontsize=12)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel("Odds ratio", size=FONT, weight="medium")
    save(fig, "fig_or_G_bars_linear")


if __name__ == "__main__":
    print(f"Writing figures to {OUT_DIR}")
    style_a_bars_log()
    style_b_forest()
    style_c_lollipop()
    style_d_bars_fill_sig()
    style_e_forest_table()
    style_f_per_condition()
    style_g_bars_linear()