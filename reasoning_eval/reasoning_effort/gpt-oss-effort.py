#!/usr/bin/env python
"""
reasoning_effort_analysis.py

Analyses the effect of reasoning effort level (low vs medium) on:
  1. Biased outcome (incorrect_and_stereotype)
  2. Reasoning behaviour labels (overthinking, group_assumption, etc.)

Only processes GPT-OSS models. Folder structure matches the main pipeline:
  outputs/<model_prompt_folder>/<timestamp>/

Run:
    python reasoning_effort_analysis.py [--output_dir ./reasoning_effort_results]
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
from pathlib import Path

# ---------------------------
# Constants
# ---------------------------

# GPT-OSS base dirs — matches BASE_DIRS in main pipeline (gpt-oss entries only)
GPT_OSS_BASE_DIRS = [
    "outputs/gpt-oss-120b_simple_prompt_low_reasoning/20251216_114545",
    "outputs/gpt-oss-120b_simple_prompt_medium_reasoning/20251217_110543",
    "outputs/gpt-oss-120b_full_prompt_low_reasoning/20251218_140849",
    "outputs/gpt-oss-120b_full_prompt_low_reasoning/20251225_204037",
    "outputs/gpt-oss-120b_full_prompt_medium_reasoning/20251218_113157",
    "outputs/gpt-oss-120b_full_prompt_medium_reasoning/20251225_224835",
    "outputs/gpt-oss-120b_full_prompt_medium_reasoning/20251226_123752",
]

error_labels = [
    'group_assumption',
    'bias_acknowledgement',
    'meta_reflection',
    'outside_demo_knowledge',
    'outside_topical_knowledge',
    'overthinking'
]

OUTCOME_COL = 'incorrect_and_stereotype'
REASONING_COL = 'reasoning_effort'


from transformers import AutoTokenizer

# Load once globally (GPT-OSS tokenizer)
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b", use_fast=True)

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))



# ---------------------------
# Utils
# ---------------------------
def find_jsons(base_dir):
    return [y for x in os.walk(base_dir) for y in glob.glob(os.path.join(x[0], 'llm_eval_*.json'))]


def parse_judge_output(r):
    jo = r.get("judge_output", {})
    errors = {lbl: 0 for lbl in error_labels}
    if not isinstance(jo, dict):
        return errors
    for lbl in error_labels:
        errors[lbl] = jo.get(lbl, 0)
    return errors

def parse_model_prompt(base_dir):
    """
    Matches parse_model_prompt() from the main pipeline.
    Returns (model_type, prompt_type, reasoning_level) inferred from folder path.
    """
    parts = base_dir.rstrip("/").split(os.sep)
    try:
        model_prompt_folder = parts[parts.index("outputs") + 1]
    except ValueError:
        model_prompt_folder = parts[-1]

    tokens = model_prompt_folder.split("_")

    if model_prompt_folder.startswith("gpt-oss-120b"):
        model_type = "GPT-OSS-120B"
        rest = "_".join(tokens[1:])
    else:
        model_type = tokens[0]
        rest = "_".join(tokens[1:])

    reasoning_level = "unknown"
    for level in ("low", "medium", "high"):
        if f"{level}_reasoning" in rest:
            reasoning_level = level
            break

    if "simple" in rest:
        prompt_type = "simple_prompt"
    elif "full" in rest:
        prompt_type = "full_prompt"
    else:
        prompt_type = "unknown"

    return model_type, prompt_type, reasoning_level


def get_full_subfolder(base_dir):
    """Find the 'full' annotation subfolder within a timestamped base dir."""
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path) and "full" in name.lower():
            return full_path
    return None


def load_base_dir(base_dir):
    """Load all llm_eval_*.json from the 'full' subfolder of a base_dir."""
    model_type, prompt_type, reasoning_level = parse_model_prompt(base_dir)

    full_subfolder = get_full_subfolder(base_dir)
    if full_subfolder is None:
        # Some dirs have jsons directly (no subfolder structure)
        json_files = find_jsons(base_dir)
    else:
        json_files = find_jsons(full_subfolder)

    if not json_files:
        print(f"  No JSONs found in {base_dir}")
        return pd.DataFrame()

    rows = []
    for f in json_files:
        with open(f) as file:
            data = json.load(file)
        category = data.get("metadata", {}).get("bbq_category", "unknown")


        for r in data.get("results", []):


            errors = parse_judge_output(r)
            row = {
                'sample_id':                r.get('sample_id'),
                'bbq_category':             category,
                'model':                    model_type,
                'ambiguous':                int(r.get('ambiguous', False)),
                'prompt_type':              prompt_type,
                'reasoning_effort':         reasoning_level,
                'base_dir':                 base_dir,
                'is_correct':               int(r.get('is_correct', False)),
                'stereotype_aligned':       int(r.get('stereotype_alignment', False)),
                'incorrect_and_stereotype': int(r.get('incorrect_and_stereotype', False)),
            }
            row.update(errors)
            row['agg_errors'] = sum(errors.values())
            row['incorrect'] = 1 - row['is_correct']
            rows.append(row)

    print(f"  Loaded {len(rows):,} rows | {model_type} | {prompt_type} | reasoning={reasoning_level}")
    return pd.DataFrame(rows)


# ---------------------------
# Analysis functions
# ---------------------------
def descriptive_stats(df, group_col=REASONING_COL):
    """Compute mean rates for outcome and all error labels by reasoning effort."""
    cols = [OUTCOME_COL, 'incorrect'] + error_labels + ['agg_errors']
    summary = df.groupby(group_col)[cols].agg(['mean', 'std', 'sum', 'count'])
    return summary


def chi2_test(df, col, group_col=REASONING_COL):
    """Chi-squared test between two reasoning effort groups for a binary column."""
    groups = df[group_col].unique()
    if len(groups) < 2:
        return None
    contingency = pd.crosstab(df[group_col], df[col])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {'chi2': chi2, 'p': p, 'dof': dof}


def run_logistic(df, predictors, outcome=OUTCOME_COL):
    """Logistic regression with reasoning effort and optional covariates."""
    df_clean = df.dropna(subset=predictors + [outcome])
    if df_clean.empty or df_clean[outcome].nunique() < 2:
        return None
    X = sm.add_constant(df_clean[predictors])
    y = df_clean[outcome]
    try:
        model = sm.Logit(y, X).fit(disp=False, method='bfgs', maxiter=200)
        return model
    except Exception as e:
        print(f"  Logistic regression failed: {e}")
        return None


def cohens_d(group1, group2):
    """Cohen's d effect size."""
    diff = group1.mean() - group2.mean()
    pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
    return diff / pooled_std if pooled_std > 0 else np.nan


def effect_size_table(df, cols, group_col=REASONING_COL):
    """Compute Cohen's d between all pairs of reasoning effort groups for each column."""
    groups = sorted(df[group_col].unique())
    rows = []
    for col in cols:
        for g1, g2 in combinations(groups, 2):
            a = df[df[group_col] == g1][col].dropna()
            b = df[df[group_col] == g2][col].dropna()
            d = cohens_d(a, b)
            rows.append({'metric': col, 'group1': g1, 'group2': g2, 'cohens_d': d})
    return pd.DataFrame(rows)


# ---------------------------
# Plotting
# ---------------------------
def plot_outcome_by_reasoning(df, output_dir, group_col=REASONING_COL):
    """Bar chart: outcome rates by reasoning effort level, split by prompt type."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    outcomes = [OUTCOME_COL, 'incorrect']
    titles = ['Biased Outcomes', 'Overall Incorrect']

    for ax, col, title in zip(axes, outcomes, titles):
        summary = df.groupby([group_col, 'prompt_type'])[col].mean().reset_index()
        summary[col] *= 100
        sns.barplot(data=summary, x=group_col, y=col, hue='prompt_type', ax=ax,
                    palette='muted', edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel('GPT-OSS-120B Reasoning Effort')
        ax.set_ylabel('Rate (%)')
        ax.legend(title='Prompt Type')

        handles, labels = ax.get_legend_handles_labels()

        label_map = {
            'full_prompt': 'Guided Prompt',
            'simple_prompt': 'Simple Prompt'
        }

        ax.legend(
            handles,
            [label_map.get(l, l) for l in labels],
            title='Prompt Type'
        )

    #plt.suptitle('GPT-OSS: Effect of Reasoning Effort on Bias Outcomes', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outcome_by_reasoning_effort.pdf'), dpi=300)
    plt.close()
    print("Saved: outcome_by_reasoning_effort.pdf")


def plot_error_labels_by_reasoning(df, output_dir, group_col=REASONING_COL):
    """Bar chart: each error label rate by reasoning effort."""
    summary = df.groupby(group_col)[error_labels].mean().reset_index()
    summary_long = summary.melt(id_vars=group_col, var_name='Error Label', value_name='Rate')
    summary_long['Rate'] *= 100

    plt.figure(figsize=(13, 5))
    sns.barplot(data=summary_long, x='Error Label', y='Rate', hue=group_col,
                palette='Set1', edgecolor='black')
    plt.title('GPT-OSS: Reasoning Behaviour Labels by Reasoning Effort Level')
    plt.ylabel('Prevalence (%)')
    plt.xlabel('')
    plt.xticks(rotation=20, ha='right')
    plt.legend(title='Reasoning Effort')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_labels_by_reasoning_effort.pdf'), dpi=300)
    plt.close()
    print("Saved: error_labels_by_reasoning_effort.pdf")


def plot_error_labels_by_reasoning_and_prompt(df, output_dir, group_col=REASONING_COL):
    """Faceted: error label rates by reasoning effort AND prompt type."""
    prompt_types = df['prompt_type'].unique()
    fig, axes = plt.subplots(1, len(prompt_types), figsize=(14, 5), sharey=True)
    if len(prompt_types) == 1:
        axes = [axes]

    for ax, pt in zip(axes, prompt_types):
        sub = df[df['prompt_type'] == pt]
        summary = sub.groupby(group_col)[error_labels].mean().reset_index()
        summary_long = summary.melt(id_vars=group_col, var_name='Error Label', value_name='Rate')
        summary_long['Rate'] *= 100
        sns.barplot(data=summary_long, x='Error Label', y='Rate', hue=group_col,
                    palette='Set1', edgecolor='black', ax=ax)
        ax.set_title(pt.replace('_', ' ').title())
        ax.set_xlabel('')
        ax.set_ylabel('Prevalence (%)')
        ax.tick_params(axis='x', rotation=25)
        ax.legend(title='Reasoning Effort', fontsize=8)

    plt.suptitle('GPT-OSS: Error Labels by Reasoning Effort and Prompt Type', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_labels_by_reasoning_and_prompt.pdf'), dpi=300)
    plt.close()
    print("Saved: error_labels_by_reasoning_and_prompt.pdf")


def plot_category_heatmap(df, col, output_dir, group_col=REASONING_COL):
    """Heatmap: outcome rate by BBQ category and reasoning effort."""
    pivot = df.groupby(['bbq_category', group_col])[col].mean().unstack() * 100
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='Reds', cbar_kws={'label': 'Biased Outcomes (%)'})
    #plt.title(f'GPT-OSS: {col} Rate (%) by Category and Reasoning Effort')
    plt.ylabel('BBQ Category')
    plt.xlabel('GPT-OSS-120B Reasoning Effort')
    plt.tight_layout()
    fname = f'heatmap_{col}_by_category_reasoning.pdf'
    plt.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.close()
    print(f"Saved: {fname}")


# ---------------------------
# Main
# ---------------------------
def main(output_dir, base_dirs=None):
    os.makedirs(output_dir, exist_ok=True)

    dirs_to_process = base_dirs if base_dirs else GPT_OSS_BASE_DIRS
    print(f"\nProcessing {len(dirs_to_process)} GPT-OSS directories...")

    all_dfs = []
    for base_dir in dirs_to_process:
        if not os.path.isdir(base_dir):
            print(f"  Skipping (not found): {base_dir}")
            continue
        df = load_base_dir(base_dir)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("No data loaded. Check that GPT_OSS_BASE_DIRS paths exist.")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows: {len(df)}")
    print(f"Reasoning effort levels: {sorted(df[REASONING_COL].unique())}")
    print(f"Prompt types: {sorted(df['prompt_type'].unique())}")

        # =========================
    # Reasoning token summary by effort
    # =========================

    token_summary = (
        df.groupby("reasoning_effort")["reasoning_tokens"]
        .agg(
            n="count",
            mean_tokens="mean",
            std_tokens="std",
            median_tokens="median",
            min_tokens="min",
            max_tokens="max",
        )
        .reset_index()
    )

    token_summary["mean_log_tokens"] = (
        df.groupby("reasoning_effort")["log_reasoning_tokens"].mean().values
    )

    print("\n--- Reasoning Token Summary by Effort ---")
    print(token_summary)

    token_summary.to_csv(
        os.path.join(output_dir, "reasoning_token_summary_by_effort.csv"),
        index=False
    )

    # ---------------------------
    # 1. Descriptive stats
    # ---------------------------
    print("\n--- Descriptive Statistics by Reasoning Effort ---")
    desc = df.groupby(REASONING_COL)[[OUTCOME_COL, 'incorrect'] + error_labels + ['agg_errors']].agg(
        ['mean', 'std', 'sum', 'count']
    )
    desc.to_csv(os.path.join(output_dir, 'descriptive_stats_by_reasoning.csv'))
    print(desc.round(4))

    # Also by reasoning effort × prompt type
    desc2 = df.groupby([REASONING_COL, 'prompt_type'])[[OUTCOME_COL, 'incorrect'] + error_labels].mean()
    desc2.to_csv(os.path.join(output_dir, 'descriptive_stats_by_reasoning_and_prompt.csv'))
    print("\n--- Means by Reasoning Effort × Prompt Type ---")
    print(desc2.round(4))

    # By category
    desc3 = df.groupby([REASONING_COL, 'bbq_category'])[[OUTCOME_COL, 'incorrect'] + error_labels].mean()
    desc3.to_csv(os.path.join(output_dir, 'descriptive_stats_by_reasoning_and_category.csv'))

    # ---------------------------
    # 2. Statistical tests: chi-squared for each metric
    # ---------------------------
    print("\n--- Chi-Squared Tests (low vs medium reasoning) ---")
    test_cols = [OUTCOME_COL, 'incorrect'] + error_labels
    chi2_results = []
    for col in test_cols:
        result = chi2_test(df, col)
        if result:
            chi2_results.append({'metric': col, **result})
            sig = '***' if result['p'] < 0.001 else ('**' if result['p'] < 0.01 else ('*' if result['p'] < 0.05 else ''))
            print(f"  {col:35s}: χ²={result['chi2']:.2f}, p={result['p']:.4f} {sig}")

    chi2_df = pd.DataFrame(chi2_results)
    chi2_df.to_csv(os.path.join(output_dir, 'chi2_tests_reasoning_effort.csv'), index=False)

    # ---------------------------
    # 3. Effect sizes (Cohen's d)
    # ---------------------------
    print("\n--- Cohen's d: Effect of Reasoning Effort ---")
    effect_df = effect_size_table(df, test_cols)
    effect_df.to_csv(os.path.join(output_dir, 'effect_sizes_cohens_d.csv'), index=False)
    print(effect_df.to_string(index=False))

    # ---------------------------
    # 4. Logistic regression: reasoning effort → biased outcome
    #    (controlling for prompt type, ambiguity, category)
    # ---------------------------
    print("\n--- Logistic Regression: Reasoning Effort → Biased Outcome ---")

    # Encode reasoning effort as binary (low=0, medium=1)
    effort_map = {'low': 0, 'medium': 1, 'high': 2}
    df['reasoning_effort_num'] = df[REASONING_COL].map(effort_map)

    # Dummies for prompt type
    df['is_full_prompt'] = (df['prompt_type'] == 'full_prompt').astype(int)

    # Category dummies (drop first to avoid multicollinearity)
    cat_dummies = pd.get_dummies(df['bbq_category'], prefix='cat', drop_first=True).astype(int)
    df_logit = pd.concat([df, cat_dummies], axis=1)
    cat_cols = list(cat_dummies.columns)

    # Model 1: reasoning effort only
    m1 = run_logistic(df_logit, ['reasoning_effort_num'], OUTCOME_COL)
    # Model 2: + prompt type and ambiguity
    m2 = run_logistic(df_logit, ['reasoning_effort_num', 'is_full_prompt', 'ambiguous'], OUTCOME_COL)
    # Model 3: + category dummies
    m3 = run_logistic(df_logit, ['reasoning_effort_num', 'is_full_prompt', 'ambiguous'] + cat_cols, OUTCOME_COL)

    logit_rows = []
    for name, model in [("M1: effort only", m1), ("M2: +prompt+ambig", m2), ("M3: +categories", m3)]:
        if model is None:
            continue
        print(f"\n  {name}")
        pseudo_r2 = 1 - model.llf / model.llnull
        print(f"  Pseudo-R²: {pseudo_r2:.4f}")
        for var in model.params.index:
            logit_rows.append({
                'model': name,
                'variable': var,
                'coef': model.params[var],
                'OR': np.exp(model.params[var]),
                'p_value': model.pvalues[var],
                'ci_lower': np.exp(model.conf_int().loc[var, 0]),
                'ci_upper': np.exp(model.conf_int().loc[var, 1]),
                'pseudo_r2': pseudo_r2
            })
        # Print key predictor only
        if 'reasoning_effort_num' in model.params.index:
            coef = model.params['reasoning_effort_num']
            p = model.pvalues['reasoning_effort_num']
            OR = np.exp(coef)
            print(f"  reasoning_effort_num: coef={coef:.4f}, OR={OR:.4f}, p={p:.4f}")

    logit_df = pd.DataFrame(logit_rows)
    logit_df.to_csv(os.path.join(output_dir, 'logistic_regression_reasoning_effort.csv'), index=False)
    print("\nSaved: logistic_regression_reasoning_effort.csv")

    # ---------------------------
    # 5. Logistic regression: reasoning effort → each error label
    # ---------------------------
    print("\n--- Logistic Regression: Reasoning Effort → Each Error Label ---")
    label_rows = []
    for lbl in error_labels:
        if df_logit[lbl].nunique() < 2:
            continue
        model = run_logistic(df_logit, ['reasoning_effort_num', 'is_full_prompt', 'ambiguous'], lbl)
        if model is None:
            continue
        coef = model.params.get('reasoning_effort_num', np.nan)
        p = model.pvalues.get('reasoning_effort_num', np.nan)
        OR = np.exp(coef)
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"  {lbl:35s}: coef={coef:.4f}, OR={OR:.4f}, p={p:.4f} {sig}")
        label_rows.append({
            'error_label': lbl,
            'coef': coef, 'OR': OR, 'p_value': p,
            'ci_lower': np.exp(model.conf_int().loc['reasoning_effort_num', 0]),
            'ci_upper': np.exp(model.conf_int().loc['reasoning_effort_num', 1]),
        })

    label_df = pd.DataFrame(label_rows)
    label_df.to_csv(os.path.join(output_dir, 'logistic_regression_error_labels.csv'), index=False)

    # ---------------------------
    # 6. Plots
    # ---------------------------
    plot_outcome_by_reasoning(df, output_dir)
    plot_error_labels_by_reasoning(df, output_dir)
    plot_error_labels_by_reasoning_and_prompt(df, output_dir)
    plot_category_heatmap(df, OUTCOME_COL, output_dir)
    plot_category_heatmap(df, 'incorrect', output_dir)

    # OR forest plot for error labels
    if not label_df.empty:
        label_df_sorted = label_df.sort_values('OR')
        fig, ax = plt.subplots(figsize=(7, 5))
        y_pos = range(len(label_df_sorted))
        ax.barh(y_pos, label_df_sorted['OR'] - 1,
                left=1,
                xerr=[label_df_sorted['OR'] - label_df_sorted['ci_lower'],
                      label_df_sorted['ci_upper'] - label_df_sorted['OR']],
                color=['#d73027' if or_ > 1 else '#4575b4' for or_ in label_df_sorted['OR']],
                edgecolor='black', height=0.5, capsize=4)
        ax.axvline(1, color='black', linewidth=1, linestyle='--')
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(label_df_sorted['error_label'])
        ax.set_xlabel('Odds Ratio (medium vs low reasoning)')
        ax.set_title('Effect of Reasoning Effort on Error Labels\n(controlling for prompt type and ambiguity)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'OR_forest_plot_error_labels.pdf'), dpi=300)
        plt.close()
        print("Saved: OR_forest_plot_error_labels.pdf")

    # ---------------------------
    # 7. Summary table for paper
    # ---------------------------
    summary_rows = []
    for effort in sorted(df[REASONING_COL].unique()):
        sub = df[df[REASONING_COL] == effort]
        row = {'reasoning_effort': effort, 'n': len(sub)}
        for col in [OUTCOME_COL, 'incorrect'] + error_labels:
            row[f'{col}_pct'] = round(sub[col].mean() * 100, 2)
        summary_rows.append(row)

    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(os.path.join(output_dir, 'summary_table_for_paper.csv'), index=False)
    print("\nSaved: summary_table_for_paper.csv")
    print(summary_table.to_string(index=False))

    print("\n✓ Analysis complete. All outputs saved to:", output_dir)



# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse reasoning effort effects on bias outcomes for GPT-OSS models"
    )
    parser.add_argument(
        "--output_dir", default="./reasoning_effort_results",
        help="Directory to save all outputs (default: ./reasoning_effort_results)"
    )
    parser.add_argument(
        "--base_dirs", nargs='+', default=None,
        help="Optional override: list of base dirs to process instead of GPT_OSS_BASE_DIRS"
    )
    args = parser.parse_args()
    main(args.output_dir, base_dirs=args.base_dirs)