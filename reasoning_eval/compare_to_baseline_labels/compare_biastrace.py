#!/usr/bin/env python
"""
Rule-based BiasTrace baseline: correlate weighted BiasTrace features
(using RQ1 logistic regression coefficients) with incorrect_and_stereotype,
broken down by model + prompt + reasoning-level setting.

Includes:
  - Weighted / unweighted / any-flagged rule scores
  - Three RQ1-identified high-risk combinations
  - Combined "any high-risk combo fired" indicator

This addresses the reviewer's ask: does the LLM-judge evaluation prompt
add value beyond a rule-based combination of the RQ1 behaviors?
"""
import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# ---------------------------
# Config
# ---------------------------
ERROR_LABELS = [
    'group_assumption',
    'bias_acknowledgement',
    'meta_reflection',
    'outside_demo_knowledge',
    'outside_topical_knowledge',
    'overthinking',
]

RQ1_WEIGHTS = {
    'overthinking': 1.6833,
    'group_assumption': 0.2675,
    'bias_acknowledgement': 1.2012,
    'meta_reflection': 0.4392,
    'outside_demo_knowledge': 0.4775,
    'outside_topical_knowledge': 0.1941,
}

# ---------------------------
# Helpers
# ---------------------------
def find_jsons(base_dir):
    return [y for x in os.walk(base_dir)
            for y in glob.glob(os.path.join(x[0], 'llm_eval_*.json'))]

def parse_judge_output(r):
    jo = r.get("judge_output", {})
    errors = {lbl: 0 for lbl in ERROR_LABELS}
    if not isinstance(jo, dict):
        return errors
    for lbl in ERROR_LABELS:
        errors[lbl] = jo.get(lbl, 0)
    return errors

def model_from_parent_folder(parent_folder_name):
    pf = parent_folder_name.lower()

    if "full_prompt" in pf:
        prompt = "full"
    elif "simple_prompt" in pf:
        prompt = "simple"
    else:
        prompt = "unknown"

    if "qwen" in pf:
        if "14b" in pf:
            base = "Qwen3-14B"
        elif "8b" in pf:
            base = "Qwen3-8B"
        else:
            base = "Qwen3-unknown"
        return f"{base}_{prompt}"

    if "gpt-oss" in pf:
        if "120b" in pf:
            base = "gpt-oss-120b"
        elif "20b" in pf:
            base = "gpt-oss-20b"
        else:
            base = "gpt-oss-unknown"

        if "low_reasoning" in pf:
            reasoning = "low"
        elif "medium_reasoning" in pf:
            reasoning = "medium"
        elif "high_reasoning" in pf:
            reasoning = "high"
        else:
            reasoning = "unknown"

        return f"{base}_{prompt}_{reasoning}"

    return f"unknown_{prompt}"

def prompt_from_parent_folder(parent_folder_name):
    pf = parent_folder_name.lower()
    if "full_prompt" in pf:
        return "full_prompt"
    elif "simple_prompt" in pf:
        return "simple_prompt"
    return "unknown"

def get_full_subfolder(parent_folder):
    for name in os.listdir(parent_folder):
        full_path = os.path.join(parent_folder, name)
        if os.path.isdir(full_path) and "full" in name and "prompt" not in name:
            return full_path
    return None

# ---------------------------
# Loading
# ---------------------------
def load_full_annotations(folder, parent_folder_name):
    model = model_from_parent_folder(parent_folder_name)
    prompt_type = prompt_from_parent_folder(parent_folder_name)

    rows = []
    for f in find_jsons(folder):
        with open(f) as file:
            data = json.load(file)
        category = data.get("metadata", {}).get("bbq_category", "unknown")
        for r in data.get("results", []):
            errors = parse_judge_output(r)
            row = {
                'sample_id': r.get('sample_id'),
                'bbq_category': category,
                'model': model,
                'model_raw': r.get('model', 'unknown'),
                'ambiguous': int(r.get('ambiguous', False)),
                'prompt_type': prompt_type,
                'is_correct': int(r.get('is_correct', False)),
                'incorrect_and_stereotype': int(r.get('incorrect_and_stereotype', False)),
                'parent_folder': parent_folder_name,
            }
            row.update(errors)
            rows.append(row)
    return pd.DataFrame(rows)

# ---------------------------
# Rule-based scores
# ---------------------------
def compute_rule_based_scores(df):
    """Add rule-based BiasTrace scores.

    High-risk combinations from RQ1 pathway analysis:
      HRC1: overthinking + outside_demo_knowledge
      HRC2: group_assumption + outside_demo_knowledge
      HRC3: overthinking + meta_reflection + (outside_demo_knowledge OR outside_topical_knowledge)
    """
    df['rule_weighted'] = df[ERROR_LABELS].apply(
        lambda row: sum(row[lbl] * RQ1_WEIGHTS[lbl] for lbl in ERROR_LABELS), axis=1
    )
    df['rule_count'] = df[ERROR_LABELS].sum(axis=1)
    df['rule_any'] = (df[ERROR_LABELS].sum(axis=1) > 0).astype(int)

    df['hrc1_over_demo'] = (
        (df['overthinking'] == 1) &
        (df['outside_demo_knowledge'] == 1)
    ).astype(int)

    df['hrc2_group_demo'] = (
        (df['group_assumption'] == 1) &
        (df['outside_demo_knowledge'] == 1)
    ).astype(int)

    df['hrc3_over_meta_knowledge'] = (
        (df['overthinking'] == 1) &
        (df['meta_reflection'] == 1) &
        ((df['outside_demo_knowledge'] == 1) | (df['outside_topical_knowledge'] == 1))
    ).astype(int)

    df['rule_any_hrc'] = (
        (df['hrc1_over_demo'] == 1) |
        (df['hrc2_group_demo'] == 1) |
        (df['hrc3_over_meta_knowledge'] == 1)
    ).astype(int)

    df['rule_hrc_count'] = (
        df['hrc1_over_demo'] + df['hrc2_group_demo'] + df['hrc3_over_meta_knowledge']
    )

    return df

# ---------------------------
# Correlation analysis
# ---------------------------
def correlate_scores(df, score_cols, outcome_col='incorrect_and_stereotype'):
    results = []
    for col in score_cols:
        sub = df.dropna(subset=[col, outcome_col])
        if sub.empty or sub[col].nunique() < 2 or sub[outcome_col].nunique() < 2:
            results.append({
                'score': col, 'n': len(sub),
                'pearson_r': np.nan, 'pearson_p': np.nan,
                'spearman_r': np.nan, 'spearman_p': np.nan,
            })
            continue
        pear_r, pear_p = pearsonr(sub[col], sub[outcome_col])
        spear_r, spear_p = spearmanr(sub[col], sub[outcome_col])
        results.append({
            'score': col,
            'n': len(sub),
            'pearson_r': pear_r,
            'pearson_p': pear_p,
            'spearman_r': spear_r,
            'spearman_p': spear_p,
        })
    return pd.DataFrame(results)

# ---------------------------
# Main
# ---------------------------
def main(parent_folders, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    dfs = []
    for pf in parent_folders:
        full_sub = get_full_subfolder(pf)
        if full_sub is None:
            print(f"Skipping {pf}: no 'full' subfolder found")
            continue
        df = load_full_annotations(full_sub, pf)
        print(f"  Loaded {len(df)} rows from {pf} -> model='{df['model'].iloc[0] if len(df) else 'N/A'}'")
        dfs.append(df)

    if not dfs:
        print("No data loaded.")
        return

    df_all = pd.concat(dfs, ignore_index=True).drop_duplicates(
        subset=['sample_id', 'model', 'prompt_type', 'parent_folder']
    )
    print(f"\nTotal annotated traces: {len(df_all)}")
    print(f"Models (normalized): {sorted(df_all['model'].unique().tolist())}")

    for lbl in ERROR_LABELS:
        df_all[lbl] = df_all[lbl].fillna(0).astype(int)

    df_all = compute_rule_based_scores(df_all)

    score_cols = [
        'rule_weighted',
        'rule_count',
        'rule_any',
        'hrc1_over_demo',
        'hrc2_group_demo',
        'hrc3_over_meta_knowledge',
        'rule_any_hrc',
        'rule_hrc_count',
    ]

    # Overall
    overall_corr = correlate_scores(df_all, score_cols)
    overall_corr['model'] = 'ALL'
    overall_corr['score_type'] = 'rule_based'
    print("\n=== Overall correlations (all models pooled) ===")
    print(overall_corr.to_string(index=False))

    # Per-model
    per_model = []
    for model in sorted(df_all['model'].unique()):
        sub = df_all[df_all['model'] == model].copy()

        corr = correlate_scores(sub, score_cols)
        corr['model'] = model
        corr['score_type'] = 'rule_based'
        per_model.append(corr)

        behavior_corr = correlate_scores(sub, ERROR_LABELS)
        behavior_corr['model'] = model
        behavior_corr['score_type'] = 'individual_behavior'
        per_model.append(behavior_corr)

    per_model_df = pd.concat(per_model, ignore_index=True)
    print("\n=== Per-model correlations ===")
    print(per_model_df.to_string(index=False))

    # Save
    combined = pd.concat([overall_corr, per_model_df], ignore_index=True)
    combined = combined[['model', 'score_type', 'score', 'n',
                         'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p']]
    out_path = os.path.join(output_dir, "rule_based_correlations.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Also save HRC firing rates per model for context
    hrc_cols = ['hrc1_over_demo', 'hrc2_group_demo', 'hrc3_over_meta_knowledge', 'rule_any_hrc']
    hrc_rates = df_all.groupby('model')[hrc_cols].mean().reset_index()
    hrc_rates_path = os.path.join(output_dir, "hrc_firing_rates_per_model.csv")
    hrc_rates.to_csv(hrc_rates_path, index=False)
    print(f"Saved: {hrc_rates_path}")
    print("\n=== High-risk combo firing rates per model ===")
    print(hrc_rates.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_folders", nargs='+', required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    main(args.parent_folders, args.output_dir)