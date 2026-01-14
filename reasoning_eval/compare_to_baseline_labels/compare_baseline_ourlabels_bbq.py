#!/usr/bin/env python
import os
import glob
import json
import argparse
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Error labels
# ---------------------------
error_labels = ['group_assumption','bias_acknowledgement','meta_reflection',
                'outside_demo_knowledge','outside_topical_knowledge','unresolved',
                'overthinking','missing_logic']

# ---------------------------
# Utils
# ---------------------------
def find_jsons(base_dir):
    """Recursively find all llm_eval_*.json files in base_dir"""
    return [y for x in os.walk(base_dir) for y in glob.glob(os.path.join(x[0], 'llm_eval_*.json'))]

def parse_judge_output(r):
    """Extract score and detailed errors"""
    jo = r.get("judge_output", {})
    score = 0
    errors = {lbl: 0 for lbl in error_labels}

    if "score" in jo:
        score = jo["score"]
    elif "bias_label" in jo:
        score = jo["bias_label"]
    else:
        for lbl in error_labels:
            errors[lbl] = jo.get(lbl, 0)
        score = sum(errors.values())
    return score, errors

# ---------------------------
# Main
# ---------------------------
def main(baseline_folder, baseline_0_5_folder, full_annotation_folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def load_folder(folder, baseline_type=None):
        rows = []
        for f in find_jsons(folder):
            with open(f) as file:
                data = json.load(file)
            category = data.get("metadata", {}).get("bbq_category","unknown")
            for r in data.get("results", []):
                score, errors = parse_judge_output(r)
                row = {
                    'sample_id': r.get('sample_id'),
                    'bbq_category': category,
                    'model': r.get('model','unknown'),
                    'prompt_type': r.get('prompt_type','unknown'),
                    'is_correct': int(r.get('is_correct', False)),
                    'stereotype_aligned': int(r.get('stereotype_alignment', False)),
                    'incorrect_and_stereotype': int(r.get('incorrect_and_stereotype', False)),
                    'baseline_score': score if baseline_type else None,
                    'baseline_type': baseline_type
                }
                # Detailed errors for full_annotation
                if baseline_type is None:
                    row.update(errors)
                    row['agg_errors'] = sum(errors.values()) - errors['bias_acknowledgement']
                rows.append(row)
        return pd.DataFrame(rows)

    # ---------------------------
    # Load all folders
    # ---------------------------
    df_baseline = load_folder(baseline_folder, baseline_type='baseline')
    df_baseline_0_5 = load_folder(baseline_0_5_folder, baseline_type='baseline_0-5')
    df_full = load_folder(full_annotation_folder, baseline_type=None)

    # ---------------------------
    # Pivot baselines
    # ---------------------------
    baseline_wide = pd.concat([df_baseline, df_baseline_0_5])
    baseline_wide = baseline_wide.pivot(index='sample_id', columns='baseline_type', values='baseline_score')

    # ---------------------------
    # Merge with full annotation (only keep samples with both baselines)
    # ---------------------------
    final_df = df_full.set_index('sample_id').join(baseline_wide, how='inner')
    print(f"Final df shape (samples with both baselines): {final_df.shape}")

    # Ensure all error columns exist
    for lbl in error_labels:
        if lbl not in final_df.columns:
            final_df[lbl] = 0
    if 'agg_errors' not in final_df.columns:
        final_df['agg_errors'] = final_df[error_labels].sum(axis=1) - final_df['bias_acknowledgement']

    # ---------------------------
    # Save combined CSV
    # ---------------------------
    out_csv = os.path.join(output_dir, "combined_results.csv")
    final_df.to_csv(out_csv)
    print(f"Saved combined results: {out_csv}")

    # ---------------------------
    # Correlations with outcomes
    # ---------------------------
    corr_results = []
    metrics = ['baseline','baseline_0-5','agg_errors'] + error_labels
    outcomes = ['is_correct','stereotype_aligned','incorrect_and_stereotype']

    for metric in metrics:
        if metric not in final_df.columns:
            continue
        for outcome in outcomes:
            x = final_df[metric]
            y = final_df[outcome]
            valid = x.notna() & y.notna()
            if valid.sum() == 0:
                continue
            r, p = spearmanr(x[valid], y[valid])
            corr_results.append({
                'metric': metric,
                'outcome': outcome,
                'spearman_r': r,
                'p_value': p
            })

    corr_df = pd.DataFrame(corr_results)
    out_corr = os.path.join(output_dir, "correlations_outcomes.csv")
    corr_df.to_csv(out_corr, index=False)
    print(f"Saved correlations with outcomes: {out_corr}")

    # ---------------------------
    # Full correlations between all labels/metrics
    # ---------------------------
    numeric_cols = ['baseline','baseline_0-5','agg_errors'] + error_labels + outcomes
    present_cols = [c for c in numeric_cols if c in final_df.columns]
    corr_matrix = final_df[present_cols].corr(method='spearman')
    out_corr_matrix = os.path.join(output_dir, "correlations_matrix.csv")
    corr_matrix.to_csv(out_corr_matrix)
    print(f"Saved full correlation matrix: {out_corr_matrix}")

    # ---------------------------
    # Heatmaps
    # ---------------------------
    plt.figure(figsize=(10,6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title("Spearman Correlation Matrix (all metrics and outcomes)")
    plt.tight_layout()
    out_fig = os.path.join(output_dir, "correlation_matrix_heatmap.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved correlation matrix heatmap: {out_fig}")

    return final_df, corr_df, corr_matrix

# ---------------------------
# CLI
# ---------------------------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_folder", required=True)
    parser.add_argument("--baseline_0_5_folder", required=True)
    parser.add_argument("--full_annotation_folder", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    main(args.baseline_folder, args.baseline_0_5_folder, args.full_annotation_folder, args.output_dir)
