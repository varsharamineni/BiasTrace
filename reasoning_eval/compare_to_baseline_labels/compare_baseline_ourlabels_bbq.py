#!/usr/bin/env python
import os
import glob
import json
import argparse
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Error labels
# ---------------------------
error_labels = ['group_assumption','bias_acknowledgement','meta_reflection',
                'outside_demo_knowledge','outside_topical_knowledge',
                'overthinking']

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

def load_folder(folder, baseline_type=None, parent_folder_name=None):
    """Load all JSON results in a folder"""

        # Coefficients from the first logistic regression (p < 0.05)
    weighted_error_labels = {
        'overthinking': 1.6833,
        'group_assumption': 0.2675,
        'bias_acknowledgement': 1.2012,
        'meta_reflection': 0.4392,
        'outside_demo_knowledge': 0.4775,
        'outside_topical_knowledge': 0.1941
    }


    rows = []
    categories_seen = set()
    for f in find_jsons(folder):
        with open(f) as file:
            data = json.load(file)
        category = data.get("metadata", {}).get("bbq_category","unknown")
        categories_seen.add(category)
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
                'baseline_type': baseline_type,
                'parent_folder': parent_folder_name
            }
            # Detailed errors for full_annotation
            if baseline_type is None:
                row.update(errors)
                row['agg_errors'] = sum(errors.values()) 
                row['agg_errors_minus'] = sum(errors.values()) - errors['bias_acknowledgement'] 
                row['weighted_agg_errors'] = sum(errors[lbl] * weighted_error_labels[lbl] for lbl in error_labels)
                row['at_least_one_error'] = int(any(errors[lbl] == 1 for lbl in error_labels))
            rows.append(row)
    return pd.DataFrame(rows), categories_seen

def run_logistic(df, predictors, outcome='incorrect_and_stereotype'):
    df_clean = df.dropna(subset=predictors + [outcome])
    X = df_clean[predictors]
    X = sm.add_constant(X)
    y = df_clean[outcome]
    model = sm.Logit(y, X).fit(disp=False)
    pseudo_r2 = 1 - model.llf / model.llnull
    return model, pseudo_r2

def coverage_analysis(df, baseline_col='baseline', error_cols=error_labels, output_dir=None):
    coverage_results = []
    for val in df[baseline_col].dropna().unique():
        subset = df[df[baseline_col] == val]
        unique_patterns = subset[error_cols].drop_duplicates()
        n_patterns = unique_patterns.shape[0]
        total_samples = subset.shape[0]
        coverage_results.append({
            'baseline_score': val,
            'n_samples': total_samples,
            'n_unique_error_combos': n_patterns,
            'fraction_unique': n_patterns / total_samples
        })
    coverage_df = pd.DataFrame(coverage_results)
    if output_dir:
        coverage_df.to_csv(os.path.join(output_dir, f'coverage_{baseline_col}.csv'), index=False)
        print(f"Saved coverage summary for {baseline_col}")
    return coverage_df

def plot_coverage(coverage_df, baseline_col='baseline', output_dir=None):
    plt.figure(figsize=(8,5))
    sns.barplot(x='baseline_score', y='n_unique_error_combos', data=coverage_df)
    plt.xlabel(f"{baseline_col} score")
    plt.ylabel("Number of unique error combinations")
    plt.title("Coverage of error combinations per baseline score")
    plt.tight_layout()
    if output_dir:
        out_fig = os.path.join(output_dir, f'coverage_{baseline_col}.pdf')
        plt.savefig(out_fig, dpi=300)
        print(f"Saved coverage figure: {out_fig}")
    plt.close()

def plot_coverage_stacked(df, baseline_col='baseline', error_cols=error_labels, output_dir=None):
    """
    Plot a stacked bar chart showing the breakdown of each error type per baseline score.
    """
    # Aggregate counts per baseline score
    agg_counts = df.groupby(baseline_col)[error_cols].sum()

    # Plot
    plt.figure(figsize=(10,6))
    agg_counts.plot(kind='bar', stacked=True, colormap='tab20', edgecolor='black')

    plt.xlabel(f"{baseline_col} score")
    plt.ylabel("Number of errors")
    plt.title(f"Breakdown of error types per {baseline_col} score")
    plt.xticks(rotation=0)
    plt.legend(title="Error Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if output_dir:
        out_fig = os.path.join(output_dir, f'stacked_errors_{baseline_col}.pdf')
        plt.savefig(out_fig, dpi=300)
        print(f"Saved stacked error figure: {out_fig}")
    plt.close()


def get_subfolders(parent_folder):
    """Return dict with keys: baseline, baseline_0-5, full"""
    mapping = {}
    for name in os.listdir(parent_folder):
        full_path = os.path.join(parent_folder, name)
        if os.path.isdir(full_path):
            if "baseline_0-5" in name:
                mapping["baseline_0-5"] = full_path
            elif "baseline" in name:
                mapping["baseline"] = full_path
            elif "full" in name:
                mapping["full"] = full_path
    return mapping

def plot_correlation_heatmap(df=None, numeric_cols=None, corr_matrix=None, output_dir=None, 
                             output_name="correlation_heatmap.pdf", method="spearman"):
    """
    Plots a correlation heatmap for selected numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numeric columns (optional if corr_matrix is provided)
    numeric_cols : list[str]
        List of numeric columns to include in correlation (required if df is provided)
    corr_matrix : pd.DataFrame
        Precomputed correlation matrix (optional)
    output_dir : str
        Directory to save CSV and heatmap
    output_name : str
        Filename for heatmap PDF
    method : str
        Correlation method: 'spearman' or 'pearson'
    """
    # Compute correlation if not provided
    if corr_matrix is None:
        if df is None or numeric_cols is None:
            raise ValueError("Must provide either df + numeric_cols or a precomputed corr_matrix")
        corr_matrix = df[numeric_cols].corr(method=method)
    
    # Plot heatmap
    plt.figure(figsize=(12,10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(f"{method.capitalize()} correlation between baseline scores, error labels, and outcome")
    plt.tight_layout()

    # Save CSV table
    if output_dir:
        out_csv = os.path.join(output_dir, f"correlation_table_{method}.csv")
        corr_matrix.to_csv(out_csv)
        print(f"Saved correlation table ({method}): {out_csv}")

    # Print nicely in console
    print(f"\n{method.capitalize()} Correlation Table:")
    print(corr_matrix.round(3))

    # Save heatmap
    if output_dir:
        out_fig = os.path.join(output_dir, output_name)
        plt.savefig(out_fig, dpi=300)
        print(f"Saved correlation heatmap ({method}): {out_fig}")
    plt.close()

# ---------------------------
# Main
# ---------------------------
def main(parent_folders, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df_baseline_list = []
    df_baseline_0_5_list = []
    df_full_list = []

    # Load each parent folder
    for pf in parent_folders:
        subfolders = get_subfolders(pf)
        if not all(k in subfolders for k in ["baseline", "baseline_0-5", "full"]):
            print(f"Skipping {pf}, missing required subfolders")
            continue

        df_b, cats_b = load_folder(subfolders["baseline"], "baseline", pf)
        df_b05, cats_b05 = load_folder(subfolders["baseline_0-5"], "baseline_0-5", pf)
        df_f, cats_f = load_folder(subfolders["full"], None, pf)

        # Only keep categories that exist in all three subfolders
        common_categories = cats_b & cats_b05 & cats_f
        print(f"{pf}: keeping categories {sorted(common_categories)}")

        df_b = df_b[df_b['bbq_category'].isin(common_categories)]
        df_b05 = df_b05[df_b05['bbq_category'].isin(common_categories)]
        df_f = df_f[df_f['bbq_category'].isin(common_categories)]

        df_baseline_list.append(df_b)
        df_baseline_0_5_list.append(df_b05)
        df_full_list.append(df_f)

    # Concatenate all parent folders
    df_baseline = pd.concat(df_baseline_list, ignore_index=True).drop_duplicates(subset=['sample_id','baseline_type','model','prompt_type','parent_folder'])
    df_baseline_0_5 = pd.concat(df_baseline_0_5_list, ignore_index=True).drop_duplicates(subset=['sample_id','baseline_type','model','prompt_type','parent_folder'])
    df_full = pd.concat(df_full_list, ignore_index=True).drop_duplicates(subset=['sample_id','model','prompt_type','parent_folder'])

    # Pivot baselines using multi-index
    baseline_wide = pd.concat([df_baseline, df_baseline_0_5])
    baseline_wide = baseline_wide.pivot_table(
        index=['sample_id','model','prompt_type','parent_folder'],
        columns='baseline_type',
        values='baseline_score',
        aggfunc='first'  # safe if duplicates exist
    ).reset_index()

    # Merge with full annotation
    final_df = df_full.set_index(['sample_id','model','prompt_type','parent_folder']).join(
        baseline_wide.set_index(['sample_id','model','prompt_type','parent_folder']),
        how='inner'
    ).reset_index()

    print(f"Final df shape: {final_df.shape}")

    # Ensure all error columns exist
    for lbl in error_labels:
        if lbl not in final_df.columns:
            final_df[lbl] = 0

    # Save combined CSV
    final_df.to_csv(os.path.join(output_dir, "combined_results.csv"), index=False)
    print("Saved combined results")

    # ---------------------------
    # Logistic regression comparisons
    # ---------------------------
    final_df['incorrect'] = 1 - final_df['is_correct']
    outcome_col = 'incorrect'

    predictor_sets = {
        'Error Labels': error_labels,
        'Baseline 0/1': ['baseline'],
        'Baseline 0-0.5': ['baseline_0-5']
    }

    logit_summaries = []
    for name, predictors in predictor_sets.items():
        if all(col in final_df.columns for col in predictors):
            model, pseudo_r2 = run_logistic(final_df, predictors, outcome=outcome_col)
            for var in predictors:
                coef = model.params[var]
                pval = model.pvalues[var]
                logit_summaries.append({
                    'predictor_set': name,
                    'variable': var,
                    'coef': coef,
                    'p_value': pval,
                    'pseudo_r2': pseudo_r2
                })
    logit_df = pd.DataFrame(logit_summaries)
    logit_df.to_csv(os.path.join(output_dir, "logistic_regression_comparison.csv"), index=False)
    print("Saved logistic regression comparison")

    final_df['incorrect'] = 1 - final_df['is_correct']

    numeric_cols = error_labels + ['baseline', 'baseline_0-5', 'incorrect', 'incorrect_and_stereotype', 'agg_errors', 'agg_errors_minus', 'weighted_agg_errors', 'at_least_one_error']

    # ---------------------------
    # Coverage analysis
    # ---------------------------
    for baseline_col in ['baseline','baseline_0-5']:
        if baseline_col in final_df.columns:
            cov_df = coverage_analysis(final_df, baseline_col=baseline_col, error_cols=error_labels, output_dir=output_dir)
            plot_coverage(cov_df, baseline_col=baseline_col, output_dir=output_dir)
            plot_coverage_stacked(final_df, baseline_col=baseline_col, error_cols=error_labels, output_dir=output_dir)
            plot_correlation_heatmap(df=final_df, numeric_cols=numeric_cols, output_dir=output_dir,
                         output_name="correlation_heatmap_spearman.pdf", method="spearman")
            plot_correlation_heatmap(df=final_df, numeric_cols=numeric_cols, output_dir=output_dir,
                         output_name="correlation_heatmap_pearson.pdf", method="pearson")



    print("Analysis complete")
    return final_df, logit_df

# ---------------------------
# CLI
# ---------------------------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_folders", nargs='+', required=True,
                        help="List of parent folders, each containing baseline, baseline_0-5, full subfolders")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    main(args.parent_folders, args.output_dir)
