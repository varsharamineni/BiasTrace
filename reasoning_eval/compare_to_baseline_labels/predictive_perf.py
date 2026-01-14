import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import minimize
import argparse

# ---------------------------
# Utils
# ---------------------------
def find_jsons_annotation(base_dirs, prompt_types=None):
    """
    Recursively find all JSONs under annotation folders for the given base directories.
    
    Corrected for your folder structure:
    outputs/qwen_full_8B_full_prompt/<annotation_type>/<bbq_category>/llm_eval_*.json
    """
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]

    all_files = []
    for base in base_dirs:
        if not os.path.exists(base):
            continue

        # List annotation folders (baseline_annotation, full_annotation, etc)
        annotation_folders = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        if prompt_types:
            annotation_folders = [p for p in annotation_folders if p in prompt_types]

        for annotation in annotation_folders:
            annotation_path = os.path.join(base, annotation)

            # Loop over bbq categories
            for category in os.listdir(annotation_path):
                category_path = os.path.join(annotation_path, category)
                if not os.path.isdir(category_path):
                    continue

                # Collect JSONs
                for root, _, files in os.walk(category_path):
                    for file in files:
                        if file.endswith(".json") and "llm_eval" in file:
                            all_files.append(os.path.join(root, file))

    return all_files

def parse_judge_output(r):
    score = 0
    errors = {}
    jo = r.get("judge_output", {})

    # Baseline-style
    if "score" in jo:
        score = jo["score"]
    # Bias-label
    elif "bias_label" in jo:
        score = jo["bias_label"]
    # Detailed error flags
    else:
        errors = {k: jo.get(k, 0) for k in ['group_assumption','bias_acknowledgement','meta_reflection',
                                            'outside_demo_knowledge','outside_topical_knowledge','unresolved',
                                            'overthinking','missing_logic']}
        score = sum(errors.values())

    return score, errors

def learn_weighted_errors(df, error_labels, outcome_col):
    X = df[error_labels].values.astype(float)
    y = df[outcome_col].values.astype(float)
    bias_idx = error_labels.index('bias_acknowledgement') if 'bias_acknowledgement' in error_labels else None

    def compute_weighted(weights):
        w = np.array(weights)
        weighted = X @ w
        if bias_idx is not None:
            weighted -= w[bias_idx]*X[:, bias_idx]
        return weighted

    def loss(weights):
        return -spearmanr(compute_weighted(weights), y)[0]

    res = minimize(loss, np.ones(X.shape[1]), method='BFGS')
    weighted_scores = compute_weighted(res.x)
    learned_weights = dict(zip(error_labels, res.x))
    return weighted_scores, learned_weights

def fit_logistic_outcome(outcome_name, predictors, df):
    y = df[outcome_name].astype(float)
    df_encoded = pd.get_dummies(df[['prompt_type','model','bbq_category']], drop_first=True)
    X = pd.concat([df[predictors], df_encoded], axis=1)
    X = sm.add_constant(X).astype(float).fillna(0)

    try:
        model = sm.Logit(y, X)
        result = model.fit(method='lbfgs', disp=False)
    except Exception as e:
        print(f"Model failed for {outcome_name} with predictors {predictors}: {e}")
        return None, None, None

    summary_df = pd.DataFrame({
        'coef': result.params,
        'std_err': result.bse,
        'z': result.tvalues,
        'pval': result.pvalues,
        'OR': np.exp(result.params)
    }).sort_values('coef', ascending=False)

    mcfadden_r2 = result.prsquared
    ll_null = result.llnull
    n_obs = result.nobs
    cox_snell_r2 = 1 - np.exp((ll_null - result.llf) * 2 / n_obs)
    nagelkerke_r2 = cox_snell_r2 / (1 - np.exp(ll_null * 2 / n_obs))

    print(f"\n=== Logistic Regression for {outcome_name} ===")
    print(summary_df)
    print(f"Pseudo-R²: McFadden={mcfadden_r2:.3f}, CoxSnell={cox_snell_r2:.3f}, Nagelkerke={nagelkerke_r2:.3f}\n")

    return result, summary_df, (mcfadden_r2, cox_snell_r2, nagelkerke_r2)

# ---------------------------
# Main function
# ---------------------------
def main(base_dirs, output_dir, models=None, prompt_types=None):
    os.makedirs(output_dir, exist_ok=True)

    # Collect JSONs
    json_files = find_jsons_annotation(base_dirs, prompt_types=prompt_types)
    print(f"Found {len(json_files)} JSON files")

    # Build DataFrame
    rows = []
    error_labels = ['group_assumption','bias_acknowledgement','meta_reflection',
                    'outside_demo_knowledge','outside_topical_knowledge','unresolved','overthinking','missing_logic']

    for file_path in json_files:
        with open(file_path) as f:
            data = json.load(f)
        category = data.get("metadata", {}).get("bbq_category", "unknown")
        for r in data.get("results", []):
            model_name = r.get('model','unknown')
            prompt_type = r.get('prompt_type','unknown')
            if models and model_name not in models:
                continue
            if prompt_types and prompt_type not in prompt_types:
                continue

            score, errors = parse_judge_output(r)
            row = {
                'sample_id': r['sample_id'],
                'bbq_category': category,
                'model': model_name,
                'prompt_type': prompt_type,
                'is_correct': int(r.get('is_correct', False)),
                'stereotype_aligned': int(r.get('stereotype_alignment', False)),
                'baseline_score': score,
                'num_errors': sum(errors.values()),
                **errors
            }

            # Ensure all error columns exist
            for lbl in error_labels:
                if lbl not in row:
                    row[lbl] = 0

            rows.append(row)

    df = pd.DataFrame(rows)

    # Derived metrics
    df['agg_errors'] = df[error_labels].sum(axis=1) - df['bias_acknowledgement']
    df['mean_errors'] = df[error_labels].mean(axis=1)
    df['error_or_stereo'] = ((~df['is_correct']) | df['stereotype_aligned']).astype(int)
    df['weighted_error'], _ = learn_weighted_errors(df, error_labels, 'is_correct')
    df['weighted_stereotype'], _ = learn_weighted_errors(df, error_labels, 'stereotype_aligned')
    df['weighted_error_or_stereo'], _ = learn_weighted_errors(df, error_labels, 'error_or_stereo')
    df['reasoning_length'] = df.get('model_reasoning', pd.Series(['']*len(df))).apply(lambda x: len(str(x).split()))

    # ---------------------------
    # Logistic regression
    # ---------------------------
    outcomes = ['is_correct','stereotype_aligned','error_or_stereo']
    predictor_sets = {'error_labels': error_labels, 'baseline': ['baseline_score']}

    model_results = {}
    summary_dfs = {}
    pseudo_r2_summary = {}

    for outcome in outcomes:
        for name, predictors in predictor_sets.items():
            result, summary_df, r2_vals = fit_logistic_outcome(outcome, predictors, df)
            if result is None: continue
            model_results[f"{name}_{outcome}"] = result
            summary_dfs[f"{name}_{outcome}"] = summary_df
            pseudo_r2_summary[f"{name}_{outcome}"] = r2_vals

    # ---------------------------
    # Save combined CSV
    # ---------------------------
    df.to_csv(os.path.join(output_dir,"combined_results.csv"), index=False)
    print(f"Saved combined results to {output_dir}/combined_results.csv")

    return df, summary_dfs, pseudo_r2_summary

# ---------------------------
# CLI
# ---------------------------
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Process BBQ JSON results for multiple folders and models.")
    parser.add_argument("--folders", required=True, nargs='+', help="Base folders containing JSONs")
    parser.add_argument("--output_dir", required=True, help="Output directory for combined CSV and plots")
    parser.add_argument("--models", nargs='*', help="Optional list of models to include")
    parser.add_argument("--prompt_types", nargs='*', help="Optional list of prompt_types to include")
    args = parser.parse_args()

    df, summary_dfs, pseudo_r2_summary = main(args.folders, args.output_dir, args.models, args.prompt_types)
