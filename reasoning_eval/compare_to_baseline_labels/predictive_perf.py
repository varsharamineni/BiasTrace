import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import minimize

# ---------------------------
# File paths
# ---------------------------
baseline_file = "reasoning_eval/llm_judge_samples/test_set/baseline/llm_eval_claude-opus-4-1-20250805_baseline_temp0.6_seed42_max_tokens2048.json"
labels_file   = "reasoning_eval/llm_judge_samples/test_set/our_labels/llm_eval_claude-opus-4-1-20250805_detailed_example_clarification_opt_temp0.6_seed42_max_tokens2048_reasoning.json"
val_file      = "reasoning_eval/ground_truth_samples/test_set.json"
output_dir    = "reasoning_eval/compare_baseline_labels/"

# ---------------------------
# Load JSONs
# ---------------------------
with open(baseline_file) as f: baseline_data = json.load(f)
with open(labels_file) as f: label_data = json.load(f)
with open(val_file) as f: val_data = json.load(f)

# ---------------------------
# Prepare DataFrame
# ---------------------------
baseline_scores = {r['sample_id']: r['judge_output']['score'] for r in baseline_data['results']}
error_labels = ['group_assumption','bias_acknowledgement','meta_reflection','outside_demo_knowledge',
                'outside_topical_knowledge','unresolved','overthinking','missing_logic']

rows = []
for r in val_data:
    sid = r['sample_id']
    row = {lbl: r[lbl] for lbl in error_labels}
    row['sample_id'] = sid
    row['is_correct'] = r['is_correct']
    row['is_error'] = int(not r['is_correct'])
    row['stereotype_aligned'] = int(r['stereotype_aligned'])
    row['error_or_stereo'] = int((not r['is_correct']) or r['stereotype_aligned'])
    row['num_errors'] = sum(row[lbl] for lbl in error_labels)
    row['reasoning_length'] = len(r.get('model_reasoning','').split())
    row['baseline_score'] = baseline_scores.get(sid, 0)
    row['prompt_type'] = r['prompt_type']
    row['model'] = r['model']
    row['bbq_category'] = r['bbq_category']
    rows.append(row)
df = pd.DataFrame(rows)
df['agg_errors'] = df[error_labels].sum(axis=1) - df['bias_acknowledgement']
df['mean_errors'] = df[error_labels].mean(axis=1)

# ---------------------------
# Logistic regression
# ---------------------------
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
    
    # Pseudo-R²
    mcfadden_r2 = result.prsquared
    ll_null = result.llnull
    n_obs = result.nobs
    cox_snell_r2 = 1 - np.exp((ll_null - result.llf) * 2 / n_obs)
    nagelkerke_r2 = cox_snell_r2 / (1 - np.exp(ll_null * 2 / n_obs))
    
    # Print results
    print(f"\n=== Logistic Regression for {outcome_name} ===")
    print(summary_df)
    print(f"Pseudo-R²: McFadden={mcfadden_r2:.3f}, CoxSnell={cox_snell_r2:.3f}, Nagelkerke={nagelkerke_r2:.3f}\n")
    
    return result, summary_df, (mcfadden_r2, cox_snell_r2, nagelkerke_r2)

# ---------------------------
# Combined logistic regression plot
# ---------------------------
def plot_combined_logit(models_dict, predictors, output_path, title):
    """
    Plot combined logistic regression coefficients including all covariates
    for multiple models side-by-side, with significance markers.
    
    Parameters:
    - models_dict: dict of {model_name: summary_df}, where summary_df contains coef, pval etc.
    - output_path: file path to save the plot
    - title: figure title
    """
    n = len(models_dict)
    fig, axes = plt.subplots(1, n, figsize=(8*n, max(6, len(next(iter(models_dict.values())))//2)), sharey=True)
    if n == 1: axes = [axes]

    for ax, (name, summary_df) in zip(axes, models_dict.items()):
        df_plot = summary_df.copy()
        # Sort by coefficient magnitude for visual clarity
        df_plot = df_plot.sort_values('coef', ascending=True)
        colors = ['red' if p<0.05 else 'grey' for p in df_plot['pval']]
        sns.barplot(x='coef', y=df_plot.index, data=df_plot, palette=colors, ax=ax, orient='h', hue=None)
        
        # Add significance stars
        for i, row in enumerate(df_plot.itertuples()):
            sig = '***' if row.pval<0.001 else '**' if row.pval<0.01 else '*' if row.pval<0.05 else ''
            ax.text(row.coef, i, f" {sig}", va='center')
        
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("Coefficient")
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Combined regression plot saved: {output_path}")

# ---------------------------
# Weighted errors
# ---------------------------
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

df['weighted_error'], _ = learn_weighted_errors(df, error_labels, 'is_error')
df['weighted_stereotype'], _ = learn_weighted_errors(df, error_labels, 'stereotype_aligned')
df['weighted_error_or_stereo'], _ = learn_weighted_errors(df, error_labels, 'error_or_stereo')

# ---------------------------
# Fit models and print
# ---------------------------
outcomes = ['is_error','stereotype_aligned']
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

# Combined regression plots
for outcome in outcomes:
    plot_combined_logit({f"error_labels": summary_dfs[f"error_labels_{outcome}"],
                         f"baseline": summary_dfs[f"baseline_{outcome}"]},
                        predictors=error_labels,
                        output_path=f"{output_dir}combined_regression_{outcome}.png",
                        title=f"Regression Coefficients for {outcome}")

# ---------------------------
# Correlation analysis
# ---------------------------
predictors_all = error_labels + ['baseline_score','num_errors','mean_errors','agg_errors',
                                 'weighted_error','weighted_stereotype','weighted_error_or_stereo','reasoning_length']

outcomes.append('error_or_stereo')

corr_data = []
for outcome in outcomes:
    print(f"\n=== Spearman and Pearson Correlations for {outcome} ===")
    for metric in predictors_all:
        r, p = spearmanr(df[metric], df[outcome])
        r1, p1 = pearsonr(df[metric], df[outcome])  
        print(f"{metric:25s} | r={r:.3f}, p={p:.3f}, Pearson_r={r1:.3f}, Pearson_p={p1:.3f}")
        corr_data.append({'Outcome': outcome, 'Metric': metric, 'Spearman_r': r, 'p_value': p, 'Pearson_r': r1, 'Pearson_p': p1})
corr_df = pd.DataFrame(corr_data)

# ---------------------------
# Combined correlation plot
# ---------------------------
def plot_combined_correlations(corr_df, output_path, title):
    outcomes = corr_df['Outcome'].unique()
    n = len(outcomes)
    fig, axes = plt.subplots(1, n, figsize=(6*n,6), sharey=True)
    if n==1: axes=[axes]
    
    for ax, outcome in zip(axes, outcomes):
        df_plot = corr_df[corr_df['Outcome']==outcome].sort_values('Spearman_r')
        colors = ['red' if p<0.05 else 'grey' for p in df_plot['p_value']]
        sns.barplot(x='Spearman_r', y='Metric', data=df_plot, palette=colors, ax=ax, orient='h', hue=None)
        for i, row in enumerate(df_plot.itertuples()):
            ax.text(row.Spearman_r + 0.01, i, '*' if row.p_value<0.05 else '', va='center')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(outcome)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Combined correlation plot saved: {output_path}")

plot_combined_correlations(corr_df, f"{output_dir}combined_correlations.png", "Combined Spearman Correlations")



def plot_variance_explained(pseudo_r2_summary, output_path):
    """
    Plots variance explained (pseudo-R²) for all models and outcomes in a single figure,
    allowing side-by-side comparison.

    Parameters
    ----------
    pseudo_r2_summary : dict
        Keys: 'model_outcome' strings (e.g., 'baseline_is_error')
        Values: tuple of (McFadden, CoxSnell, Nagelkerke)
    output_path : str
        Path to save the plot
    """
    # Convert to DataFrame
    df_r2 = pd.DataFrame(pseudo_r2_summary, index=['McFadden','CoxSnell','Nagelkerke']).T
    df_r2.reset_index(inplace=True)
    df_r2 = df_r2.melt(id_vars='index', var_name='PseudoR2', value_name='Value')

    # Split 'index' properly: last part is Outcome, rest is Model
    split_df = df_r2['index'].str.rsplit('_', n=1, expand=True)
    df_r2['Model'] = split_df[0]
    df_r2['Outcome'] = split_df[1]

    # Plot: grouped bar plot with Outcome on x-axis, Pseudo-R² as hue, multiple models side by side
    plt.figure(figsize=(12,6))
    sns.barplot(
        x='Outcome',
        y='Value',
        hue='PseudoR2',
        data=df_r2,
        palette='Set2',
        ci=None
    )

    # Add model labels as annotations
    for i, row in df_r2.iterrows():
        plt.text(
            x=i % len(df_r2['Outcome'].unique()),  # approximate x-position for outcome
            y=row['Value'] + 0.02,                # small offset above bar
            s=row['Model'],
            rotation=90,
            fontsize=8,
            ha='center'
        )

    plt.title("Variance Explained (Pseudo-R²) Across Models and Outcomes")
    plt.ylabel("Pseudo-R²")
    plt.ylim(0, 1)
    plt.xlabel("Outcome")
    plt.legend(title='Pseudo-R²')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Variance explained plot saved: {output_path}")

plot_variance_explained(pseudo_r2_summary, f"{output_dir}variance_explained.png")
