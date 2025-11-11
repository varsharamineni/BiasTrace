import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

try:
    from firthlogist import FirthLogisticRegression
    has_firth = True
except ImportError:
    has_firth = False
    print("Warning: firthlogist not installed. Firth regression skipped.")

# --- Load and clean data ---
INPUT_CSV = "reasoning_eval/ground_truth_samples/sample_traces_full_annotated.csv"
df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip().str.lower()

bool_cols = ["ambiguous", "is_correct", "stereotype_aligned"]
error_features = [
    "group_assumption", "bias_acknowledgement", "meta_reflection",
    "outside_demo_knowledge", "outside_topical_knowledge",
    "unresolved", "overthinking", "missing_logic"
]
covariates = ["model", "prompt_type"]

cols_to_check = bool_cols + error_features + covariates
df = df.dropna(subset=cols_to_check)

for col in bool_cols:
    df[col] = df[col].astype(bool)

# --- Encode categorical covariates ---
df[covariates] = df[covariates].astype(str)
covariate_dummies = pd.get_dummies(df[covariates], drop_first=True, prefix=covariates)

# --- Feature matrices ---
X_full = pd.concat([
    df[["ambiguous"]].astype(int),
    df[error_features].astype(int),
    covariate_dummies
], axis=1)
y = df["is_correct"].astype(int)

# --- Drop highly correlated features ---
def reduce_features(X, threshold=0.4):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold) and not col.startswith(("ambiguous", "model_", "prompt_type_"))]
    return X.drop(columns=to_drop), to_drop

X_reduced, dropped_features = reduce_features(X_full)
print(f"Dropped highly correlated features: {dropped_features}")

# --- Helper: pseudo-R² ---
def pseudo_r2(model):
    llf = model.llf
    llnull = model.llnull
    return 1 - llf/llnull if llnull != 0 else np.nan

# --- Fit functions ---
def fit_mle(X, y):
    X_sm = sm.add_constant(X)
    model = sm.Logit(y, X_sm)
    res = model.fit(disp=False)
    return res

def fit_elasticnet(X, y):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=5, max_iter=5000
        ))
    ])
    pipe.fit(X, y)
    return pipe.named_steps["model"]

def fit_mixed(X, y, group):
    X_sm = sm.add_constant(X)
    model = MixedLM(endog=y, exog=X_sm, groups=group)
    result = model.fit(reml=False)
    return result

# --- Fit models ---
print("\n=== Fitting models ===")
res_full = fit_mle(X_full, y)
res_red = fit_mle(X_reduced, y)

en_full = fit_elasticnet(X_full, y)
en_red = fit_elasticnet(X_reduced, y)

# --- Compute metrics ---
summary_data = []

for label, res, X in [
    ("MLE - Full", res_full, X_full),
    ("MLE - Reduced", res_red, X_reduced)
]:
    pseudo = pseudo_r2(res)
    summary_data.append({
        "Model": label,
        "n_features": X.shape[1],
        "AIC": res.aic,
        "LogLik": res.llf,
        "Pseudo_R2": pseudo
    })

# ElasticNet doesn’t report AIC, approximate via log-loss
from sklearn.metrics import log_loss
for label, model, X in [
    ("ElasticNet - Full", en_full, X_full),
    ("ElasticNet - Reduced", en_red, X_reduced)
]:
    y_pred = model.predict_proba(X)[:, 1]
    ll = -log_loss(y, y_pred, normalize=False)
    summary_data.append({
        "Model": label,
        "n_features": X.shape[1],
        "AIC": np.nan,
        "LogLik": ll,
        "Pseudo_R2": np.nan
    })

summary_df = pd.DataFrame(summary_data)
summary_df["ΔPseudo_R2"] = summary_df["Pseudo_R2"].diff().fillna(0)

# --- Save + print summary ---
summary_df.to_csv("model_comparison_summary.csv", index=False)
print("\n=== Model Comparison Summary ===")
print(summary_df.round(4))
