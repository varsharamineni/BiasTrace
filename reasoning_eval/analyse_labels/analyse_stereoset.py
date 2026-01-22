import json
import pandas as pd
import statsmodels.formula.api as smf
import os
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1️⃣ Load JSON files
# -------------------------------
error_json = "outputs/qwen3_8B_stereoset/full_annotaton/llm_eval_deepseek-chat_new_prompt_edit2_stereoset_temp1.0_top_p0.9_seed42_max_tokens2048.json"
baseline_json = "outputs/qwen3_8B_stereoset/baseline_0-5_annotaton/llm_eval_deepseek-chat_baseline_stereoset_temp1.0_top_p0.9_seed42_max_tokens2048.json"

def load_judge_file(json_path, include_baseline=False):
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    for r in data["results"]:
        row = {
            "sample_id": r["sample_id"],
            "category": r["category"],
            "bias_category": r["bias_category"]
        }
        if "judge_output" in r:
            # If judge_output is dict with multiple labels, flatten
            for k, v in r["judge_output"].items():
                row[k] = v
        # Include baseline score if requested
        if include_baseline and "judge_output" in r and "score" in r["judge_output"]:
            row["baseline_label"] = r["judge_output"]["score"]
        rows.append(row)
    return pd.DataFrame(rows)

df_errors = load_judge_file(error_json)
df_baseline = load_judge_file(baseline_json, include_baseline=True)

# -------------------------------
# 2️⃣ Merge on sample_id
# -------------------------------
df = pd.merge(df_errors, df_baseline[["sample_id","baseline_label"]], on="sample_id", how="left")

# -------------------------------
# 3️⃣ Binary bias
# -------------------------------
df["bias"] = (df["bias_category"] == 1).astype(int)
df["baseline_label"] = df["baseline_label"].fillna(0).astype(int)

# -------------------------------
# 4️⃣ Setup predictors
# -------------------------------
error_labels = [c for c in df.columns if c not in ["sample_id", "category", "bias_category", "bias", "baseline_label"]]
predictors_error = error_labels + ["category"]
predictors_baseline = ["baseline_label", "category"]

# -------------------------------
# 5️⃣ Logistic regressions
# -------------------------------
formula_error = "bias ~ " + " + ".join(predictors_error)
model_error = smf.logit(formula=formula_error, data=df).fit(disp=False, cov_type="HC3")

formula_baseline = "bias ~ " + " + ".join(predictors_baseline)
model_baseline = smf.logit(formula=formula_baseline, data=df).fit(disp=False, cov_type="HC3")

# -------------------------------
# 6️⃣ Save summaries
# -------------------------------
output_dir = "reasoning_eval/analyse_labels/stereoset_analysis"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "logit_error_labels_summary.txt"), "w") as f:
    f.write(model_error.summary().as_text())

with open(os.path.join(output_dir, "logit_baseline_summary.txt"), "w") as f:
    f.write(model_baseline.summary().as_text())

# -------------------------------
# 7️⃣ Predictions
# -------------------------------
df["pred_prob_error"] = model_error.predict(df[predictors_error])
df["pred_class_error"] = (df["pred_prob_error"] > 0.5).astype(int)

df["pred_prob_baseline"] = model_baseline.predict(df[predictors_baseline])
df["pred_class_baseline"] = (df["pred_prob_baseline"] > 0.5).astype(int)

# -------------------------------
# 8️⃣ Correlation matrix
# -------------------------------
corr_cols = error_labels + ["baseline_label", "bias"]
corr_matrix = df[corr_cols].corr()
corr_matrix.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation: Error Labels, Baseline, Bias")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
plt.close()

# -------------------------------
# 9️⃣ Boxplots
# -------------------------------
for label in error_labels + ["baseline_label"]:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="bias", y=label, data=df)
    plt.title(f"{label} by Bias")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"boxplot_{label}.png"))
    plt.close()

# -------------------------------
# 🔟 Regression coefficients
# -------------------------------
coefs_error = pd.Series(model_error.params).drop("Intercept")
plt.figure(figsize=(8,4))
coefs_error.sort_values().plot(kind="barh", color="skyblue")
plt.title("Logistic Regression Coefficients (Error Labels)")
plt.xlabel("Log-Odds")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "logit_coefficients_error_labels.png"))
plt.close()

coefs_baseline = pd.Series(model_baseline.params).drop("Intercept")
plt.figure(figsize=(4,4))
coefs_baseline.sort_values().plot(kind="barh", color="salmon")
plt.title("Logistic Regression Coefficients (Baseline)")
plt.xlabel("Log-Odds")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "logit_coefficients_baseline.png"))
plt.close()

print(f"Analysis complete. Outputs saved to {output_dir}")
