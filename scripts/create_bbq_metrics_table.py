import json
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def parse_model_name(model_str):
    """
    Converts strings like:
    'gpt-oss-120b_simple_prompt_low_reasoning'
        -> 'GPT-OSS-120B | Simple Prompt | Low Reasoning'
    'qwen-8b_simple_prompt'
        -> 'QWEN-8B | Simple Prompt'
    'qwen_full_8b_simple_prompt'
        -> 'QWEN-8B | Simple Prompt'
    """

    parts = model_str.lower().split("_")

    # ------------------------
    # 1. Find model size
    # ------------------------
    size_idx = None
    size_token = None
    for i, p in enumerate(parts):
        if re.fullmatch(r"\d+b", p):
            size_idx = i
            size_token = p.upper()
            break

    # ------------------------
    # 2. Split model vs prompt parts
    # ------------------------
    prompt_keywords = {
        "simple", "full", "prompt",
        "low", "medium", "high", "reasoning"
    }

    if size_idx is not None:
        before_size = parts[:size_idx]
        after_size = parts[size_idx + 1:]

        base_parts = [p for p in before_size if p not in prompt_keywords]
        prompt_parts = [p for p in before_size if p in prompt_keywords] + after_size

        base = "-".join(p.upper() for p in base_parts) if base_parts else "UNKNOWN"
        model_name = f"{base}-{size_token}"
    else:
        model_name = parts[0].upper()
        prompt_parts = parts[1:]

    # ------------------------
    # 3. Prompt type
    # ------------------------
    if "simple" in prompt_parts:
        prompt_type = "Simple Prompt"
    else:
        prompt_type = "Full Prompt"

    # ------------------------
    # 4. Reasoning level
    # ------------------------
    reasoning_level = None
    for level in ["low", "medium", "high"]:
        if level in prompt_parts:
            reasoning_level = f"{level.capitalize()} Reasoning"
            break

    # ------------------------
    # 5. Assemble display name
    # ------------------------
    components = [model_name, prompt_type]
    if reasoning_level:
        components.append(reasoning_level)

    return " | ".join(components)

def load_model_summary(model_name=None, file_path=None, base_folder="results"):
    if file_path:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model summary file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        model_name = os.path.basename(file_path).replace(".json", "") if not model_name else model_name
    elif model_name:
        file_path = os.path.join(base_folder, model_name, "acc_and_bias_results", f"{model_name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model summary file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        raise ValueError("Either model_name or file_path must be provided.")
    return model_name, summary

        # ---------------------------
    # Summary table by context, model type, model size, prompt
    # ---------------------------

def extract_model_type_and_size(full_model_name):
        """
        Extract model type (Qwen, GPT-OSS) and size (8B, 14B, 120B)
        from full model string like 'GPT-OSS-120B | Simple Prompt (Low Reasoning)'
        """
        model_type_size = full_model_name.split("|")[0].strip()
        if "-" in model_type_size:
            parts = model_type_size.split("-")
            model_type = "-".join(parts[:-1])
            model_size = parts[-1]
        else:
            model_type = model_type_size
            model_size = "UNKNOWN"
        return model_type, model_size



def generate_table_and_plots(model_inputs, base_folder, output_file, plot_folder):
    all_rows = []

    for inp in model_inputs:
        if os.path.isfile(inp):
            model_name, summary = load_model_summary(file_path=inp)
        else:
            model_name, summary = load_model_summary(model_name=inp, base_folder=base_folder)

        full_model_name = parse_model_name(model_name)

        for cat, metrics in summary.items():
            disamb_acc = metrics["disamb_n_correct"] / metrics["n_disamb"] if metrics["n_disamb"] > 0 else 0
            disamb_inacc = metrics["disamb_n_incorrect"] / metrics["n_disamb"] if metrics["n_disamb"] > 0 else 0
            amb_inacc = metrics["amb_n_incorrect"] / metrics["n_amb"] if metrics["n_amb"] > 0 else 0
            amb_acc = metrics["amb_n_correct"] / metrics["n_amb"] if metrics["n_amb"] > 0 else 0
            prop_disamb_wrong_stereo = metrics["disamb_n_incorrect_and_stereotype"] / metrics["disamb_n_incorrect"] if metrics["disamb_n_incorrect"] > 0 else 0
            prop_amb_wrong_stereo = metrics["amb_n_incorrect_and_stereotype"] / metrics["amb_n_incorrect"] if metrics["amb_n_incorrect"] > 0 else 0
            sAMB = (1 - amb_acc) * metrics["sDIS"]

            all_rows.append({
                "Model": full_model_name,
                "Category": cat,
                "Acc": round(metrics["accuracy"]*100, 1),
                "Acc_DIS": round(disamb_acc * 100, 1),
                "Acc_AMB": round(amb_acc * 100, 1),
                "Incorrect_AMB": round(amb_inacc * 100, 1),
                "Incorrect_DIS": round(disamb_inacc * 100, 1),
                "sDIS": round(metrics["sDIS"] * 100, 1),
                "sAMB": round(metrics['sAMB'] * 100, 1),
                "Incorrect&Stereo_DIS": round(prop_disamb_wrong_stereo * 100, 1),
                "Incorrect&Stereo_AMB": round(prop_amb_wrong_stereo * 100, 1)
            })

    df = pd.DataFrame(all_rows)

    # Add columns for model_type and model_size
    df["Model_Type"], df["Model_Size"] = zip(*df["Model"].map(extract_model_type_and_size))
    df["Prompt_Type"] = df["Model"].apply(lambda x: x.split("|")[1].strip())

    summary_stats = []

    group_cols = ["Context", "Model_Type", "Model_Size", "Prompt_Type"]
    df_long = df.melt(id_vars=["Model", "Category", "Model_Type", "Model_Size", "Prompt_Type"], 
                    value_vars=["Incorrect_DIS", "Incorrect_AMB", "Incorrect&Stereo_DIS", "Incorrect&Stereo_AMB"],
                    var_name="Metric", value_name="Value")

    # Add Context column
    df_long["Context"] = df_long["Metric"].apply(lambda x: "Disambiguated" if "DIS" in x else "Ambiguous")

    # Aggregate statistics
    agg_funcs = ["mean", "min", "max", "median"]
    summary_table = df_long.groupby(["Context", "Model_Type", "Model_Size", "Prompt_Type", "Metric"]).agg(Value_Avg=("Value","mean"),
                                                                                                        Value_Min=("Value","min"),
                                                                                                        Value_Max=("Value","max"),
                                                                                                        Value_Median=("Value","median")).reset_index()

    # Save summary table as CSV and LaTeX
    summary_csv = os.path.join(plot_folder, "bbq_summary_table.csv")
    summary_tex = os.path.join(plot_folder, "bbq_summary_table.tex")
    summary_table.to_csv(summary_csv, index=False)
    with open(summary_tex, "w") as f:
        f.write(summary_table.to_latex(index=False, caption="BBQ Summary Statistics by Model Type, Size, and Prompt", label="tab:bbq_summary"))
    print(f"Summary table saved to {summary_csv} and {summary_tex}")





    # Save LaTeX table
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(df.to_latex(index=False, caption="BBQ Metrics Across Models", label="tab:bbq_multi_model"))
    print(f"LaTeX table saved to {output_file}")

    df = df[df['Category'] != 'Race_x_SES']  
    df = df[df['Category'] != 'Race_x_gender']  


    # ---------------------------
    # Generate 3x2 Heatmaps: Accuracy, Bias, Wrong&Stero
    # ---------------------------
    os.makedirs(plot_folder, exist_ok=True)

    def flatten_columns(hm):
        hm.columns = list(hm.columns)
        return hm

    # heatmap_disamb_acc = flatten_columns(df.pivot_table(index='Category', columns='Model', values='Acc_DIS'))
    # heatmap_amb_acc   = flatten_columns(df.pivot_table(index='Category', columns='Model', values='Acc_AMB'))
    # heatmap_disamb_inacc = flatten_columns(df.pivot_table(index='Category', columns='Model', values='Incorrect_DIS'))
    # heatmap_amb_inacc   = flatten_columns(df.pivot_table(index='Category', columns='Model', values='Incorrect_AMB'))
    # heatmap_disamb_bias = flatten_columns(df.pivot_table(index='Category', columns='Model', values='sDIS'))
    # heatmap_amb_bias   = flatten_columns(df.pivot_table(index='Category', columns='Model', values='sAMB'))
    # heatmap_disamb_wrong = flatten_columns(df.pivot_table(index='Category', columns='Model', values='Incorrect&Stereo_DIS'))
    # heatmap_amb_wrong   = flatten_columns(df.pivot_table(index='Category', columns='Model', values='Incorrect&Stereo_AMB'))


    heatmap_disamb_acc = flatten_columns(df.pivot_table(index='Model', columns='Category', values='Acc_DIS'))
    heatmap_amb_acc   = flatten_columns(df.pivot_table(index='Model', columns='Category', values='Acc_AMB'))
    heatmap_disamb_inacc = flatten_columns(df.pivot_table(index='Model', columns='Category', values='Incorrect_DIS'))
    heatmap_amb_inacc   = flatten_columns(df.pivot_table(index='Model', columns='Category', values='Incorrect_AMB'))
    heatmap_disamb_bias = flatten_columns(df.pivot_table(index='Model', columns='Category', values='sDIS'))
    heatmap_amb_bias   = flatten_columns(df.pivot_table(index='Model', columns='Category', values='sAMB'))
    heatmap_disamb_wrong = flatten_columns(df.pivot_table(index='Model', columns='Category', values='Incorrect&Stereo_DIS'))
    heatmap_amb_wrong   = flatten_columns(df.pivot_table(index='Model', columns='Category', values='Incorrect&Stereo_AMB'))


    # -----------------------------
    # 1. Helper to extract row sort keys
    # -----------------------------
    def extract_row_sort_keys(model_str):
        """
        Returns a tuple to sort by:
        1. Model size ascending
        2. Prompt type: Simple -> Full
        3. Reasoning: Low -> Medium -> High
        """
        # --- size ---
        size_match = re.search(r"(\d+)B", model_str)
        size = int(size_match.group(1)) if size_match else float("inf")

        # --- prompt type ---
        if "Simple Prompt" in model_str:
            prompt_rank = 0
        elif "Full Prompt" in model_str:
            prompt_rank = 1
        else:
            prompt_rank = 2

        # --- reasoning ---
        if "Low Reasoning" in model_str:
            reasoning_rank = 0
        elif "Medium Reasoning" in model_str:
            reasoning_rank = 1
        elif "High Reasoning" in model_str:
            reasoning_rank = 2
        else:
            reasoning_rank = -1

        return (size, prompt_rank, reasoning_rank)

    # -----------------------------
    # 2. Helper to reorder columns
    # -----------------------------
    def reorder_columns(df):
        cols = df.columns.tolist()
        # "OVERALL" first, then the rest alphabetically
        cols_sorted = sorted([c for c in cols if c != "OVERALL"])  
        if "OVERALL" in cols:
            cols_sorted = ["OVERALL"] + cols_sorted
        return df[cols_sorted]

    # -----------------------------
    # 3. Reorder rows and columns
    # -----------------------------
    def reorder_heatmap(df):
        # reorder rows
        df = df.loc[sorted(df.index, key=extract_row_sort_keys)]
        # reorder columns
        df = reorder_columns(df)
        return df

    # -----------------------------
    # 4. Apply to all your heatmaps
    # -----------------------------
    heatmap_disamb_acc    = reorder_heatmap(heatmap_disamb_acc)
    heatmap_amb_acc       = reorder_heatmap(heatmap_amb_acc)
    heatmap_disamb_inacc  = reorder_heatmap(heatmap_disamb_inacc)
    heatmap_amb_inacc     = reorder_heatmap(heatmap_amb_inacc)
    heatmap_disamb_bias   = reorder_heatmap(heatmap_disamb_bias)
    heatmap_amb_bias      = reorder_heatmap(heatmap_amb_bias)
    heatmap_disamb_wrong  = reorder_heatmap(heatmap_disamb_wrong)
    heatmap_amb_wrong     = reorder_heatmap(heatmap_amb_wrong)

    # Order categories: OVERALL first, then alphabetical
    #cat_order = ["OVERALL"] + sorted([c for c in df['Category'].unique() if c != "OVERALL"])

    # heatmap_disamb_acc   = heatmap_disamb_acc.reindex(cat_order)
    # heatmap_amb_acc      = heatmap_amb_acc.reindex(cat_order)
    # heatmap_disamb_inacc = heatmap_disamb_inacc.reindex(cat_order)
    # heatmap_amb_inacc    = heatmap_amb_inacc.reindex(cat_order)
    # heatmap_disamb_bias  = heatmap_disamb_bias.reindex(cat_order)
    # heatmap_amb_bias     = heatmap_amb_bias.reindex(cat_order)
    # heatmap_disamb_wrong = heatmap_disamb_wrong.reindex(cat_order)
    # heatmap_amb_wrong    = heatmap_amb_wrong.reindex(cat_order)
    
    
    fig, axes = plt.subplots(2, 2, figsize=(28, 20), sharey='row', sharex=True)

    plt.subplots_adjust(wspace=0.000001)

    sns.set_context("notebook")

    annot_kws={"size": 16, "weight": "medium"}

    # Accuracy
    sns.heatmap(heatmap_disamb_inacc, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100, ax=axes[0,0], cbar=False, annot_kws=annot_kws)
    axes[0,0].set_title("% Bias Related Error (Disambiguated)")
    sns.heatmap(heatmap_amb_inacc, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100, ax=axes[0,1], annot_kws=annot_kws)
    axes[0,1].set_title("% Bias Related Error (Ambiguous)")
    axes[0,1].yaxis.set_visible(False)

    # Wrong & Stereotype
    sns.heatmap(heatmap_disamb_wrong, annot=True, fmt=".1f", cmap="Reds", vmin=0, vmax=100, ax=axes[1,0], cbar=False, annot_kws=annot_kws)
    axes[1,0].set_title("% of Bias Related Errors that Reinforce Stereotypes (Disambiguated)")
    sns.heatmap(heatmap_amb_wrong, annot=True, fmt=".1f", cmap="Reds", vmin=0, vmax=100, ax=axes[1,1], annot_kws=annot_kws)
    axes[1,1].set_title("% of Bias Related Errors that Reinforce Stereotypes (Ambiguous)")
    axes[1,1].yaxis.set_visible(False)  


    # Bias (-100 to 100)
    #sns.heatmap(heatmap_disamb_bias, annot=True, fmt=".1f", cmap="coolwarm", center=0, vmin=-100, vmax=100, ax=axes[2,0])
    #axes[2,0].set_title("sDIS (Disambiguated Bias)")
    #sns.heatmap(heatmap_amb_bias, annot=True, fmt=".1f", cmap="coolwarm", center=0, vmin=-100, vmax=100, ax=axes[2,1])
    #axes[2,1].set_title("sAMB (Ambiguous Bias)")

    for ax in fig.get_axes():
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    for ax in axes.flat:
        ax.tick_params(axis='both', labelsize=16)
        ax.yaxis.label.set_size(16)
        ax.xaxis.label.set_size(16)
        ax.set_title(ax.get_title(), fontsize=18)

    plt.tight_layout()
    base_path = os.path.join(plot_folder, "heatmaps_all_metrics")
    plt.savefig(f"{base_path}.png", dpi=300)
    plt.savefig(f"{base_path}.pdf")
    plt.close()
    print(f"Heatmaps saved to {plot_folder}/heatmaps_all_metrics.png")

    def plot_metrics_side_by_side_all(df, plot_folder="results/plots"):
        """
        Generate side-by-side heatmaps for Accuracy, Bias, and Incorrect&Stereotype
        for both Disambiguated and Ambiguous contexts.
        """
        os.makedirs(plot_folder, exist_ok=True)

        def pivot_metric(metric):
            hm = df.pivot_table(index='Category', columns='Model', values=metric)
            hm.columns = list(hm.columns)
            return hm

        # Define metrics for each context
        contexts = {
            "Disambiguated": {"Incorrect": "Incorrect_DIS", "Bias": "sDIS", "Wrong": "Incorrect&Stereo_DIS"},
            "Ambiguous": {"Incorrect": "Incorrect_AMB", "Bias": "sAMB", "Wrong": "Incorrect&Stereo_AMB"}
        }

        for ctx_name, metrics in contexts.items():
            heatmaps = {k: pivot_metric(v) for k, v in metrics.items()}

            # 1 row, 3 columns: Accuracy | Bias | Wrong&Stero
            fig, axes = plt.subplots(1, 3, figsize=(24, 10), sharey=True)

            # Accuracy
            sns.heatmap(heatmaps["Incorrect"], annot=True, fmt=".1f", cmap="Greens", vmin=70, vmax=100, ax=axes[0])
            axes[0].set_title(f"% Incorrect ({ctx_name})")


            # Wrong & Stereotype
            sns.heatmap(heatmaps["Wrong"], annot=True, fmt=".1f", cmap="Reds", vmin=0, vmax=100, ax=axes[1])
            axes[1].set_title(f"% of Incorrect Answers that Align with Stereotypes ({ctx_name})")

            # Bias score
            sns.heatmap(heatmaps["Bias"], annot=True, fmt=".1f", cmap="coolwarm", center=0, vmin=-10, vmax=10, ax=axes[2])
            axes[2].set_title(f"Bias Score ({ctx_name})")

            plt.tight_layout()
            base_path = os.path.join(plot_folder, f"metrics_side_by_side_{ctx_name.lower()}")
            plt.savefig(f"{base_path}.png", dpi=300)
            plt.savefig(f"{base_path}.pdf")
            plt.close() 
            print(f"{ctx_name} side-by-side metrics heatmap saved to {base_path}")
    
    plot_metrics_side_by_side_all(df, plot_folder)


def main():
    parser = argparse.ArgumentParser(description="Generate multi-model BBQ LaTeX table and plots")
    parser.add_argument("--inputs", required=True, help="Comma-separated list of model names or JSON files")
    parser.add_argument("--base_folder", default="results", help="Base folder for model results")
    parser.add_argument("--out_file", default="results/bias_summary_table_multi_model.tex", help="LaTeX output file")
    parser.add_argument("--plot_folder", default="results/plots", help="Folder to save plots")
    args = parser.parse_args()

    model_inputs = [i.strip() for i in args.inputs.split(",")]
    generate_table_and_plots(model_inputs, args.base_folder, args.out_file, args.plot_folder)

if __name__ == "__main__":
    main()