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
    into:
        Model = 'GPT-OSS-120B'
        Prompt = 'Simple_Low_Reasoning'
    """
    parts = model_str.split('_')
    
    # Find size token (e.g., '14B', '120B')
    size_token = next((p for p in parts if re.match(r'^\d+B$', p, re.IGNORECASE)), "")
    
    # Model = base + size
    base = parts[0].upper() if parts else "UNKNOWN"
    model_name = f"{base}_{size_token}" if size_token else base

    # Everything else except base and size = prompt
    prompt_tokens = [p for p in parts[1:] if p != size_token]
    prompt_str = "_".join(prompt_tokens).replace("prompt", "").strip("_").capitalize() if prompt_tokens else "Full"

    return model_name, prompt_str

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

def generate_table_and_plots(model_inputs, base_folder, output_file, plot_folder):
    all_rows = []

    for inp in model_inputs:
        if os.path.isfile(inp):
            model_name, summary = load_model_summary(file_path=inp)
        else:
            model_name, summary = load_model_summary(model_name=inp, base_folder=base_folder)

        model_short, prompt_type = parse_model_name(model_name)

        for cat, metrics in summary.items():
            disamb_acc = metrics["disamb_n_correct"] / metrics["n_disamb"] if metrics["n_disamb"] > 0 else 0
            amb_acc = metrics["amb_n_correct"] / metrics["n_amb"] if metrics["n_amb"] > 0 else 0
            prop_disamb_wrong_stereo = metrics["disamb_n_incorrect_and_stereotype"] / metrics["disamb_n_incorrect"] if metrics["disamb_n_incorrect"] > 0 else 0
            prop_amb_wrong_stereo = metrics["amb_n_incorrect_and_stereotype"] / metrics["amb_n_incorrect"] if metrics["amb_n_incorrect"] > 0 else 0
            sAMB = (1 - amb_acc) * metrics["sDIS"]

            all_rows.append({
                "Model": model_short,
                "Prompt": prompt_type,
                "Category": cat,
                "Acc": round(metrics["accuracy"]*100, 1),
                "Acc_DIS": round(disamb_acc * 100, 1),
                "Acc_AMB": round(amb_acc * 100, 1),
                "sDIS": round(metrics["sDIS"] * 100, 1),
                "sAMB": round(sAMB * 100, 1),
                "Wrong&Stero_DIS": round(prop_disamb_wrong_stereo * 100, 1),
                "Wrong&Stero_AMB": round(prop_amb_wrong_stereo * 100, 1)
            })

    df = pd.DataFrame(all_rows)

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
        hm.columns = [f"{m}_{p}" for m, p in hm.columns]
        return hm

    heatmap_disamb_acc = flatten_columns(df.pivot_table(index='Category', columns=['Model', 'Prompt'], values='Acc_DIS'))
    heatmap_amb_acc   = flatten_columns(df.pivot_table(index='Category', columns=['Model', 'Prompt'], values='Acc_AMB'))
    heatmap_disamb_bias = flatten_columns(df.pivot_table(index='Category', columns=['Model', 'Prompt'], values='sDIS'))
    heatmap_amb_bias   = flatten_columns(df.pivot_table(index='Category', columns=['Model', 'Prompt'], values='sAMB'))
    heatmap_disamb_wrong = flatten_columns(df.pivot_table(index='Category', columns=['Model', 'Prompt'], values='Wrong&Stero_DIS'))
    heatmap_amb_wrong   = flatten_columns(df.pivot_table(index='Category', columns=['Model', 'Prompt'], values='Wrong&Stero_AMB'))

    fig, axes = plt.subplots(3, 2, figsize=(20, 18), sharey='row')

    # Accuracy
    sns.heatmap(heatmap_disamb_acc, annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1, ax=axes[0,0])
    axes[0,0].set_title("Accuracy Disambiguated (Acc_DIS)")
    sns.heatmap(heatmap_amb_acc, annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1, ax=axes[0,1])
    axes[0,1].set_title("Accuracy Ambiguous (Acc_AMB)")

    # Bias (-100 to 100)
    sns.heatmap(heatmap_disamb_bias, annot=True, fmt=".1f", cmap="coolwarm", center=0, vmin=-100, vmax=100, ax=axes[1,0])
    axes[1,0].set_title("sDIS (Disambiguated Bias)")
    sns.heatmap(heatmap_amb_bias, annot=True, fmt=".1f", cmap="coolwarm", center=0, vmin=-100, vmax=100, ax=axes[1,1])
    axes[1,1].set_title("sAMB (Ambiguous Bias)")

    # Wrong & Stereotype
    sns.heatmap(heatmap_disamb_wrong, annot=True, fmt=".1f", cmap="Reds", vmin=0, vmax=100, ax=axes[2,0])
    axes[2,0].set_title("Wrong & Stereotype (Disambiguated)")
    sns.heatmap(heatmap_amb_wrong, annot=True, fmt=".1f", cmap="Reds", vmin=0, vmax=100, ax=axes[2,1])
    axes[2,1].set_title("Wrong & Stereotype (Ambiguous)")

    plt.tight_layout()
    base_path = os.path.join(plot_folder, "heatmaps_all_metrics")
    plt.savefig(f"{base_path}.png", dpi=300)
    plt.savefig(f"{base_path}.pdf")
    plt.close()
    print(f"Heatmaps saved to {plot_folder}/heatmaps_all_metrics.png")

    def plot_metrics_side_by_side_all(df, plot_folder="results/plots"):
        """
        Generate side-by-side heatmaps for Accuracy, Bias, and Wrong&Stereotype
        for both Disambiguated and Ambiguous contexts.
        """
        os.makedirs(plot_folder, exist_ok=True)

        def pivot_metric(metric):
            hm = df.pivot_table(index='Category', columns=['Model', 'Prompt'], values=metric)
            hm.columns = [f"{m}_{p}" for m, p in hm.columns]
            return hm

        # Define metrics for each context
        contexts = {
            "Disambiguated": {"Acc": "Acc_DIS", "Bias": "sDIS", "Wrong": "Wrong&Stero_DIS"},
            "Ambiguous": {"Acc": "Acc_AMB", "Bias": "sAMB", "Wrong": "Wrong&Stero_AMB"}
        }

        for ctx_name, metrics in contexts.items():
            heatmaps = {k: pivot_metric(v) for k, v in metrics.items()}

            # 1 row, 3 columns: Accuracy | Bias | Wrong&Stero
            fig, axes = plt.subplots(1, 3, figsize=(24, 10), sharey=True)

            # Accuracy
            sns.heatmap(heatmaps["Acc"], annot=True, fmt=".1f", cmap="Greens", vmin=70, vmax=100, ax=axes[0])
            axes[0].set_title(f"Accuracy ({ctx_name})")

            # Bias score
            sns.heatmap(heatmaps["Bias"], annot=True, fmt=".1f", cmap="coolwarm", center=0, vmin=-10, vmax=10, ax=axes[1])
            axes[1].set_title(f"Bias Score ({ctx_name})")

            # Wrong & Stereotype
            sns.heatmap(heatmaps["Wrong"], annot=True, fmt=".1f", cmap="Reds", vmin=0, vmax=100, ax=axes[2])
            axes[2].set_title(f"Wrong & Stereotype ({ctx_name})")

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
