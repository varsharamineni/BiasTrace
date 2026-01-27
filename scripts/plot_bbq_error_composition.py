import json
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Parsing helpers
# -------------------------
def parse_model_prompt_reasoning(model_str):
    """
    Model = first two tokens (uppercase)
    Prompt = everything after model tokens but before '_prompt', capitalized, default Unknown
    Reasoning = low/medium/high if present, default Unknown
    """
    model_str_lower = model_str.lower()
    parts = model_str_lower.split("_")

    # Model = first two tokens
    model = f"{parts[0].upper()}_{parts[1].upper()}"

    # Reasoning = low/medium/high if present
    reasoning = next((p for p in parts if p in ["low", "medium", "high"]), "Unknown").capitalize()

    # Prompt = everything after model tokens up to '_prompt'
    try:
        prompt_start_idx = 2
        prompt_end_idx = next(i for i, p in enumerate(parts) if "prompt" in p)
        prompt_tokens = parts[prompt_start_idx:prompt_end_idx]
        prompt = "_".join([t.capitalize() for t in prompt_tokens]) or "Unknown"
    except StopIteration:
        prompt = "Unknown"

    return model, prompt, reasoning

# -------------------------
# Load summaries
# -------------------------
def load_summaries(inputs, base_folder="results"):
    rows = []

    for inp in inputs:
        if os.path.isfile(inp):
            model_name = os.path.basename(inp).replace(".json", "")
            path = inp
        else:
            model_name = inp
            results_dir = os.path.join(base_folder, model_name, "acc_and_bias_results")
            if not os.path.isdir(results_dir):
                raise FileNotFoundError(f"Results dir not found: {results_dir}")
            json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
            if len(json_files) != 1:
                raise ValueError(f"Expected 1 JSON in {results_dir}, found {json_files}")
            path = os.path.join(results_dir, json_files[0])

        with open(path, "r") as f:
            summary = json.load(f)

        model, prompt, reasoning = parse_model_prompt_reasoning(model_name)

        for category, m in summary.items():
            for ctx in ["disamb", "amb"]:
                n_incorrect = m[f"{ctx}_n_incorrect"]
                n_incorrect_stereo = m[f"{ctx}_n_incorrect_and_stereotype"]
                n_total = m[f"n_{ctx}"]

                rows.append({
                    "Model": model,
                    "Prompt": prompt,
                    "Reasoning": reasoning,
                    "Category": category,
                    "Context": ctx.capitalize(),
                    "IncorrectRate": n_incorrect / n_total if n_total > 0 else 0,
                    "IncorrectStereoRate": (
                        n_incorrect_stereo / n_incorrect if n_incorrect > 0 else 0
                    )
                })

    return pd.DataFrame(rows)

# -------------------------
# Plotting bar composition
# -------------------------
def plot_bar_error_composition(df, context, out_file):
    """
    Barplot with:
    - X-axis: Model_Prompt_Reasoning
    - Bars stacked/colored by Category
    - Bar height: IncorrectRate
    - Inner overlay: Incorrect & Stereotype
    """
    import matplotlib.patches as mpatches

    df_ctx = df.copy()
    if context:
        df_ctx = df_ctx[df_ctx["Context"] == context]

    df_ctx["ModelPromptReasoning"] = df_ctx.apply(
        lambda x: f"{x['Model']}_{x['Prompt']}_{x['Reasoning']}", axis=1
    )

    models = df_ctx["ModelPromptReasoning"].unique()
    categories = df_ctx["Category"].unique()
    n_models = len(models)
    bar_width = 0.6

    fig, ax = plt.subplots(figsize=(max(10, n_models*0.8), 6))

    # Assign a color per category
    palette = sns.color_palette("tab10", n_colors=len(categories))
    category_colors = {cat: palette[i] for i, cat in enumerate(categories)}

    # Plot bars per model
    for i, model in enumerate(models):
        subset = df_ctx[df_ctx["ModelPromptReasoning"] == model]
        bottom = 0
        for cat in categories:
            cat_row = subset[subset["Category"] == cat]
            if cat_row.empty:
                continue
            val = cat_row["IncorrectRate"].values[0]
            stereo_val = cat_row["IncorrectStereoRate"].values[0] * val

            # Main bar = IncorrectRate
            ax.bar(
                i,
                val,
                bar_width,
                bottom=bottom,
                color=category_colors[cat],
                edgecolor="black"
            )

            # Overlay = Incorrect & Stereotype
            ax.bar(
                i,
                stereo_val,
                bar_width*0.6,
                bottom=bottom,
                color="black",
                alpha=0.7
            )

            bottom += val  # Stack bars for multiple categories

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Proportion Incorrect")
    ax.set_title(f"Error Composition — {context if context else 'Overall'}")

    # Legend for categories
    patches = [mpatches.Patch(color=col, label=cat) for cat, col in category_colors.items()]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Category")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file + ".pdf")
    plt.close()
    print(f"Saved barplot to {out_file}.pdf")

def plot_overall_bar_composition(df, out_file):
    """
    Clear overall bar plot:
    - X-axis: Model_Prompt_Reasoning
    - Two bars per model: Disamb and Amb
    - Bar height = IncorrectRate
    - Inner overlay = Incorrect & Stereotype proportion
    - Aggregated over all categories
    """
    # Aggregate over categories
    agg = df.groupby(["Model", "Prompt", "Reasoning", "Context"]).agg(
        total_incorrect=pd.NamedAgg(column="IncorrectRate", aggfunc="mean"),
        total_stereo=pd.NamedAgg(column="IncorrectStereoRate", aggfunc="mean")
    ).reset_index()

    agg["ModelPromptReasoning"] = agg.apply(
        lambda x: f"{x['Model']}_{x['Prompt']}_{x['Reasoning']}", axis=1
    )

    # Pivot for easier plotting
    contexts = ["Disamb", "Amb"]
    x_labels = agg["ModelPromptReasoning"].unique()
    x_pos = range(len(x_labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(x_labels)*0.8), 6))

    for i, ctx in enumerate(contexts):
        subset = agg[agg["Context"] == ctx]
        heights = subset.set_index("ModelPromptReasoning")["total_incorrect"].reindex(x_labels).fillna(0)
        heights_stereo = subset.set_index("ModelPromptReasoning")["total_stereo"].reindex(x_labels).fillna(0)

        positions = [p + i*bar_width - bar_width/2 for p in x_pos]

        # Main bar = IncorrectRate
        ax.bar(positions, heights, width=bar_width, label=f"{ctx} Incorrect")

        # Overlay = Incorrect & Stereotype
        ax.bar(positions, heights_stereo*heights, width=bar_width*0.6, color="black", alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Proportion Incorrect")
    ax.set_title("Overall BBQ Error Composition by Model and Context")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file + ".pdf")
    plt.close()
    print(f"Saved overall barplot to {out_file}.pdf")

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="Comma-separated model names or JSON paths")
    parser.add_argument("--base_folder", default="results")
    parser.add_argument("--out_dir", default="results/error_composition_bars")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    inputs = [i.strip() for i in args.inputs.split(",")]

    # Load all summaries
    df = load_summaries(inputs, args.base_folder)


    # ----- Overall barplot per model, aggregated over categories -----
    out_file_overall = os.path.join(args.out_dir, "ErrorComposition_Overall")
    plot_overall_bar_composition(df, out_file_overall)

    # ----- Optional: per-context category-level plots -----
    for ctx in ["Disamb", "Amb"]:
        out_file_ctx = os.path.join(args.out_dir, f"ErrorComposition_{ctx}")
        plot_bar_error_composition(df, ctx, out_file_ctx)




if __name__ == "__main__":
    main()
