import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


def load_outputs(outputs_path):
    with open(outputs_path, 'r') as f:
        outputs = json.load(f)
    return outputs


def load_metadata(metadata_path):
    return pd.read_csv(metadata_path)


def assign_example_ids(outputs, filtered_meta):
    if len(outputs) != len(filtered_meta):
        raise ValueError(f"Count mismatch: outputs={len(outputs)} vs metadata={len(filtered_meta)}")
    for i, output in enumerate(outputs):
        output['example_id'] = filtered_meta.loc[i, 'example_id']
    return outputs


def analyze_accuracy(outputs, metadata, group_cols):
    outputs_df = pd.DataFrame(outputs)
    merged = pd.merge(outputs_df, metadata, how='inner', on=['category', 'example_id'])

    if merged.empty:
        raise ValueError("Merged dataframe is empty. Check that 'category' and 'example_id' match.")

    acc_by_group = merged.groupby(group_cols)['correct'].mean().reset_index()
    acc_by_group = acc_by_group.sort_values(by='correct')
    return acc_by_group


def save_plot(acc_df, group_cols, fig_dir, category):
    group_label = '_'.join(group_cols)
    acc_series = acc_df.set_index(group_cols)['correct']

    plt.figure(figsize=(10, 6))
    acc_series.plot(kind='barh')
    plt.title(f"Accuracy by {group_label} for category '{category}'")
    plt.xlabel("Accuracy")
    plt.ylabel(group_label)
    plt.tight_layout()

    os.makedirs(fig_dir, exist_ok=True)
    plot_path = os.path.join(fig_dir, f"{category}_{group_label}_accuracy_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Saved plot to {plot_path}")


def save_csv(acc_df, csv_dir, category, group_cols):
    os.makedirs(csv_dir, exist_ok=True)
    group_label = '_'.join(group_cols)
    csv_path = os.path.join(csv_dir, f"{category}_{group_label}_accuracy.csv")
    acc_df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV to {csv_path}")


def get_available_categories(model_dir):
    json_paths = glob.glob(os.path.join(model_dir, '*_detailed_per_trace.json'))
    categories = [os.path.basename(path).replace('_detailed_per_trace.json', '') for path in json_paths]
    return categories


def get_available_models(eval_root='eval_results'):
    return [
        name for name in os.listdir(eval_root)
        if os.path.isdir(os.path.join(eval_root, name))
    ]


def run_analysis_for_model_and_category(model_name, category, metadata, group_cols, eval_root='eval_results'):
    model_dir = os.path.join(eval_root, model_name)
    outputs_path = os.path.join(model_dir, f'{category}_detailed_per_trace.json')
    fig_dir = os.path.join(model_dir, 'figs')
    csv_dir = os.path.join(model_dir, 'extra_eval')

    if not os.path.exists(outputs_path):
        print(f"❌ Output file not found: {outputs_path}")
        return

    try:
        outputs = load_outputs(outputs_path)
        filtered_meta = metadata[metadata['category'] == category].reset_index(drop=True)
        outputs = assign_example_ids(outputs, filtered_meta)
        acc_df = analyze_accuracy(outputs, filtered_meta, group_cols)
        save_csv(acc_df, csv_dir, category, group_cols)
        save_plot(acc_df, group_cols, fig_dir, category)
    except Exception as e:
        print(f"⚠️ Error processing {model_name} - {category}: {e}")


def main():
    # 🔧 Customize this
    group_cols = ['Relevant_social_values', 'Known_stereotyped_groups']
    metadata_path = 'bbq_additional_metadata.csv'
    eval_root = 'eval_results'

    metadata = load_metadata(metadata_path)
    model_names = get_available_models(eval_root)

    if not model_names:
        print("❌ No models found in eval_results/")
        return

    print(f"\n📂 Found models: {model_names}")

    for model_name in model_names:
        model_dir = os.path.join(eval_root, model_name)
        categories = get_available_categories(model_dir)

        if not categories:
            print(f"⚠️ No categories found for model: {model_name}")
            continue

        print(f"\n📊 Running evaluations for model: {model_name}")
        for category in categories:
            print(f"▶ Analyzing category: {category}")
            run_analysis_for_model_and_category(model_name, category, metadata, group_cols, eval_root)


if __name__ == '__main__':
    main()
