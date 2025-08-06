import json
import pandas as pd
import matplotlib.pyplot as plt
import os


def load_outputs(outputs_path):
    with open(outputs_path, 'r') as f:
        outputs = json.load(f)
    return outputs

def load_metadata(metadata_path):
    return pd.read_csv(metadata_path)

def assign_example_ids(outputs, metadata, category):
    # Filter metadata for the given category
    filtered_meta = metadata[metadata['category'] == category].reset_index(drop=True)

    if len(outputs) != len(filtered_meta):
        raise ValueError(f"Count mismatch: outputs={len(outputs)} vs metadata filtered={len(filtered_meta)}")

    # Assign example_id to each output by order
    for i, output in enumerate(outputs):
        output['example_id'] = filtered_meta.loc[i, 'example_id']

    return outputs, filtered_meta

def merge_and_analyze(outputs, metadata, group_col):
    outputs_df = pd.DataFrame(outputs)

    # Merge on category and example_id
    merged = pd.merge(outputs_df, metadata, how='inner', on=['category', 'example_id'])

    # Calculate accuracy per group
    acc_by_group = merged.groupby(group_col)['correct'].mean()

    return acc_by_group

def main():
    outputs_path = 'eval_results/deepseek-70B/Religion_detailed_per_trace.json'
    metadata_path = 'bbq_additional_metadata.csv'
    category = 'Religion'
    group_col = 'Relevant_social_values'

    outputs = load_outputs(outputs_path)
    metadata = load_metadata(metadata_path)

    print(f"Total outputs: {len(outputs)}")
    filtered_meta = metadata[metadata['category'] == category].reset_index(drop=True)
    print(f"Metadata rows for category '{category}': {len(filtered_meta)}")

    if len(outputs) != len(filtered_meta):
        print(f"Warning: outputs count {len(outputs)} != filtered meta count {len(filtered_meta)}")

    # Assign example_id
    for i, output in enumerate(outputs):
        output['example_id'] = filtered_meta.loc[i, 'example_id']

    outputs_df = pd.DataFrame(outputs)
    print("Example IDs in outputs:", outputs_df['example_id'].head())
    print("Filtered meta example_ids:", filtered_meta['example_id'].head())

    merged = pd.merge(outputs_df, filtered_meta, how='inner', on=['category', 'example_id'])
    print(f"Merged dataframe shape: {merged.shape}")
    if merged.empty:
        print("No matching rows after merge! Check your keys.")
        return

    print("Unique groups in", group_col, ":", merged[group_col].unique())
    print("Sample 'correct' values:", merged['correct'].head())

    acc_by_group = merged.groupby(group_col)['correct'].mean()

    print(f"Accuracy by {group_col} for category '{category}':")
    for group, acc in acc_by_group.items():
        print(f"  {group}: {acc:.3f}")

    # Convert to Series for plotting
    acc_series = pd.Series(acc_by_group).sort_values()

    # Plotting
    plt.figure(figsize=(10, 6))
    acc_series.plot(kind='barh')
    plt.title(f"Accuracy by {group_col} for category '{category}'")
    plt.xlabel("Accuracy")
    plt.ylabel(group_col)
    plt.tight_layout()

    # Define correct output directory
    output_dir = "eval_results/deepseek-70B"
    os.makedirs(output_dir, exist_ok=True)

    # Save plot
    plot_path = os.path.join(output_dir, f"{category}_{group_col}_accuracy_plot.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

if __name__ == '__main__':
    main()
