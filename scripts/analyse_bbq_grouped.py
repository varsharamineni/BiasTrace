import json
import pandas as pd
import os


def load_outputs(outputs_path):
    with open(outputs_path, 'r') as f:
        outputs = json.load(f)
    return outputs

def load_metadata(metadata_path):
    return pd.read_csv(metadata_path)

def assign_example_ids(outputs, metadata, category):
    filtered_meta = metadata[metadata['category'] == category].reset_index(drop=True)

    if len(outputs) != len(filtered_meta):
        raise ValueError(f"Count mismatch: outputs={len(outputs)} vs metadata filtered={len(filtered_meta)}")

    for i, output in enumerate(outputs):
        output['example_id'] = filtered_meta.loc[i, 'example_id']

    return outputs, filtered_meta

def main():
    outputs_path = 'eval_results/deepseek-70B/SES_detailed_per_trace.json'
    metadata_path = 'bbq_additional_metadata.csv'
    category = 'SES'
    group_cols = ['Relevant_social_values', 'Known_stereotyped_groups']  # 👈 Use a list for multi-grouping

    # Load data
    outputs = load_outputs(outputs_path)
    metadata = load_metadata(metadata_path)

    print(f"Total outputs: {len(outputs)}")
    filtered_meta = metadata[metadata['category'] == category].reset_index(drop=True)
    print(f"Metadata rows for category '{category}': {len(filtered_meta)}")

    if len(outputs) != len(filtered_meta):
        print(f"Warning: outputs count {len(outputs)} != filtered meta count {len(filtered_meta)}")

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

    print("Grouping by columns:", group_cols)
    print("Sample 'correct' values:", merged['correct'].head())

    acc_by_group = merged.groupby(group_cols)['correct'].mean().reset_index()
    acc_by_group = acc_by_group.sort_values(by='correct')

    print(f"\nAccuracy by {group_cols} for category '{category}':")
    print(acc_by_group)

    # Define output directory and save CSV
    output_dir = "eval_results/deepseek-70B"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{category}_{'_'.join(group_cols)}_accuracy.csv")
    acc_by_group.to_csv(csv_path, index=False)
    print(f"Saved grouped accuracy to {csv_path}")


if __name__ == '__main__':
    main()
