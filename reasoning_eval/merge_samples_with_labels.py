import json
import pandas as pd

def merge_samples_with_labels(
    json_path,
    csv_path,
    output_path,
    labeller=None  # e.g., "VR" — if None, use all (but warn if multiple exist)
):
    # Load data
    with open(json_path, "r") as f:
        samples = json.load(f)
    df_samples = pd.DataFrame(samples)

    df_labels = pd.read_csv(csv_path)

    # Standardize type
    df_samples["sample_id"] = df_samples["sample_id"].astype(int)
    df_labels["sample_id"] = df_labels["sample_id"].astype(int)

    # If user specifies a particular labeller:
    if labeller is not None:
        df_labels = df_labels[df_labels["labeller"] == labeller].copy()
        print(f"✅ Using only labeller: {labeller}")
    else:
        # Detect if multiple labellers exist for any sample_id
        duplicates = df_labels.duplicated(subset=["sample_id"], keep=False)
        if duplicates.any():
            print("⚠️ Warning: Multiple labellers found for some sample_ids.")
            print("⚠️ Specify a labeller in the function call: labeller='VR'")
            # This will keep all annotation columns duplicated by labeller
            # but samples may get expanded (one row per labeller)

    # Merge
    df_merged = df_samples.merge(df_labels, on="sample_id", how="left")

    # Convert back to JSON and save
    merged_records = df_merged.to_dict(orient="records")
    with open(output_path, "w") as f:
        json.dump(merged_records, f, indent=2)

    print(f"✅ Saved merged annotated JSON to {output_path}")
    return df_merged  # useful if you want it in memory


merge_samples_with_labels(
    "../reasoning_eval/data_to_label/sample_traces_inital.json",
    "../reasoning_error_taxonomy/updated_initial_annotations.csv",
    "../reasoning_eval/ground_truth_samples/sample_traces_initial_annotated.json",
    labeller="VR"
)