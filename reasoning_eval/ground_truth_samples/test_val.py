#!/usr/bin/env python3
"""
Merge initial + full sample JSONs, convert error labels to integers,
drop duplicates, and split into validation (20%) and test (80%) sets
based on unique (bbq_category, example_id) points.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------
# File paths
# ----------------------------
json_initial = "reasoning_eval/ground_truth_samples/sample_traces_initial_annotated.json"
json_full = "reasoning_eval/ground_truth_samples/sample_traces_full_annotated.json"

val_output = "reasoning_eval/ground_truth_samples/val_set.json"
test_output = "reasoning_eval/ground_truth_samples/test_set.json"

# ----------------------------
# Load JSONs
# ----------------------------
df_initial = pd.read_json(json_initial)
df_full = pd.read_json(json_full)

# Combine
df_all = pd.concat([df_initial, df_full], ignore_index=True)
print(f"Total rows: {len(df_all)}")

# ----------------------------
# Error labels
# ----------------------------
error_labels = [
    "group_assumption",
    "bias_acknowledgement",
    "meta_reflection",
    "outside_demo_knowledge",
    "outside_topical_knowledge",
    "unresolved",
    "overthinking",
    "missing_logic",
]

# ----------------------------
# Drop rows where all error labels are NaN
# ----------------------------
df_all = df_all.dropna(subset=error_labels, how='all')
print(f"Total rows after dropping rows with all error labels missing: {len(df_all)}")

# ----------------------------------------------
# Fill remaining NaNs with 0 and convert to integers
# ----------------------------
for col in error_labels:
    if col in df_all.columns:
        df_all[col] = df_all[col].fillna(0).astype(int)

# ----------------------------
# Normalize trace_id and drop duplicates
# ----------------------------

# Create normalized version: convert any backslashes or escaped slashes to "/"
df_all["trace_id"] = (
    df_all["trace_id"]
    .astype(str)
    .str.replace(r"\/", "/", regex=False)
)

# Drop duplicates based on the normalized trace_id
df_all = df_all.drop_duplicates(subset=["trace_id"], keep="first")

print(f"Total rows after dropping duplicates: {len(df_all)}")

# ----------------------------
# Identify unique data points
# ----------------------------
unique_points = df_all[["trace_id"]]
print(f"Unique data points: {len(unique_points)}")

# ----------------------------
# Random 20/80 validation/test split
# ----------------------------
val_points, test_points = train_test_split(
    unique_points,
    test_size=0.8,
    random_state=132,
    shuffle=True
)

# Safety check: ensure no overlap
overlap = pd.merge(val_points, test_points, on=["trace_id"])
assert len(overlap) == 0, "Validation and test sets overlap!"

# ----------------------------
# Select rows based on unique points
# ----------------------------
df_val = df_all.merge(val_points, on=['trace_id'], how='inner')
df_test = df_all.merge(test_points, on=['trace_id'], how='inner')

print(f"Validation rows: {len(df_val)}, Test rows: {len(df_test)}")

# ----------------------------
# Compute class balances (proportion of positives) for each split
# ----------------------------
print("\nClass balances in validation set:")
for col in error_labels:
    prop = df_val[col].mean()
    print(f"  {col}: {prop:.3f}")

print("\nClass balances in test set:")
for col in error_labels:
    prop = df_test[col].mean()
    print(f"  {col}: {prop:.3f}")


# ----------------------------
# Add fresh sample IDs
# ----------------------------
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_val["sample_id"] = df_val.index
df_test["sample_id"] = df_test.index

# ----------------------------
# Save to JSON
# ----------------------------
df_val.to_json(val_output, orient="records", indent=2)
df_test.to_json(test_output, orient="records", indent=2)

print(f"✅ Saved validation set to {val_output}")
print(f"✅ Saved test set to {test_output}")
