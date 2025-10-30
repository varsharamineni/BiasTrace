import pandas as pd
import json

# Load CSV
df = pd.read_csv("reasoning_eval/ground_truth_samples/sample_traces_full_annotated_test.csv")

# Convert to list of dicts
records = df.to_dict(orient="records")

# Save as JSON
with open("sample_traces_full_annotated_test.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)