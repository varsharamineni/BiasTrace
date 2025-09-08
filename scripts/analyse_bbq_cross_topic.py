import os
import json
import pandas as pd
import glob
import argparse

parser = argparse.ArgumentParser(description="Compute accuracy and variance across Relevant_social_values")
parser.add_argument("model_folders", nargs="+", help="Paths to model folders")
args = parser.parse_args()

all_summaries = []

for folder in args.model_folders:
    model_name = os.path.basename(folder.rstrip("/"))
    json_files = glob.glob(os.path.join(folder, "bbq_*_results_merged.json"))
    
    for json_file in json_files:
        category = os.path.basename(json_file).replace("bbq_", "").replace("_results_merged.json", "")
        with open(json_file) as f:
            data = json.load(f)
        
        # Flatten results
        records = []
        for r in data.get("results", []):
            records.append({
                "Ambiguous": r.get("ambiguous", False),
                "Relevant_social_values": str(r.get("Relevant_social_values", [])),
                "question_polarity": r.get("question_polarity"),
                "is_correct": 1 if r.get("is_correct", False) else 0
            })
        df = pd.DataFrame(records)
        
        if df.empty:
            continue
        
        # Mean accuracy per group
        group_acc = df.groupby(['Ambiguous', 'question_polarity', 'Relevant_social_values'], as_index=False)['is_correct'].mean()
        group_acc['Model'] = model_name
        group_acc['Category'] = category
        
        all_summaries.append(group_acc)

# Concatenate all
final_df = pd.concat(all_summaries, ignore_index=True)

# Compute standard deviation of accuracy across Relevant_social_values for each Ambiguous × question_polarity
std_df = final_df.groupby(['Model','Category','Ambiguous','question_polarity'])['is_correct'].std().reset_index()
std_df.rename(columns={'is_correct':'accuracy_std_across_social_values'}, inplace=True)

# Save
final_df.to_csv("accuracy_per_group.csv", index=False)
std_df.to_csv("accuracy_std_across_social_values.csv", index=False)

print("Accuracy per group saved to accuracy_per_group.csv")
print("Std dev across Relevant_social_values saved to accuracy_std_across_social_values.csv")
