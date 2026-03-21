import os
import json
import argparse
from bias_detection import FairnessPRM  # your simplified PRM script

# ---------------------------
# 1️⃣ Argument parsing
# ---------------------------
parser = argparse.ArgumentParser(description="Batch score reasoning traces with Fairness PRM")
parser.add_argument(
    "--parent_folder", type=str, required=True,
    help="Parent folder containing reasoning JSON files for all categories"
)
parser.add_argument(
    "--judge_folder", type=str, default="fairness-prm_0-5_annotation",
    help="Subfolder to save judge outputs"
)
args = parser.parse_args()

parent_folder = args.parent_folder
judge_folder = args.judge_folder

# ---------------------------
# 2️⃣ Initialize PRM
# ---------------------------
prm = FairnessPRM()

# ---------------------------
# 3️⃣ Helper function to wrap PRM output
# ---------------------------
def wrap_prm_output(ex, step_scores):
    overall_score = sum(step_scores) / len(step_scores) if step_scores else 0.5
    step_scores_str = ", ".join(f"{s:.3f}" for s in step_scores)
    return {
        "sample_id": f"{ex['category']}_{ex['example_id']}",
        "category": ex["category"],
        "example_id": ex["example_id"],
        "model": ex.get("model", "unknown"),
        "judge_model": "fairness-prm",
        "raw_output": step_scores_str,
        "judge_output": {"score": overall_score},
        "judge_explanations": step_scores_str,
        "is_correct": ex.get("is_correct", True),
        "ambiguous": ex.get("ambiguous", False),
        "stereotype_alignment": ex.get("stereotype_alignment", False),
        "incorrect_and_stereotype": ex.get("incorrect_and_stereotype", False),
        "step_scores": step_scores
    }

# ---------------------------
# 4️⃣ Find all category JSON files
# ---------------------------
all_files = [
    os.path.join(parent_folder, f)
    for f in os.listdir(parent_folder)
    if f.endswith("_results_merged.json")
]

if not all_files:
    print("⚠️ No '_results_merged.json' files found in the parent folder.")
    exit(1)

# ---------------------------
# 5️⃣ Process each file
# ---------------------------
for input_file in all_files:
    print(f"Processing {input_file} ...")
    
    with open(input_file, "r") as f:
        data = json.load(f)

    # Use category from the first example
    category = data["results"][0]["category"] if data["results"] else "unknown"
    results_with_scores = []

    # Score each example
    for ex in data["results"]:
        question = f"{ex['context']} {ex['question']}"
        trace = ex["model_reasoning"]

        step_scores, _ = prm.score_trace(question, trace)
        wrapped = wrap_prm_output(ex, step_scores)
        results_with_scores.append(wrapped)

    # Prepare output folder
    save_dir = os.path.join(parent_folder, judge_folder, category)
    os.makedirs(save_dir, exist_ok=True)

    input_stem = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(save_dir, f"llm_eval_{category}_fairness-prm.json")

    output_json = {
        "metadata": {
            "judge_model": "fairness-prm",
            "input_file": input_file,
            "input_stem": input_stem,
            "bbq_category": category
        },
        "results": results_with_scores
    }

    with open(output_file, "w") as f:
        json.dump(output_json, f, indent=2)

    print(f"✅ Saved {output_file}")

print("🎉 All files processed!")