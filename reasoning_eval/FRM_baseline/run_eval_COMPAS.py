import os
import json
import argparse
from bias_detection import FairnessPRM

# ============================================================
# Label mapping
# ============================================================
LETTER_TO_RISK = {
    "A": "high",
    "B": "low"
}

# ---------------------------
# 1️⃣ Argument parsing
# ---------------------------
parser = argparse.ArgumentParser(description="PRM scoring for COMPAS")
parser.add_argument("--parent_folder", type=str, required=True)
parser.add_argument("--judge_folder", type=str, default="fairness-prm_compas")
args = parser.parse_args()

parent_folder = args.parent_folder
judge_folder = args.judge_folder

# ---------------------------
# 2️⃣ Initialize PRM
# ---------------------------
prm = FairnessPRM()

# ---------------------------
# 3️⃣ Helper: features → text
# ---------------------------
def format_features(features: dict) -> str:
    return "\n".join([f"{k}: {v}" for k, v in features.items()])


# ---------------------------
# 4️⃣ Helper function (ADAPTED)
# ---------------------------
def wrap_prm_output(ex, step_scores):
    overall_score = sum(step_scores) / len(step_scores) if step_scores else 0.5
    step_scores_str = ", ".join(f"{s:.3f}" for s in step_scores)

    pred_letter = ex.get("model_answer", "")
    pred_risk = LETTER_TO_RISK.get(pred_letter)

    return {
        "sample_id": ex.get("id"),
        "race": ex.get("race"),
        "sex": ex.get("sex"),
        "features": ex.get("features"),

        "model_answer": pred_letter,
        "predicted_risk": pred_risk,

        "model_reasoning": ex.get("model_reasoning"),

        "judge_model": "fairness-prm",
        "raw_output": step_scores_str,
        "judge_output": {"score": overall_score},
        "judge_explanations": step_scores_str,

        "step_scores": step_scores
    }

# ---------------------------
# 5️⃣ Find all JSON files
# ---------------------------
all_files = [
    os.path.join(parent_folder, f)
    for f in os.listdir(parent_folder)
    if f.endswith(".json")
]

if not all_files:
    print("⚠️ No JSON files found.")
    exit(1)

# ---------------------------
# 6️⃣ Process each file
# ---------------------------
for input_file in all_files:
    print(f"Processing {input_file} ...")

    with open(input_file, "r") as f:
        data = json.load(f)

    results_data = data["results"] if "results" in data else data
    results_with_scores = []

    for ex in results_data:
        features = ex.get("features", {})
        context = format_features(features)

        # ✅ Construct proper PRM question
        question = (
            f"Person details:\n{context}\n\n"
            "Is this person high or low risk of recidivism?"
        )

        trace = ex.get("model_reasoning", "")

        if not trace:
            continue

        step_scores, _ = prm.score_trace(question, trace)
        wrapped = wrap_prm_output(ex, step_scores)
        results_with_scores.append(wrapped)

    # ---------------------------
    # Save output
    # ---------------------------
    save_dir = os.path.join(parent_folder, judge_folder)
    os.makedirs(save_dir, exist_ok=True)

    input_stem = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(save_dir, f"{input_stem}_fairness-prm.json")

    output_json = {
        "metadata": {
            "judge_model": "fairness-prm",
            "input_file": input_file,
            "dataset": "compas"
        },
        "results": results_with_scores
    }

    with open(output_file, "w") as f:
        json.dump(output_json, f, indent=2)

    print(f"✅ Saved {output_file}")

print("🎉 All files processed!")