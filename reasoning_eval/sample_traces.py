import json
import random
from pathlib import Path
from collections import defaultdict
import pandas as pd

# === CONFIG ===
FOLDERS_TO_USE = [
    "outputs/qwen_full_8B_full_prompt",
    "outputs/qwen_full_14B_simple_prompt/20250828_215719",
    "outputs/qwen_full_14B_full_prompt",
    "outputs/qwen_full_8B_simple_prompt/20250827_163953"

]
OUT_PATH = Path("reasoning_eval/data_to_label/sample_traces.json")
N_PER_MODEL_PROMPT = 25  # number of traces per model × prompt type
random.seed(42)

# === LOAD RESULTS (only _merged.json) ===
def load_results(folders):
    traces = []

    SKIP_CATEGORIES = {"Race_x_gender", "Race_x_SES"}



    for folder in folders:
        folder_path = Path(folder)
        folder_str = str(folder_path)
        if "simple_prompt" in folder_str:
            prompt_type = "simple_prompt"
        elif "full_prompt" in folder_str:
            prompt_type = "full_prompt"
        else:
            prompt_type = folder_str  # fallback

        for path in folder_path.rglob("*_merged.json"):
            try:
                data = json.load(open(path, "r"))
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue

            model = data.get("metadata", {}).get("model", "unknown")
            category = data.get("metadata", {}).get("category", "unknown")

            # skip unwanted categories
            if category in SKIP_CATEGORIES:
                continue

            for r in data.get("results", []):
                example_id = r.get("example_id")
                traces.append({
                    "trace_id": f"{category}_{example_id}_{model}_{prompt_type}",
                    "example_id": example_id,
                    "model": model,
                    "prompt_type": prompt_type,
                    "bbq_category": category,
                    "context": r.get("context"),
                    "question": r.get("question"),
                    "answer_options": r.get("answer_options"),
                    "model_reasoning": r.get("model_reasoning"),
                    "model_answer": r.get("model_answer"),
                    "correct_answer": r.get("correct_answer"),
                    "is_correct": r.get("is_correct"),
                    "stereotype_aligned": r.get("stereotype_alignment"),
                })
    return traces

# === WEIGHTED SAMPLING PER MODEL × PROMPT TYPE ===
def weighted_sample(traces, n_per_combo):
    samples = []

    # group by (model, prompt_type)
    combos = defaultdict(list)
    for t in traces:
        key = (t["model"], t["prompt_type"])
        combos[key].append(t)

    for (model, prompt_type), combo_traces in combos.items():
        # create 4 buckets: (correct/incorrect) × (aligned/not_aligned)
        buckets = defaultdict(list)
        for t in combo_traces:
            key = ("correct" if t["is_correct"] else "incorrect",
                   "aligned" if t["stereotype_aligned"] else "not_aligned")
            buckets[key].append(t)

        # weights: more from incorrect + not_aligned
        weights = {
            ("incorrect", "not_aligned"): 0.4,
            ("incorrect", "aligned"): 0.4,
            ("correct", "not_aligned"): 0.1,
            ("correct", "aligned"): 0.1,
        }

        combo_sample = []
        total_weight = sum(weights.values())
        for key, w in weights.items():
            k = int(n_per_combo * (w / total_weight))
            choices = buckets.get(key, [])
            if choices:
                k = min(k, len(choices))
                combo_sample.extend(random.sample(choices, k))

        # fill up to n_per_combo if needed
        while len(combo_sample) < n_per_combo:
            combo_sample.append(random.choice(combo_traces))

        samples.extend(combo_sample[:n_per_combo])

    return samples

# === MAIN ===
if __name__ == "__main__":
    traces = load_results(FOLDERS_TO_USE)
    print(f"Loaded {len(traces)} traces from specified folders")

    sampled = weighted_sample(traces, N_PER_MODEL_PROMPT)
    print(f"Sampled {len(sampled)} traces")

     # --- Add sample_id counting number of datapoints ---
    for i, trace in enumerate(sampled):
        trace["sample_id"] = i

    with open(OUT_PATH, "w") as f:
        json.dump(sampled, f, indent=2)

    print(f"Saved sampled traces to {OUT_PATH}")


    # Convert JSON list of dicts to DataFrame
    df = pd.DataFrame(sampled)

    # Save to CSV
    df.to_csv("reasoning_eval/data_to_label/sampled_traces_for_labeling.csv", index=False)

