import json
from pathlib import Path

from pathlib import Path
import json

def extract_all_wrong_traces(base_dir, output_path):
    base_dir = Path(base_dir)
    all_wrong = []

    # Recursively find all JSON files ending with _results_merged.json
    json_files = list(base_dir.rglob("*_results_merged.json"))
    print(f"Found {len(json_files)} JSON files total")  # debug

    for file in json_files:
        print(f"Processing file: {file}")  # debug
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse {file}")
            continue

        wrong_traces = [
            {
                **trace,
                "source_file": file.name,
                "model_dir": file.parent.name,  # parent folder as model_dir
                "model": data.get("metadata", {}).get("model"),
                "category": data.get("metadata", {}).get("category"),
            }
            for trace in data.get("results", [])
            if not trace.get("is_correct", True)
        ]

        all_wrong.extend(wrong_traces)

    # Split wrong traces by stereotype_alignment
    wrong_with_stereotype = [t for t in all_wrong if t.get("stereotype_alignment") is True]
    wrong_against_stereotype = [t for t in all_wrong if t.get("stereotype_alignment") is False]
    wrong_unknown = [t for t in all_wrong if "stereotype_alignment" not in t]

    output_data = {
        "num_wrong_total": len(all_wrong),
        "num_wrong_stereotype_true": len(wrong_with_stereotype),
        "num_wrong_stereotype_false": len(wrong_against_stereotype),
        "num_wrong_stereotype_missing": len(wrong_unknown),
        "results_all": all_wrong,
        "results_stereotype_true": wrong_with_stereotype,
        "results_stereotype_false": wrong_against_stereotype,
        "results_stereotype_missing": wrong_unknown
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(all_wrong)} wrong traces total")
    print(f"  → {len(wrong_with_stereotype)} stereotype_alignment=True")
    print(f"  → {len(wrong_against_stereotype)} stereotype_alignment=False")
    print(f"  → {len(wrong_unknown)} missing stereotype_alignment")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python extract_wrong_traces_all.py outputs/ wrong_traces_all.json")
    else:
        extract_all_wrong_traces(sys.argv[1], sys.argv[2])