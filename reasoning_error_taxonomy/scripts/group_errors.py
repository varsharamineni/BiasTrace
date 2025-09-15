import json
from pathlib import Path
from utils import call_claude, extract_json_array

RAW_FILE = Path("reasoning_error_taxonomy/data/grouped_errors_raw.txt")
JSON_FILE = Path("reasoning_error_taxonomy/data/grouped_errors.json")
ANNOTATED_FILE = Path("reasoning_error_taxonomy/data/annotated_traces.json")  # input

def load_annotated_traces(path):
    with open(path, "r") as f:
        return json.load(f)

def create_grouping_prompt(traces):
    return f"""
You are an expert analyst of reasoning traces from large language models.

Task:
1. Look at all reasoning errors in these traces.
2. Group very similar errors together into around 10 broad categories.
3. For each category, provide:
   - A short error_category name
   - A general_description of the category
   - A few specific examples, each with:
       * specific_description
       * example text from the trace
4. Return the result as a JSON array like this:

[
  {{
    "error_category": "...",
    "general_description": "...",
    "examples": [
      {{
        "specific_description": "...",
        "example": "..."
      }},
      ...
    ]
  }},
  ...
]

Here are the annotated traces:
---
{json.dumps(traces, indent=2)}
---
"""

def main():
    traces = load_annotated_traces(ANNOTATED_FILE)
    prompt = create_grouping_prompt(traces)

    print("Sending grouping prompt to Claude...")
    completion_text = call_claude(prompt, thinking=False)

    # Save raw output first
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_FILE, "w") as f:
        f.write(completion_text)
    print(f"Raw output saved to {RAW_FILE}")

    # Attempt to parse JSON
    grouped = extract_json_array(completion_text)
    if grouped is None:
        print("Failed to parse JSON, check the raw output.")
        return

    # Save parsed JSON
    with open(JSON_FILE, "w") as f:
        json.dump(grouped, f, indent=2)
    print(f"Grouped errors JSON saved to {JSON_FILE}")

if __name__ == "__main__":
    main()
