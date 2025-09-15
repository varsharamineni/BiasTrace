import json
from pathlib import Path
import os
from utils import call_claude

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
SAMPLE_FILE = Path("reasoning_error_taxonomy/data/sample_traces.json")
ANNOTATED_FILE = Path("reasoning_error_taxonomy/data/annotated_traces.json")
PROMPT_FILE = Path("reasoning_error_taxonomy/prompts/annotate_traces_prompt.txt")

BATCH_SIZE = 5  # Adjust based on how many traces you can safely send per request


def load_traces(path):
    with open(path, "r") as f:
        return json.load(f)


def load_prompt(path):
    with open(path, "r") as f:
        return f.read()


def batch_traces(traces, batch_size=BATCH_SIZE):
    for i in range(0, len(traces), batch_size):
        yield traces[i:i + batch_size]


def main():
    traces = load_traces(SAMPLE_FILE)
    prompt_template = load_prompt(PROMPT_FILE)

    annotated_traces = []

    for batch_num, batch in enumerate(batch_traces(traces), start=1):
        print(f"Processing batch {batch_num} ({len(batch)} traces)...")

        # Fill in the prompt template
        prompt = prompt_template.replace("{{sample_traces}}", json.dumps(batch, indent=2))

        # Call Claude (thinking disabled)
        completion_text = call_claude(prompt, thinking=False)

        # Extract JSON array safely
        try:
            start_idx = completion_text.find("[")
            end_idx = completion_text.rfind("]") + 1
            json_text = completion_text[start_idx:end_idx]
            batch_annotations = json.loads(json_text)
        except Exception as e:
            print(f"Failed to parse LLM output for batch {batch_num}: {e}")
            print("Raw output:\n", completion_text)
            continue

        annotated_traces.extend(batch_annotations)

    # Save all annotated traces
    ANNOTATED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATED_FILE, "w") as f:
        json.dump(annotated_traces, f, indent=2)

    print(f"Annotated traces saved to {ANNOTATED_FILE}")


if __name__ == "__main__":
    main()
