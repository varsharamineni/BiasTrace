#!/usr/bin/env python
import json
import re
import argparse
from pathlib import Path

def extract_reasoning(raw_text: str) -> str:
    """Extract reasoning from <think> ... <answer> block."""
    if not raw_text:
        return None
    match = re.search(r"<think>(.*?)<answer>", raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def extract_answer_letter(raw_text: str) -> str:
    """Extract the letter (A/B/C/...) from the <answer> block."""
    if not raw_text:
        return None
    match = re.search(r"<answer>\s*([A-Z])", raw_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def correct_json(input_path: str, output_path: str):
    # Load JSON
    with open(input_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    updated_count = 0

    for item in results:
        raw_output = item.get("raw_output", "")
        
        # Extract reasoning
        reasoning = extract_reasoning(raw_output)
        if reasoning:
            item["model_reasoning"] = reasoning

        # Extract answer letter
        answer_letter = extract_answer_letter(raw_output)
        if answer_letter:
            item["model_answer"] = answer_letter
            updated_count += 1

    print(f"✅ Updated reasoning and model_answer for {updated_count}/{len(results)} samples")

    # Save corrected JSON
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved corrected JSON → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill in model_reasoning and extract letter for model_answer")
    parser.add_argument("--input", type=str, required=True, help="Path to original JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to save corrected JSON")
    args = parser.parse_args()

    correct_json(args.input, args.output)