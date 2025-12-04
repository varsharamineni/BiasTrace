import json
import re

input_file = "reasoning_eval/llm_judge_samples/val_set/llm_eval_claude-opus-4-1-20250805_detailed_example_clarification_opt.json"
output_file = input_file.replace(".json", "_fixed.json")

with open(input_file, "r") as f:
    data = json.load(f)

json_block_pattern = re.compile(r"\{[\s\S]*?\}", re.MULTILINE)

for item in data["results"]:
    explanations = item["judge_explanations"].get(item["judge_prompt"], "")
    matches = json_block_pattern.findall(explanations)
    if matches:
        try:
            parsed_json = json.loads(matches[-1])
            item["judge_output"] = parsed_json
        except Exception:
            pass

with open(output_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Fixed outputs saved to {output_file}")