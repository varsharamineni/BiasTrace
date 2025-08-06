import os
import re
import subprocess

# Root input/output dirs
root_input_dir = "backup_outputs"
root_output_dir = "eval_results"

# Regex to match result files and extract categories
file_pattern = re.compile(r'^bbq_(.+?)_results\.json$')

# Loop through each model's output directory
for model_dir_name in os.listdir(root_input_dir):
    model_input_path = os.path.join(root_input_dir, model_dir_name)
    
    if not os.path.isdir(model_input_path):
        continue  # skip files
    
    # Extract model name, e.g., "deepseek-70B" from "bbq_results_deepseek-70B"
    model_match = re.match(r'bbq_results_(.+)', model_dir_name)
    if not model_match:
        print(f"Skipping unrecognized directory: {model_dir_name}")
        continue
    
    model_name = model_match.group(1)
    model_output_path = os.path.join(root_output_dir, model_name)
    os.makedirs(model_output_path, exist_ok=True)

    # Extract category names from result filenames
    category_files = [
        f for f in os.listdir(model_input_path)
        if file_pattern.match(f)
    ]
    categories = [
        file_pattern.match(f).group(1)
        for f in category_files
    ]
    categories.sort()

    if not categories:
        print(f"No result files found in {model_input_path}")
        continue

    print(f"\nEvaluating model: {model_name}")
    print(f"  Categories: {' '.join(categories)}")

    # Build and run the command
    cmd = [
        "python", "scripts/evaluate_bbq_outputs.py",
        "--results_dir", model_input_path,
        "--output_dir", model_output_path,
        "--categories", *categories
    ]
    
    subprocess.run(cmd, check=True)