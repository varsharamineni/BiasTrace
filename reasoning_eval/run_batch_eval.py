#!/usr/bin/env python
import subprocess
import glob
import os
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing bbq_{category}_results JSON files"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="reasoning_eval/llm_judge_samples",
        help="Base output folder"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LLM model name (e.g., gpt-4.1)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt key to evaluate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--reasoning_level",
        type=str,
        choices=["low", "medium", "high"],
        default="medium"
    )
    parser.add_argument(
        "--reasoning_prompt_text",
        type=str,
        default=None
    )
    parser.add_argument(
        "--script_path",
        type=str,
        default="run_llm_eval.py",
        help="Path to the main evaluation script"
    )
    args = parser.parse_args()

    # Find all files matching bbq_*_results*.json
    pattern = os.path.join(args.input_folder, "bbq_*_results.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"⚠️ No files found in {args.input_folder} matching pattern bbq_*_results.json")
        return

    print(f"Found {len(files)} files to process.")

    for data_path in files:
        
        filename = os.path.basename(data_path)
            
        # Skip multi-category files like bbq_Race_x_SES_results.json
        if "_x_" in filename:
            print(f"⏭ Skipping multi-category file {filename}")
            continue

        dataset_name = os.path.splitext(os.path.basename(data_path))[0]
        os.makedirs(args.output_folder, exist_ok=True)

        # Load JSON and extract "results"
        import json
        with open(data_path, "r") as f:
            data_json = json.load(f)

        if "results" in data_json:
            temp_data_path = os.path.join(args.output_folder, f"{dataset_name}_tmp.json")
            with open(temp_data_path, "w") as f:
                json.dump(data_json["results"], f, indent=2)
        else:
            temp_data_path = data_path  # already list of dicts

        # Construct custom output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{dataset_name}_{args.prompt}_{args.model}_{timestamp}.json"
        output_path = os.path.join(args.output_folder, output_filename)

        # Build command to call your existing evaluation script
        cmd = [
            "python", args.script_path,
            "--model", args.model,
            "--prompt", args.prompt,
            "--data_path", temp_data_path,
            "--output_dir", args.output_folder,
            "--temperature", str(args.temperature),
            "--top_p", str(args.top_p),
            "--reasoning_level", args.reasoning_level
        ]

        if args.reasoning_prompt_text:
            cmd += ["--reasoning_prompt_text", args.reasoning_prompt_text]

        print(f"\n🚀 Running eval on {data_path}")
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"⚠️ Judge failed for {data_path}, continuing…")

        print(f"✅ Output saved: {output_path}")

if __name__ == "__main__":
    main()
