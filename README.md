# Bias-Reasoning-LLM

## Installation

We recommend using **Poetry** for dependency management:

```bash
poetry install
poetry shell
source $(poetry env info --path)/bin/activate
```
If you prefer pip, you can use:


```bash
pip install -r requirements.txt
```

## Scripts

### Download BBQ Data 

Downloads BBQ data from Hugging Face and saves them locally

```bash 
python scripts/download_data/download_bbq_all_cat.py
```

BBQ data saved to `datasets/bbq_data_all_cat`

You should also download the templates to `datasets/bbq_templates`[https://github.com/nyu-mll/BBQ/tree/main/templates] folder and the metadata `datasets/bbq_additional_metadata.csv`[https://github.com/nyu-mll/BBQ/tree/main/supplemental]  

### Download Models to Cache

The `snapshot_download.py` script downloads models from Hugging Face and saves them locally

Note: Replace `base_cache_dir` with your desired local folder

```bash
 python scripts/download_models/snapshot_download.py
```

### Generate Reasoning Traces - Qwen 

Can use `job_scripts` to run these and get reasoning traces, can edit model and folders and paramters as needed

```bash
job_scripts/run_full.sh
``` 

These rely on these depending on the type of prompt used.

With full prompt
`scripts/generate_bbq_outputs_vllm_qwen.py`

With simple prompt
`scripts/generate_bbq_outputs_vllm_qwen_simple.py`

## Saved Reasoning Traces

Full Generated Traces are available at: 

[Google Drive Link](https://drive.google.com/drive/folders/1avhCGFPHafxfV80AtVIbEFRfT08Vsb67?usp=sharing)

## Process Results

Merge model outputs with the original BBQ dataset and optional metadata useful for further analysis

```bash
python scripts/process_bbq_results.py --base_folders <folder1> <folder2> ... --meta_file <metadata_csv>

```

## Accuracy and Bias scores 

```bash
python scripts/calculate_bbq_acc_and_bias_plot.py
```

```bash
python scripts/create_bbq_metrics_table.py

```

## LLM Judge 

Use the processed output files for the LLM Judge - `bbq_{category}_results_merged.json`

Judge scripts are in `reasoning_eval` folder.


## Note on Special Tokens

All scripts use the following special tokens for extracting reasoning and answers:
- `<think>` and `</think>` for reasoning content
- `<answer>` and `</answer>` for final answers

# Old Results

## Generate Reasoning Traces - Deepseek and Qwen (Old)

Generate outputs on the BBQ dataset

This uses `scripts/generate_bbq_outputs_vllm.py` and `scripts/generate_bbq_outputs_vllm_qwen.py`

```bash
sbatch generate_bbq_traces/generate_bbq.slurm deepseek-70B 16 0.6 0.95 50 2048
```

```bash
sbatch generate_bbq_traces/generate_bbq_Gender.slurm deepseek-70B 16 0.6 0.95 50 2048
```

```bash 
sbatch generate_bbq_traces/generate_bbq_qwen.slurm qwen3-32B 16 0.6 0.95 20 32768
```

```bash 
sbatch generate_bbq_traces/generate_bbq_qwen_Gender.slurm qwen3-32B 16 0.6 0.95 20 32768
```

```bash
sbatch generate_bbq_traces/generate_bbq_qwen.slurm qwen3-14B 16 0.6 0.95 20 32768
```

```bash
sbatch generate_bbq_traces/generate_bbq_qwen_Gender.slurm qwen3-14B 16 0.6 0.95 20 32768
```

Outputs saved in folder `outputs/backup_outputs`

### Evaluate Reasoning Traces

`python run_all_models_bbq_evaluation`

- Scans `backup_outputs/` for model folders named `bbq_results_{model_name}`.
- For each model:
  - Creates output directory in `eval_results/{model_name}`.
  - Finds category JSON files named `bbq_{category}_results.json`.
  - Extracts categories from filenames.
  - Runs `evaluate_bbq_outputs.py` with the model’s results, output path, and categories.

saved to `eval_results/{model_name}/`

example: `eval_results/deepseek-70B/Religion_detailed_per_trace.json`


`python scripts/evaluate_bbq_outputs_with_metadata.py`

The script will:

- Detect all models by folder names under `eval_results/`
- Detect all categories by matching JSON files for each model
- For each model-category pair:
  - Load outputs and metadata
  - Assign example IDs
  - Calculate accuracy grouped by one or more specified metadata columns (customizable in the script)
  - Save accuracy bar plots under `eval_results/{model_name}/figs/`
  - Save CSV summaries under `eval_results/{model_name}/extra_eval/`



## Acknowledgments

This repository was orginally derived from an earlier fork of [Reasoning-Towards-Fairness](https://github.com/Sanchit-404/Reasoing-Towards-Fairness), but due to complete change and diversion from this, it has been restructured into a new standalone repository.
