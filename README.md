# Bias-Reasoning-LLM

## Dataset Files

## Setup

### Installation

We recommend using Poetry for dependency management:

```bash
poetry install
poetry shell
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

### Download Models to Cache

The `snapshot_download.py` script downloads models from Hugging Face and saves them locally

Note: Replace `base_cache_dir` with your desired local folder

```bash
 python scripts/download_models/snapshot_download.py
```

- deepseek-70B (deepseek-ai/DeepSeek-R1-Distill-Llama-70B)

- qwen3-32B (Qwen/Qwen3-32B)

- qwen3-14B (Qwen/Qwen3-14B)

- qwen3-8B (Qwen/Qwen3-8B)

```bash
 python scripts/download_models/snapshot_download.py
```

## Generate Reasoning Traces - Qwen 

With full prompt
`scripts/generate_bbq_outputs_vllm_qwen.py`

With simple prompt
`scripts/generate_bbq_outputs_vllm_qwen_simple.py`

Can use job_scripts to run these and get reasoning traces 

```bash
job_scripts/run_full.sh
``` 

## Saved Reasoning Traces

Google Drive link

[Google Drive Link](https://drive.google.com/drive/folders/1avhCGFPHafxfV80AtVIbEFRfT08Vsb67?usp=sharing)


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

## Note on Special Tokens

All scripts use the following special tokens for extracting reasoning and answers:
- `<think>` and `</think>` for reasoning content
- `<answer>` and `</answer>` for final answers

## Acknowledgments

This repository was orginally derived from an earlier fork of [Reasoning-Towards-Fairness](https://github.com/Sanchit-404/Reasoing-Towards-Fairness), but due to substantial change and diversion from this, it has been restructured into a new standalone repository.
