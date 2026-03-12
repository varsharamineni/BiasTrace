# BIAS-TRACE: Linking Reasoning Behaviours to Biased Outputs in Large Language Models


## Repository Structure

```
bias-reasoning-LLM/
├── datasets/               # BBQ data, templates, and metadata
├── scripts/                # Data download, generation, and evaluation scripts
├── reasoning_eval/         # LLM judge scripts, prompts, and ground truth samples
├── job_scripts/            # SLURM job scripts for cluster execution
├── outputs/                # Model outputs (reasoning traces + merged results)
└── eval_results/           # Accuracy/bias scores and figures per model
```

## Installation

**Recommended (Poetry):**
```bash
poetry install
poetry shell
source $(poetry env info --path)/bin/activate
```
**Recommended (Poetry):**
```bash
pip install -r requirements.txt
```

## Data Setup

### Download BBQ Data 

Download all BBQ question categories from Hugging Face:
```bash 
python scripts/download_data/download_bbq_all_cat.py
# Saves to: datasets/bbq_data_all_cat/data/
```
Also download manually:
- **Templates** → `datasets/bbq_templates/` — from [BBQ GitHub](https://github.com/nyu-mll/BBQ/tree/main/templates)
- **Metadata** → `datasets/bbq_additional_metadata.csv` — from [BBQ supplemental](https://github.com/nyu-mll/BBQ/tree/main/supplemental)

You should also download the templates to `datasets/bbq_templates`[https://github.com/nyu-mll/BBQ/tree/main/templates] folder and the metadata `datasets/bbq_additional_metadata.csv`[https://github.com/nyu-mll/BBQ/tree/main/supplemental]  

### CALM Dataset Ground Truth
```bash
bash scripts/download_data/download_CALM_gt.sh
# Saves to: datasets/CALM_gt/
```

### Model Weights

Download models from Hugging Face (update `base_cache_dir` in the script first):
```bash
python scripts/download_models/snapshot_download.py
```

## Generating Reasoning Traces

Scripts are provided for Qwen3 and GPT-OSS models. Job scripts are configured for SLURM cluster execution but can be run directly.

### Qwen3 Models

**Full prompt:**
```bash
bash job_scripts/run_full.sh
# Calls: scripts/generate_bbq_outputs_vllm_qwen.py
```

**Simple prompt:**
```bash
# Edit job_scripts/run_full.sh to use:
# scripts/generate_bbq_outputs_vllm_qwen_simple.py
```

### GPT-OSS Models
```bash
bash job_scripts/run_full_gpt-oss.sh
# Calls: scripts/generate_bbq_outputs_vllm_gpt-oss.py
```

> **Special tokens:** All generation scripts use `<think>`/`</think>` for reasoning content and `<answer>`/`</answer>` for final answers.

---
## Saved Reasoning Traces

Pre-generated traces for all models are available on Google Drive:

**[Download Reasoning Traces](https://drive.google.com/drive/folders/1avhCGFPHafxfV80AtVIbEFRfT08Vsb67?usp=sharing)**

---

## Processing & Evaluation

### Step 1 — Merge Outputs with BBQ Data
```bash
python scripts/process_bbq_results.py \
  --base_folders <folder1> <folder2> \
  --meta_file <metadata_csv>
```
Merges model outputs with the original BBQ dataset and optional metadata for downstream analysis.

### Step 2 — Compute Accuracy and Bias Scores
```bash
python scripts/calculate_bbq_acc_and_bias_plot.py
python scripts/create_bbq_metrics_table.py
```

---

## LLM Judge

The LLM judge evaluates the quality of model reasoning traces. Use the merged output files (`bbq_{category}_results_merged.json`) as input.

All judge scripts are in the `reasoning_eval/` folder.

### Validate on Test Set

**Annotate:**
```bash
python reasoning_eval/llm_judge_script_vllm.py \
  --model deepseek-chat \
  --prompt new_prompt_edit2 \
  --output_dir reasoning_eval/llm_judge_samples/test_set/our_labels \
  --data_path reasoning_eval/ground_truth_samples/test_set.json \
  --temperature 1.0 --top_p 0.9 \
  --reasoning_prompt_text "<think>\nPlease carefully reason through the given reasoning trace step by step"
```

**Evaluate against human labels:**
```bash
python reasoning_eval/compare_llm_to_human.py \
  --human_file reasoning_eval/ground_truth_samples/test_set.json \
  --llm_folder reasoning_eval/llm_judge_samples/test_set/our_labels \
  --output_prefix reasoning_eval/llm_judge_eval_metrics_val
```

### Run on Full BBQ Dataset

```bash
CATEGORIES=(Age Disability_status Gender_identity Nationality Physical_appearance Race_ethnicity Religion SES Sexual_orientation)

for CAT in "${CATEGORIES[@]}"; do
  python reasoning_eval/llm_judge_script_vllm_multiple.py \
    --model deepseek-chat \
    --prompt new_prompt_edit2 \
    --output_dir outputs/qwen_full_8B_full_prompt/full_annotation/${CAT}/ \
    --temperature 1.0 --top_p 0.9 \
    --data_paths outputs/qwen_full_8B_full_prompt/bbq_${CAT}_results_merged.json
done
```

---

## Prompts

| Prompt File | Description |
|---|---|
| `baseline.txt` | Baseline rubric (0–5 score) |
| `llama70b_gt.txt` | Baseline binary rubric (0/1 score) |
| `new_prompt_edit2.txt` | **Final prompt used in paper** |

---

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
