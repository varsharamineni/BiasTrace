
# Supplementary Material - Appendices, Data, Code 
## BiasTrace: Linking Reasoning Behaviours to Biased Outputs in Large Language Models

This repository contains the supplementary materials accompanying the paper, including appendices, example data, and code used for the experiments.

--

# Appendices

The supplementary appendices referenced in the paper are provided in:

- `appendices.pdf`

--

# BiasTrace Dataset

**Full dataset.** The complete BiasTrace dataset of annotated reasoning traces (approximately 250k annotated reasoning traces) will be released publicly upon publication.

The directory contains a small representative sample of the annotated reasoning traces

- `BiasTrace_ReasoningData_example.csv`
- `BiasTrace_ReasoningData_example.parquet`

--

# Codebase 

<p align="center">
  <img src="paper_figs/BiasTrace_Fig1.png" width="600">
</p>
Figure 1: Our framework applies the BiasTrace annotation scheme to link reasoning
behaviours to biased outcomes. We present a redacted trace from Qwen3-8B on the BBQ
dataset where the model should have selected answer option ‘unknown’ but instead chose
the stereotypical answer. This example demonstrates how biased outputs could arise
from different reasoning behaviours such as overthinking, rather than from stereotypical
language.
<br />


## Code Repository Structure

```
BiasTrace/
├── datasets/               # Data, templates, and metadata
├── scripts/                # Scripts for data download, reasoning generation, processing and evaluation scripts
├── reasoning_eval/         # LLM judge scripts, prompts, and ground truth samples
├── job_scripts/            # SLURM job scripts for cluster execution
├── outputs/                # Model outputs (reasoning traces + processed outputs + judge labels)
```

## Installation

**Recommended (Poetry):**
```bash
poetry install
poetry shell
source $(poetry env info --path)/bin/activate
```

**pip + requirements.txt:**

```bash
pip install -r requirements.txt
```

## Download BBQ Data 

Download all BBQ question categories from Hugging Face:
```bash 
python scripts/download_data/download_bbq_all_cat.py
# Saves to: datasets/bbq_data_all_cat/data/

```
Also download manually:
- **Templates** → `datasets/bbq_templates/` — from [BBQ GitHub](https://github.com/nyu-mll/BBQ/tree/main/templates)
- **Metadata** → `datasets/bbq_additional_metadata.csv` — from [BBQ supplemental](https://github.com/nyu-mll/BBQ/tree/main/supplemental)


## Download Model Weights

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
---

## Processing & Evaluation

### Step 1 — Merge Outputs with BBQ Data
Merges model outputs with the original BBQ dataset and optional metadata for downstream analysis.

```bash
python scripts/process_bbq_results.py \
  --base_folders <folder1> <folder2> \
  --meta_file <metadata_csv>
```

### Step 2 — Compute Accuracy and Bias Scores
```bash
python scripts/calculate_bbq_acc_and_bias_plot.py
python scripts/create_bbq_metrics_table.py
```

---

## Annotate Reasoning Traces with BiasTrace Behaviours using LLM-as-a-Judge

The LLM judge evaluates the quality of model reasoning traces. Use the merged output files (`bbq_{category}_results_merged.json`) as input.

All judge scripts are in the `reasoning_eval/` folder. 

Configure your API credentials:

```bash
LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_API_KEY=" "
```

### Validate on Test Set

**Annotate Reasoning Behaviours:**
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

## Improved Bias Evaluation on Downstream Task Using BIASTRACE Behaviours 

### Prompts

| Prompt File | Description |
|---|---|
| `baseline.txt` | Baseline rubric (0–5 score) |
| `llama70b_gt.txt` | Baseline binary rubric (0/1 score) |
| `new_prompt_edit2.txt` | **Final prompt used in paper which incorporates BiasTrace behaviours** |

FRM baseline `reasoning_eval/FRM_baseline`

## Bias Mitigation - Best of N Experiments

```bash
export LLM_API_KEY="..."             
export LLM_BASE_URL="https://api.deepseek.com/v1"   

# only if using an API-based generation script (gpt-oss, gpt-4o, ...)
export OPENAI_API_KEY="..."          
```

```bash
python bon_new/BBQ/run_pipeline.py --config bon_new/BBQ/pipeline_configs/pipeline.json
```
---