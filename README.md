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

### Download Models to Cache

downloads BBQ data from Hugging Face and saves them locally

```bash 
python scripts/download_data/download_bbq_all_cat.py
```

### Download Models to Cache

The `snapshot_download.py` script downloads models from Hugging Face and saves them locally

deepseek-70B (deepseek-ai/DeepSeek-R1-Distill-Llama-70B)

qwen3-32b (Qwen/Qwen3-32B)

qwen3-14b (Qwen/Qwen3-14B)

```bash
 python scripts/download_models/snapshot_download.py
```

Note: Replace `base_cache_dir` with your desired local folder.


### Generate Reasoning Traces

Generate outputs on the BBQ dataset

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



## Note on Special Tokens

All scripts use the following special tokens for extracting reasoning and answers:
- `<think>` and `</think>` for reasoning content
- `<answer>` and `</answer>` for final answers

## Acknowledgments

This repository was orginally derived from an earlier fork of [Reasoning-Towards-Fairness](https://github.com/Sanchit-404/Reasoing-Towards-Fairness), but due to substantial change and diversion from this, it has been restructured into a new standalone repository.
