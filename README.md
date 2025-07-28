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

The `snapshot_download.py` script downloads models from Hugging Face and saves them locally so you don’t have to download them again.

deepseek-70B (deepseek-ai/DeepSeek-R1-Distill-Llama-70B)

qwen3-32b (Qwen/Qwen3-32B)

qwen3-14b (Qwen/Qwen3-14B)

```bash
python snapshot_download.py
```

Note: Replace `base_cache_dir` with your desired local folder.


### Generate Reasoning Traces

Generate outputs on the BBQ dataset loaded from Hugging Face:

```bash
python scripts/generate_bbq_outputs_vllm.py \
  --model_path models/... \
  --categories Age Nationality Religion \
  --output_dir outputs/...
```

### Evaluate Reasoning Traces

Evaluate model outputs on the BBQ dataset 

```bash
python scripts/evaluate_bbq_outputs.py \
  --results_dir outputs/bbq_results \
  --output_dir evaluation/bbq_evaluation \
  --reference_data
```


### Fine-tune Models on Reasoning Traces

Fine-tune models (Llama 3.1 8B, Mistral 7B, or Phi-4) on reasoning traces:

```bash
python scripts/finetune_on_traces.py \
  --model_name llama3.1-8b \
  --traces_file data/reasoning_traces/deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_correct_traces.json \
  --output_dir models/finetuned-llama \
  --num_epochs 3
```


## Note on Special Tokens

All scripts use the following special tokens for extracting reasoning and answers:
- `<think>` and `</think>` for reasoning content
- `<answer>` and `</answer>` for final answers

## Acknowledgments

This repository was orginally derived from an earlier fork of [Reasoning-Towards-Fairness](https://github.com/Sanchit-404/Reasoing-Towards-Fairness), but due to substantial change and diversion from this, it has been restructured into a new standalone repository.
