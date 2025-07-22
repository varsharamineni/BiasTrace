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

### Extract Reasoning Traces

Extract reasoning traces from a model on SQuAD v2 loaded directly from Hugging Face:

```bash
python scripts/extract_reasoning_traces.py \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --output_dir data/reasoning_traces \
  --num_examples 100 \
  --dataset_split train
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

### Generate Outputs on BBQ Dataset

Generate outputs on the BBQ dataset loaded from Hugging Face:

```bash
python scripts/generate_bbq_outputs.py \
  --model_path models/finetuned-llama \
  --categories age nationality religion \
  --output_dir outputs/bbq_results
```

### Evaluate BBQ Outputs

Evaluate model outputs on the BBQ dataset with option to reference original HuggingFace data:

```bash
python scripts/evaluate_bbq_outputs.py \
  --results_dir outputs/bbq_results \
  --output_dir evaluation/bbq_evaluation \
  --reference_data
```

## Note on Special Tokens

All scripts use the following special tokens for extracting reasoning and answers:
- `<think>` and `</think>` for reasoning content
- `<answer>` and `</answer>` for final answers


## Acknowledgments

This repository is derived from an earlier fork of [original-repo](https://github.com/Sanchit-404/Reasoing-Towards-Fairness), but due to substantial structural and conceptual changes, it has been restructured into a new standalone repository.
