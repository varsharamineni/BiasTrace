# Reasoning Towards Fairness

This repository contains scripts for extracting reasoning traces from language models, fine-tuning models on these traces, and evaluating model performance on various datasets including BBQ and SQuAD.

## Dataset Files

The extracted datasets can be found in `data/extracted_datasets/` and include:

- `both_correct.json`: Contains examples where both models generated correct answers
- `both_incorrect.json`: Contains examples where both models generated incorrect answers
- `transformed_results.json`: Contains the complete dataset with all results

## Setup

### Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

### Download NLTK Resources

The scripts automatically download NLTK resources when needed, but you can also download them manually:

```python
import nltk
nltk.download('stopwords')
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

### Evaluate Reasoning Traces

Evaluate reasoning traces against SQuAD v2 answers, with option to compare with HuggingFace dataset:

```bash
python scripts/evaluate_reasoning_traces.py \
  --traces_file data/reasoning_traces/deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_all_traces.json \
  --output_dir evaluation/traces_evaluation \
  --compare_with_squad
```

You can also directly evaluate using just the SQuAD dataset from HuggingFace without a traces file:

```bash
python scripts/evaluate_reasoning_traces.py \
  --output_dir evaluation/traces_evaluation \
  --compare_with_squad \
  --squad_split train \
  --num_examples 100
```

## Note on Special Tokens

All scripts use the following special tokens for extracting reasoning and answers:
- `<think>` and `</think>` for reasoning content
- `<answer>` and `</answer>` for final answers

## Models Used

- For extraction of reasoning traces: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
- For fine-tuning: Llama 3.1 8B, Mistral 7B, and Phi-4

## Handling "Not in Background" Responses

The scripts handle multiple synonyms for "not in background" responses, including:
- "unknown"
- "not in background"
- "cannot be determined"
- "can't be determined"
- "not answerable"
- "not known"
- "not enough info"
- "not enough information"
- "cannot answer"
- "can't answer"
- "undetermined"
