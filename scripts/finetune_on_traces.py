#!/usr/bin/env python
import argparse
import json
import os
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune models on reasoning traces")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pretrained model to finetune (llama3.1-8b, mistral-7b, or phi-4)")
    parser.add_argument("--traces_file", type=str, required=True,
                        help="JSON file containing reasoning traces")
    parser.add_argument("--output_dir", type=str, default="../models/finetuned_model",
                        help="Directory to save the finetuned model")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--num_gpus", type=int, default=4,
                        help="Number of GPUs to use for training")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="Rank of the LoRA adapter")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="Alpha parameter for LoRA scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout probability for LoRA layers")
    return parser.parse_args()

def format_prompt(example):
    """Format the prompt for finetuning, matching inference style."""
    context = example["context"]
    question = example["question"]
    reasoning_trace = example.get("reasoning_trace", "")
    extracted_reasoning = example.get("extracted_reasoning", "")
    extracted_answer = example.get("extracted_answer", "")
    
    # If we have the extracted reasoning and answer separately, use them
    if extracted_reasoning and extracted_answer:
        formatted_trace = f"<think> {extracted_reasoning} </think>\n<answer> {extracted_answer} </answer>"
    else:
        # Otherwise use the full reasoning trace
        formatted_trace = reasoning_trace
    
    # Using the format consistent with our extraction and evaluation scripts
    prompt = f"""### Instruction:
Answer the following question based on the provided context. Give your reasoning in think tags like this: <think> reason </think>. Give your final answer in the answer tags like this: <answer> answer </answer>.
If the answer is not present in the context, respond with 'Not in background.'

### Input:
Context: {context}
Question: {question}

### Response:
{formatted_trace}"""
    
    return prompt

def load_and_prepare_data(traces_file, tokenizer, max_length):
    """Load traces from a file and prepare them for training."""
    logger.info(f"Loading reasoning traces from {traces_file}")
    
    # Load reasoning traces
    with open(traces_file, "r") as f:
        traces = json.load(f)
    
    logger.info(f"Loaded {len(traces)} reasoning traces")
    
    # Format prompts and labels
    prompts = []
    labels = []
    for trace in traces:
        prompt = format_prompt(trace)
        # The label is the same as the prompt since we're doing causal language modeling
        prompts.append(prompt)
        labels.append(prompt)
    
    # Create a dataset from the prompts and labels
    dataset = Dataset.from_dict({
        "text": prompts,
        "labels": labels
    })
    
    # Tokenize the dataset
    def tokenize_function(examples):
        # Tokenize both inputs and labels
        tokenized_inputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        tokenized_labels = tokenizer(
            examples["labels"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Set the labels to the input_ids of the labels
        tokenized_inputs["labels"] = tokenized_labels["input_ids"]
        
        return tokenized_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "labels"]
    )
    
    return tokenized_dataset

def main():
    args = parse_args()
    
    # Set environment variables for distributed training
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(args.num_gpus)])
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Map model name argument to actual HuggingFace model name
    model_name_mapping = {
        "llama3.1-8b": "meta-llama/Llama-3.1-8B",
        "mistral-7b": "mistralai/Mistral-7B-v0.1",
        "phi-4": "microsoft/Phi-4"
    }
    
    # Resolve model name
    model_id = args.model_name
    for key, value in model_name_mapping.items():
        if key.lower() in args.model_name.lower():
            model_id = value
            break
    
    logger.info(f"Loading model and tokenizer: {model_id}")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding=True, padding_side="left", add_eos_token=False)
    # Ensure padding token is properly set
    # Ensure padding token is properly set
    tokenizer.pad_token = '<|finetune_right_pad_id|>'
    
    # Add special tokens for reasoning and answer
    special_tokens = {"additional_special_tokens": ["<think>", "</think>", "<answer>", "</answer>"]}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load dataset
    train_dataset = load_and_prepare_data(args.traces_file, tokenizer, args.max_length)
    
    # Load model with quantization for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Resize token embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"]
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params} ({trainable_params/total_params:.2%} of {total_params})")
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
        label_pad_token_id=tokenizer.pad_token_id
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        prediction_loss_only=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
