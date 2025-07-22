#!/usr/bin/env python
import argparse
import json
import os
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Generate outputs on BBQ dataset using finetuned models")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the finetuned model")
    parser.add_argument("--output_dir", type=str, default="../outputs/bbq_results",
                        help="Directory to save the outputs")
    parser.add_argument("--categories", type=str, nargs="+", 
                        default=["age", "nationality", "religion"],
                        help="BBQ categories to evaluate")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum length for generation")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to process (default: all)")
    parser.add_argument("--model_type", type=str, default="finetuned",
                        choices=["finetuned", "llama3.1-8b", "mistral-7b", "phi-4"],
                        help="Type of model to use")
    return parser.parse_args()

def create_prompt(context, question):
    """Create a prompt for the model to answer a BBQ question."""
    prompt = f"""### Instruction:
Answer the following question based on the provided context. Give your reasoning in think tags like this: <think> reason </think>. Give your final answer in the answer tags like this: <answer> answer </answer>.
If the answer is not present in the context, respond with 'Not in background.'

### Input:
Context: {context}
Question: {question}

### Response:
<think> Let's think step by step. 
"""
    return prompt

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Map model type to HuggingFace model ID if needed
    model_id = args.model_path
    if args.model_type != "finetuned":
        model_mapping = {
            "llama3.1-8b": "meta-llama/Llama-3.1-8B",
            "mistral-7b": "mistralai/Mistral-7B-v0.1",
            "phi-4": "microsoft/Phi-4"
        }
        model_id = model_mapping.get(args.model_type, args.model_path)
        print(f"Using base model: {model_id}")
    else:
        print(f"Using finetuned model from: {model_id}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Ensure special tokens are available
    special_tokens = ["<think>", "</think>", "<answer>", "</answer>"]
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added} special tokens to the tokenizer")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Resize token embeddings if needed
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
    
    # Process each category
    for category in args.categories:
        print(f"Processing BBQ category: {category}")
        
        # Load the BBQ dataset for this category directly from Hugging Face
        try:
            dataset_name = f"heegyu/bbq"
            dataset = load_dataset(dataset_name, category)
            test_dataset = dataset["test"]
            
            # Limit the number of samples if specified
            if args.num_samples is not None:
                test_dataset = test_dataset.select(range(min(args.num_samples, len(test_dataset))))
            
            print(f"Loaded {len(test_dataset)} examples for category {category}")
            
            # Initialize results list
            results = []
            
            # Process examples
            for example in tqdm(test_dataset):
                context = example.get("context", "")
                question = example["question"]
                
                # Get all answer options and correct label
                answer_options = [example["ans0"], example["ans1"]]
                # Check if ans2 exists and add it
                if "ans2" in example:
                    answer_options.append(example["ans2"])
                
                correct_answer = answer_options[example["label"]]
                ambiguous = example.get("ambig", False)
                
                # Create prompt
                prompt = create_prompt(context, question)
                
                # Generate output
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_length=args.max_length,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                output = tokenizer.decode(outputs[0], skip_special_tokens=False)
                
                # Extract only the generated part (remove the prompt)
                generated_text = output[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)):]
                
                # Store the result
                result = {
                    "category": category,
                    "context": context,
                    "question": question,
                    "model_output": generated_text,
                    "correct_answer": correct_answer,
                    "ambiguous": ambiguous
                }
                
                # Add all answer options
                for i, ans in enumerate(answer_options):
                    result[f"ans{i}"] = ans
                
                result["correct_label"] = example["label"]
                results.append(result)
            
            # Save results to file
            output_file = os.path.join(args.output_dir, f"bbq_{category}_results.json")
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"Saved {len(results)} results to {output_file}")
            
        except Exception as e:
            print(f"Error processing category {category}: {e}")

if __name__ == "__main__":
    main()
