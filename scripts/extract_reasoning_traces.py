#!/usr/bin/env python
import argparse
import json
import os
from tqdm import tqdm
import random
import re
import string
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk
from nltk.corpus import stopwords
from datasets import load_dataset

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def parse_args():
    parser = argparse.ArgumentParser(description="Extract reasoning traces from a model on SQuAD v2 dataset")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--output_dir", type=str, default="../data/reasoning_traces",
                        help="Directory to save reasoning traces")
    parser.add_argument("--num_examples", type=int, default=100,
                        help="Number of examples to process")
    parser.add_argument("--max_context_length", type=int, default=1024,
                        help="Maximum context length for the model")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--dataset_split", type=str, default="validation",
                        help="SQuAD dataset split to use (validation or train)")
    return parser.parse_args()

def create_prompt(context, question):
    """Create a prompt for the model to extract reasoning traces."""
    # Specific prompt for DeepSeek models
    prompt = f"""### Instruction:
Answer the following question based on the provided context. Think through the problem step by step and then provide your final answer.
If the answer is not present in the context, respond with 'Not in background'.

Context:
{context}

Question:
{question}

### Response:
<think> Let's think step by step to answer the question.
"""
    return prompt

def normalize_answer(s):
    """Lower text and remove punctuation, articles, and stopwords."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
        
    def remove_stopwords(text):
        words = text.split()
        return ' '.join([word for word in words if word.lower() not in STOPWORDS])

    return remove_stopwords(white_space_fix(remove_articles(remove_punc(lower(s)))))

def is_answer_correct(predicted, ground_truth):
    """Check if the normalized ground truth is contained in the normalized prediction."""
    if not predicted or not ground_truth:
        return False
        
    normalized_pred = normalize_answer(predicted)
    normalized_truth = normalize_answer(ground_truth)
    
    # Handle "not in background" synonyms
    not_in_background_synonyms = [
        "unknown", "not in background", "cannot be determined", 
        "can't be determined", "not answerable", "not known", 
        "not enough info", "not enough information", "cannot answer", 
        "can't answer", "undetermined"
    ]
    
    if any(syn.lower() in normalized_pred.lower() for syn in not_in_background_synonyms) and \
       any(syn.lower() in normalized_truth.lower() for syn in not_in_background_synonyms):
        return True
    
    return normalized_truth in normalized_pred

def extract_reasoning_and_answer(reasoning_trace):
    """Extract reasoning and answer from reasoning traces using tags."""
    # Extract content between <think> and </think> tags
    reasoning_pattern = r'<think>(.*?)</think>'
    reasoning_matches = re.findall(reasoning_pattern, reasoning_trace, re.DOTALL)
    reasoning = reasoning_matches[0].strip() if reasoning_matches else ""
    
    # Extract content between <answer> and </answer> tags
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = re.findall(answer_pattern, reasoning_trace, re.DOTALL)
    answer = answer_matches[0].strip() if answer_matches else ""
    
    # If no answer tag found, check for text after reasoning content
    if not answer:
        # Look for answer in the remaining text after the reasoning
        if reasoning and "</think>" in reasoning_trace:
            remainder = reasoning_trace.split("</think>", 1)[1]
            # Clean up the remainder text
            answer = remainder.strip()
    
    return reasoning, answer

def main():
    args = parse_args()
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Using model: {args.model_name}")
    if "deepseek" in args.model_name.lower():
        print("Detected DeepSeek model, using appropriate prompting")
    
    # Load SQuAD data directly from Hugging Face
    print(f"Loading SQuAD v2 dataset from Hugging Face ({args.dataset_split} split)...")
    squad_dataset = load_dataset("squad_v2", split=args.dataset_split)
    
    # Extract examples from SQuAD data
    examples = []
    for item in squad_dataset:
        examples.append({
            "id": item["id"],
            "context": item["context"],
            "question": item["question"],
            "answers": item["answers"]["text"],  # List of answer texts
            "is_impossible": len(item["answers"]["text"]) == 0
        })
    
    print(f"Loaded {len(examples)} examples from SQuAD v2")
    
    # Randomly sample examples
    if args.num_examples < len(examples):
        examples = random.sample(examples, args.num_examples)
    
    print(f"Processing {len(examples)} examples...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Add special tokens for reasoning and answer
    special_tokens = {"additional_special_tokens": ["<think>", "</think>", "<answer>", "</answer>"]}
    tokenizer.add_special_tokens(special_tokens)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Resize token embeddings if we added special tokens
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    
    model = model.cuda()
    model.eval()
    
    # Process examples
    reasoning_traces = []
    correct_reasoning_traces = []
    incorrect_reasoning_traces = []
    
    for example in tqdm(examples):
        context = example["context"]
        question = example["question"]
        gold_answers = example["answers"]
        
        # Create prompt
        prompt = create_prompt(context, question)
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_context_length)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate reasoning trace
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        reasoning_trace = generated_text[len(prompt):]
        
        # Extract reasoning and answer from the trace
        reasoning, answer = extract_reasoning_and_answer(reasoning_trace)
        
        # Check if the answer is correct
        is_correct = False
        if gold_answers:
            is_correct = any(is_answer_correct(answer, gold_answer) for gold_answer in gold_answers)
        elif example["is_impossible"] and ("unanswerable" in answer.lower() or "cannot be answered" in answer.lower() or "no answer" in answer.lower()):
            is_correct = True
        
        # Construct the trace object
        trace = {
            "id": example["id"],
            "context": context,
            "question": question,
            "gold_answers": gold_answers,
            "reasoning_trace": reasoning_trace,
            "extracted_reasoning": reasoning,
            "extracted_answer": answer,
            "is_correct": is_correct,
            "is_impossible": example["is_impossible"]
        }
        
        reasoning_traces.append(trace)
        
        # Sort into correct and incorrect traces
        if is_correct:
            correct_reasoning_traces.append(trace)
        else:
            incorrect_reasoning_traces.append(trace)
    
    # Save all traces
    all_traces_file = os.path.join(args.output_dir, f"{args.model_name.replace('/', '_')}_all_traces.json")
    with open(all_traces_file, "w") as f:
        json.dump(reasoning_traces, f, indent=2)
    
    # Save correct traces
    correct_traces_file = os.path.join(args.output_dir, f"{args.model_name.replace('/', '_')}_correct_traces.json")
    with open(correct_traces_file, "w") as f:
        json.dump(correct_reasoning_traces, f, indent=2)
    
    # Save incorrect traces
    incorrect_traces_file = os.path.join(args.output_dir, f"{args.model_name.replace('/', '_')}_incorrect_traces.json")
    with open(incorrect_traces_file, "w") as f:
        json.dump(incorrect_reasoning_traces, f, indent=2)
    
    print(f"Processed {len(reasoning_traces)} examples")
    print(f"Correct: {len(correct_reasoning_traces)}, Incorrect: {len(incorrect_reasoning_traces)}")
    print(f"Traces saved to {args.output_dir}")

if __name__ == "__main__":
    main() 