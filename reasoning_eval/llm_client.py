import argparse
import json
import os
import re
from typing import Tuple, List, Dict, Any

from tqdm import tqdm
import logging

# Suppress vLLM's tqdm progress bars
logging.getLogger("vllm.engine.llm_engine").setLevel(logging.WARNING)
logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)
from vllm import LLM, SamplingParams
from datasets import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Generate outputs on BBQ dataset using Qwen3 8B with thinking mode")
    parser.add_argument("--model", "--model_path", type=str, default="Qwen/Qwen3-4B", help="Path to the model or HuggingFace model ID (default: Qwen/Qwen3-8B)")
    parser.add_argument("--output_dir", type=str, default="../outputs/bbq_results", help="Directory to save the outputs")
    parser.add_argument("--categories", type=str, nargs="+", default=["Age", "Nationality", "Religion"], help="BBQ categories to evaluate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for VLLM inference (increased for better throughput)")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process per category (default: all)")
    parser.add_argument("--test_mode", action="store_true", help="Test mode: process only 10 samples per category")
    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (Qwen recommends 0.6 for thinking mode)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling (Qwen default)")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling (Qwen recommends 20)")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum generation length (increased for thinking mode)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation (default: 42)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    parser.add_argument("--enable_thinking", action="store_true", default=True, help="Enable thinking mode for Qwen (default: True)")
    parser.add_argument("--quiet", action="store_true", help="Suppress warning messages for missing tags")

    return parser.parse_args()

def process_batch(llm: LLM, batch_data: List[Dict[str, Any]], 
                 sampling_params: SamplingParams, batch_start_idx: int = 0,
                 enable_thinking: bool = True, quiet: bool = False) -> List[Dict[str, Any]]:
    """Process a batch of examples with optimized vLLM generation."""
    messages_batch = []
    for item in batch_data:
        messages = create_messages(item['context'], item['question'], item['answer_options'])
        messages_batch.append(messages)
    
    # Generate outputs using vLLM's chat method with thinking mode
    # Disable internal progress bar by using use_tqdm=False
    outputs = llm.chat(
        messages_batch, 
        sampling_params,
        chat_template_kwargs={"enable_thinking": enable_thinking},
        use_tqdm=False  # Disable vLLM's internal tqdm
    )
    
    results = []
    for idx, (output, item) in enumerate(zip(outputs, batch_data)):
        generated_text = output.outputs[0].text
        
        # Create item identifier for debugging
        item_id = f"{item['category']}_idx{batch_start_idx + idx}"
        if 'id' in item:
            item_id = f"{item['category']}_id{item['id']}"
        
        # Extract thinking and answer with item ID for warnings
        thinking, extracted_answer, normalized_answer = extract_reasoning_and_answer(
            generated_text, item['answer_options'], item_id, quiet
        )
        
        # Determine if the answer is correct
        correct_answer = item['answer_options'][item['label']]
        is_correct = normalized_answer == correct_answer
        
        result = {
            "category": item['category'],
            "context": item['context'],
            "question": item['question'],
            "answer_options": item['answer_options'],
            "model_output": generated_text.strip(),
            "model_reasoning": thinking,  # Using thinking content
            "model_answer": normalized_answer,  # Key name as requested
            "extracted_answer": extracted_answer,
            "normalized_answer": normalized_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "ambiguous": item.get('ambig', False),
            "correct_label": item['label'],
        }
        
        # Add individual answer options for compatibility
        for idx, ans in enumerate(item['answer_options']):
            result[f"ans{idx}"] = ans
        
        results.append(result)
    
    return results

def main():
    args = parse_args()
    enable_thinking: bool = True
    
    # Override num_samples if in test mode
    if args.test_mode:
        args.num_samples = 10
        print("TEST MODE: Processing only 10 samples per category")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize vLLM model with optimized settings
    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,  # Required for Qwen models
        max_model_len=32768,  # Qwen context length
        enable_prefix_caching=True,  # Enable prefix caching for better batching performance
        enforce_eager=False,  # Use CUDA graphs for better performance
        disable_log_stats=True,  # Disable vLLM's internal logging
    )
    
    # Optimized sampling parameters for Qwen thinking mode (based on official recommendations)
    sampling_params = SamplingParams(
        max_tokens=args.max_length,
        temperature=args.temperature,  # Use user override or default 0.6 for thinking mode
        top_p=args.top_p,  # Use user override or default 0.95
        top_k=args.top_k,  # Use user override or default 20 for thinking mode
        stop=["<|endoftext|>", "<|im_end|>", "<|im_start|>"],  # Qwen specific stop tokens
        skip_special_tokens=False,  # Keep special tokens for proper formatting
        seed=args.seed,  # Set seed for reproducibility
    )

    messages = [
        {"role": "user", "content": "What is your name?"}
    ]

    outputs = llm.chat(
        messages,
        sampling_params,
        chat_template_kwargs={"enable_thinking": enable_thinking},
        use_tqdm=False  # Disable vLLM's internal tqdm
    )

    for output in outputs:
        print(output)

if __name__ == "__main__":
    main()