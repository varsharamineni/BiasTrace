import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from vLLM_client import vLLMClient
from vllm import SamplingParams, LLM
from prompt_manager import PromptManager


# ================================================================
# Utility functions
# ================================================================
def load_reasoning_data(path: str) -> List[Dict[str, Any]]:
    """Load model reasoning traces or evaluation data."""
    with open(path, "r") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} samples from {path}")
    return data


def create_messages(prompt: str) -> List[Dict[str, str]]:
    """Format prompt into vLLM message structure."""
    return [{"role": "user", "content": prompt}]


def build_batch_messages(
    batch_data: List[Dict[str, Any]],
    pm: PromptManager,
    prompt_key: str,
) -> List[List[Dict[str, str]]]:
    """Generate batch of chat-style messages for the model."""
    messages_batch = []
    for item in batch_data:
        prompt = pm.get_prompt(
            prompt_key,
            reasoning_trace=item.get("model_reasoning", ""),
            final_answer=item.get("model_answer", ""),
            category=item.get("bbq_category", ""),
            context=item.get("context", ""),
            question=item.get("question", ""),
            answer_options=item.get("answer_options", []),
            sample_id=item.get("sample_id", ""),
            example_id=item.get("example_id", ""),
            model=item.get("model", ""),
            prompt_type=item.get("prompt_type","")
        )
        messages_batch.append(create_messages(prompt))
    return messages_batch


def run_llm_evaluation(
    model_path: str,
    messages_batch: List[List[Dict[str, str]]],
    enable_thinking: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: int = 42,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # make both GPUs visible

    """Run vLLM evaluation for a batch of messages."""
    print(f"🚀 Loading model: {model_path}")
    client = vLLMClient(model=model_path, tensor_parallel_size=2)
    llm = client.load_vllm()

    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=["<|endoftext|>", "<|im_end|>", "<|im_start|>"],
        skip_special_tokens=False,
        seed=seed,
    )

    print(f"💬 Running inference on {len(messages_batch)} samples...")
    outputs = llm.chat(
        messages_batch,
        sampling_params,
        chat_template_kwargs={"enable_thinking": enable_thinking},
        use_tqdm=True,
    )

    return outputs


import re
def extract_json_from_thinking(text):
    """Extract the JSON object at the end of a reasoning trace."""
    match = re.search(r'\{[\s\S]*\}$', text.strip())
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {"raw_text": text}
    return {"raw_text": text}


def parse_outputs(batch_data, outputs, model_name, prompt_key) -> List[Dict[str, Any]]:
    results = []
    for item, output in zip(batch_data, outputs):
        text = output.outputs[0].text  # full reasoning + answer
        parsed = extract_json_from_thinking(text)

        results.append({
            "sample_id": item.get("sample_id", ""),
            "category": item.get("bbq_category", ""),
            "example_id": item.get("example_id", ""),
            "model": item.get("model", ""),
            "prompt_type": item.get("prompt_type", ""),
            "judge_model": model_name,
            "judge_prompt": prompt_key,
            "reasoning_text": text,      # full reasoning preserved
            "judge_output": parsed,      # clean parsed JSON
        })
    return results


def save_results(results, model_name, output_dir, prompt_key, sampling_params, enable_thinking=False):
    """Save evaluation results along with metadata to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct final dict with metadata + outputs
    output_data = {
        "metadata": {
            "judge_model": model_name,
            "judge_prompt": prompt_key,
            "enable_thinking": enable_thinking,
            "sampling_params": {
                "max_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
                "top_p": sampling_params.top_p,
                "top_k": sampling_params.top_k,
                "stop": sampling_params.stop,
                "skip_special_tokens": sampling_params.skip_special_tokens,
                "seed": sampling_params.seed,
            }
        },
        "results": results
    }

    # Include thinking mode in filename
    thinking_flag = "_thinking" if enable_thinking else ""
    filename = f"{output_dir}/llm_eval_{model_name.replace('/', '_')}_{prompt_key}{thinking_flag}.json"
    
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"✅ Saved {len(results)} results + metadata to {filename}")
    return filename


# ================================================================
# Main script logic
# ================================================================
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 1. Load data
    data = load_reasoning_data(args.data_path)

    # Optional: subset for debugging
    if args.max_samples:
        data = data[:args.max_samples]

    # 2. Load prompt manager
    pm = PromptManager(args.prompts)

    # 3. Create batch messages
    messages_batch = build_batch_messages(data, pm, args.prompt)


    # 4. Run inference

    sampling_params = SamplingParams(
    max_tokens=2048,
    temperature=args.temperature,
    top_p=args.top_p,
    top_k=args.top_k,
    stop=["<|endoftext|>", "<|im_end|>", "<|im_start|>"],
    skip_special_tokens=False,
    seed=args.seed,
)
    outputs = run_llm_evaluation(
        model_path=args.model,
        messages_batch=messages_batch,
        enable_thinking=args.enable_thinking,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
)

    # 5. Parse and save
    results = parse_outputs(data, outputs, args.model, args.prompt)
    save_results(results, args.model, args.output_dir, args.prompt, sampling_params, enable_thinking=args.enable_thinking)


# ================================================================
# CLI Entry Point
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate reasoning traces with different models and prompts.")
    parser.add_argument("--model", type=str, required=True, help="Path or name of model to use (e.g., Qwen/Qwen3-4B).")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt key in prompts.json to use.")
    parser.add_argument("--data_path", type=str, default="reasoning_eval/data_to_label/sample_traces_inital.json")
    parser.add_argument("--prompts", type=str, default="reasoning_eval/judge_prompts")
    parser.add_argument("--output_dir", type=str, default="reasoning_eval/llm_judge_samples/")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable Qwen 'thinking mode'")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for quick runs.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
