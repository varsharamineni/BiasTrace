#!/usr/bin/env python
import argparse
import json
import os
import re
import string
from collections import Counter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
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
    parser = argparse.ArgumentParser(description="Evaluate quality of reasoning traces")
    parser.add_argument("--traces_file", type=str, default=None,
                        help="JSON file containing reasoning traces")
    parser.add_argument("--output_dir", type=str, default="../evaluation/traces_evaluation",
                        help="Directory to save the evaluation results")
    parser.add_argument("--compare_with_squad", action="store_true",
                        help="Compare with original SQuAD dataset from HuggingFace")
    parser.add_argument("--squad_split", type=str, default="validation",
                        help="SQuAD dataset split to use (validation or train)")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Number of examples to evaluate (default: all)")
    return parser.parse_args()

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
    """Check if normalized ground truth is contained in normalized prediction."""
    if not predicted or not ground_truth:
        return False
    
    normalized_prediction = normalize_answer(predicted)
    normalized_ground_truth = normalize_answer(ground_truth)
    
    # Check for "not in background" synonyms
    not_in_background_synonyms = [
        "unknown", "not in background", "cannot be determined", 
        "can't be determined", "not answerable", "not known", 
        "not enough info", "not enough information", "cannot answer", 
        "can't answer", "undetermined"
    ]
    
    # If both prediction and ground truth indicate "not in background", consider it correct
    if any(syn.lower() in normalized_prediction.lower() for syn in not_in_background_synonyms) and \
       any(syn.lower() in normalized_ground_truth.lower() for syn in not_in_background_synonyms):
        return True
    
    return normalized_ground_truth in normalized_prediction

def extract_reasoning_and_answer(reasoning_trace):
    """Extract reasoning and answer from reasoning trace using think and answer tags."""
    # Extract content between <think> and </think> tags
    reasoning_pattern = r'<think>(.*?)</think>'
    reasoning_matches = re.findall(reasoning_pattern, reasoning_trace, re.DOTALL)
    reasoning = reasoning_matches[0].strip() if reasoning_matches else ""
    
    # Extract content between <answer> and </answer> tags
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = re.findall(answer_pattern, reasoning_trace, re.DOTALL)
    answer = answer_matches[0].strip() if answer_matches else ""
    
    # If no answer tag found, check if there's text after the </think> tag
    if not answer and "</think>" in reasoning_trace:
        parts = reasoning_trace.split("</think>", 1)
        if len(parts) > 1:
            remainder = parts[1].strip()
            answer = remainder
    
    # If still no answer or reasoning, check for "not in background" synonyms
    not_in_background_synonyms = [
        "unknown", "not in background", "cannot be determined", 
        "can't be determined", "not answerable", "not known", 
        "not enough info", "not enough information", "cannot answer", 
        "can't answer", "undetermined"
    ]
    
    if not answer:
        for syn in not_in_background_synonyms:
            if syn.lower() in reasoning_trace.lower():
                answer = "Not in background"
                break
    
    return reasoning, answer

def evaluate_reasoning_quality(reasoning):
    """Evaluate the quality of reasoning steps."""
    # Count the number of reasoning steps
    steps = len(re.findall(r'(?:Step \d+:|First,|Second,|Third,|Next,|Then,|Finally,)', reasoning, re.IGNORECASE))
    
    # Check if the reasoning includes references to the context
    has_context_reference = bool(re.search(r'(?:according to the context|in the context|the context states|the passage says|the passage mentions)', reasoning, re.IGNORECASE))
    
    # Check if there's evidence of logical reasoning
    has_logical_reasoning = bool(re.search(r'(?:therefore|thus|because|since|as a result|this means|this implies|this suggests)', reasoning, re.IGNORECASE))
    
    # Calculate a reasoning quality score (simple heuristic)
    reasoning_score = min(1.0, (steps + has_context_reference + has_logical_reasoning) / 5.0)
    
    return {
        "num_steps": steps,
        "has_context_reference": has_context_reference,
        "has_logical_reasoning": has_logical_reasoning,
        "reasoning_score": reasoning_score
    }

def generate_visualizations(eval_results, output_dir):
    """Generate visualizations of evaluation results."""
    # Create a pie chart for correct vs incorrect answers
    plt.figure(figsize=(10, 6))
    labels = ['Correct Answers', 'Incorrect Answers']
    sizes = [eval_results["num_correct"], eval_results["num_examples"] - eval_results["num_correct"]]
    colors = ['#3A923A', '#D62728']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Accuracy of Extracted Answers')
    plt.savefig(os.path.join(output_dir, "answer_accuracy.png"))
    plt.close()
    
    # Create a bar chart for reasoning quality metrics
    metrics = ["Avg Steps", "Context References (%)", "Logical Reasoning (%)", "Reasoning Score"]
    values = [
        eval_results["avg_reasoning_steps"],
        eval_results["context_reference_percent"],
        eval_results["logical_reasoning_percent"],
        eval_results["avg_reasoning_score"]
    ]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics, values, color=['#3274A1', '#E1812C', '#3A923A', '#9467BD'])
    plt.title('Reasoning Quality Metrics')
    plt.ylabel('Score')
    plt.ylim(0, max(values) * 1.2)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f"{values[i]:.2f}", ha='center')
    
    plt.savefig(os.path.join(output_dir, "reasoning_quality.png"))
    plt.close()
    
    # Create a scatter plot comparing reasoning quality vs answer correctness
    plt.figure(figsize=(10, 6))
    correct_scores = [result["reasoning_evaluation"]["reasoning_score"] for result in eval_results["correct_reasoning"]]
    incorrect_scores = [result["reasoning_evaluation"]["reasoning_score"] for result in eval_results["incorrect_reasoning"]]
    
    # Create dummy y values for the scatter plot
    correct_y = [1] * len(correct_scores)
    incorrect_y = [0] * len(incorrect_scores)
    
    plt.scatter(correct_scores, correct_y, alpha=0.5, c='green', label='Correct Answers')
    plt.scatter(incorrect_scores, incorrect_y, alpha=0.5, c='red', label='Incorrect Answers')
    
    plt.yticks([0, 1], ['Incorrect', 'Correct'])
    plt.xlabel('Reasoning Quality Score')
    plt.title('Relationship Between Reasoning Quality and Answer Correctness')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, "reasoning_vs_correctness.png"))
    plt.close()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load reasoning traces
    traces = []
    
    if args.traces_file and os.path.exists(args.traces_file):
        with open(args.traces_file, "r") as f:
            traces = json.load(f)
        print(f"Loaded {len(traces)} reasoning traces from {args.traces_file}")
    elif args.compare_with_squad:
        # We'll create traces on the fly for comparison
        print(f"No traces file provided, will generate comparison using SQuAD dataset")
    else:
        print("Error: Either --traces_file or --compare_with_squad must be provided")
        return
    
    # If comparing with SQuAD dataset from HuggingFace
    if args.compare_with_squad:
        print(f"Loading SQuAD v2 dataset from HuggingFace ({args.squad_split} split)...")
        squad_dataset = load_dataset("squad_v2", split=args.squad_split)
        
        if args.num_examples:
            squad_dataset = squad_dataset.select(range(min(args.num_examples, len(squad_dataset))))
            
        print(f"Loaded {len(squad_dataset)} examples from SQuAD v2")
        
        # Create a mapping from question IDs to SQuAD examples for validation
        squad_map = {item["id"]: {
            "context": item["context"],
            "question": item["question"],
            "answers": item["answers"]["text"],
            "is_impossible": len(item["answers"]["text"]) == 0
        } for item in squad_dataset}
        
        # If we have traces, validate them against SQuAD data
        if traces:
            print("Validating traces against SQuAD dataset...")
            validated_traces = []
            for trace in traces:
                trace_id = trace.get("id", "")
                if trace_id in squad_map:
                    # Update with the official SQuAD data
                    squad_example = squad_map[trace_id]
                    trace["context"] = squad_example["context"]
                    trace["question"] = squad_example["question"]
                    trace["gold_answers"] = squad_example["answers"]
                    trace["is_impossible"] = squad_example["is_impossible"]
                    validated_traces.append(trace)
            
            if validated_traces:
                print(f"Validated {len(validated_traces)} traces with SQuAD dataset")
                traces = validated_traces
        else:
            # Create simple placeholder traces for evaluation
            print("Creating placeholder traces from SQuAD dataset...")
            traces = []
            for item_id, item in squad_map.items():
                trace = {
                    "id": item_id,
                    "context": item["context"],
                    "question": item["question"],
                    "gold_answers": item["answers"],
                    "is_impossible": item["is_impossible"],
                    "reasoning_trace": "",  # No reasoning or answers since we're just testing
                    "extracted_reasoning": "",
                    "extracted_answer": ""
                }
                traces.append(trace)
    
    # Limit the number of examples if specified
    if args.num_examples and len(traces) > args.num_examples:
        traces = traces[:args.num_examples]
        print(f"Limiting evaluation to {args.num_examples} examples")
    
    # Initialize metrics
    num_correct = 0
    num_incorrect = 0
    reasoning_quality_scores = []
    
    # Process each trace
    for i, trace in enumerate(traces):
        # Extract key information
        context = trace.get("context", "")
        question = trace.get("question", "")
        gold_answers = trace.get("gold_answers", trace.get("answers", []))
        reasoning_trace = trace.get("reasoning_trace", "")
        is_impossible = trace.get("is_impossible", False)
        
        # Extract reasoning and answer from the trace
        reasoning, answer = extract_reasoning_and_answer(reasoning_trace)
        
        # Check if the answer is correct
        is_correct = False
        if gold_answers:
            is_correct = any(is_answer_correct(answer, gold_answer) for gold_answer in gold_answers)
        elif is_impossible and ("unanswerable" in answer.lower() or "cannot be answered" in answer.lower() or "not in background" in answer.lower()):
            is_correct = True
        
        # Count correct and incorrect answers
        if is_correct:
            num_correct += 1
        else:
            num_incorrect += 1
        
        # Calculate reasoning quality
        reasoning_quality = 0.0
        if reasoning:
            # Check for step-by-step reasoning
            steps = len(re.findall(r'(?:Step \d+:|First,|Second,|Third,|Next,|Then,|Finally,)', reasoning, re.IGNORECASE))
            
            # Check for references to the context
            has_context_references = bool(re.search(r'(?:according to|in the context|the context says|based on)', reasoning, re.IGNORECASE))
            
            # Check for logical reasoning
            has_logical_reasoning = bool(re.search(r'(?:therefore|thus|because|since|as a result)', reasoning, re.IGNORECASE))
            
            # Simple heuristic for reasoning quality (0.0 to 1.0)
            reasoning_quality = min(1.0, (steps + has_context_references + has_logical_reasoning) / 5.0)
        
        reasoning_quality_scores.append(reasoning_quality)
    
    # Calculate overall metrics
    total_traces = len(traces)
    accuracy = num_correct / total_traces if total_traces > 0 else 0
    avg_reasoning_quality = np.mean(reasoning_quality_scores) if reasoning_quality_scores else 0
    
    # Prepare results
    results = {
        "num_traces": total_traces,
        "num_correct": num_correct,
        "num_incorrect": num_incorrect,
        "accuracy": accuracy,
        "avg_reasoning_quality": avg_reasoning_quality
    }
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Total Traces: {total_traces}")
    print(f"Correct Answers: {num_correct} ({accuracy * 100:.2f}%)")
    print(f"Incorrect Answers: {num_incorrect} ({(1 - accuracy) * 100:.2f}%)")
    print(f"Average Reasoning Quality: {avg_reasoning_quality:.4f}")
    
    # Save results
    output_file = os.path.join(args.output_dir, "reasoning_evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 
 
 
 