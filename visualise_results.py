import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_and_bias(model_name, results_path, save_path=None):
    with open(results_path, "r") as f:
        results = json.load(f)

    categories = list(results["accuracy"].keys())
    accuracy = [results["accuracy"][c] for c in categories]
    bias = [results["bias_score"][c] if results["bias_score"][c] != "N/A" else 0.0 for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(x - width/2, accuracy, width, label='Accuracy')
    ax.bar(x + width/2, bias, width, label='Bias Score')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name}: Accuracy vs Bias Score by Category")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()

# Example usage
model_name = "qwen3-32B"
results_file = f"eval_results/{model_name}/evaluation_results.json"
save_file = f"eval_results/{model_name}/accuracy_bias_plot.png"

plot_accuracy_and_bias(model_name, results_file, save_path=save_file)