import json
import os
import matplotlib.pyplot as plt

# -----------------------------
# Detect Unknown Answers
# -----------------------------
def is_unknown(r):
    """
    Determine if the model's extracted answer is 'Unknown' based on answer_info.
    """
    extracted = r.get("extracted_answer")
    normalized = r.get("normalized_answer", "").strip().lower()
    if extracted is None:
        return False
    for key in ["ans0", "ans1", "ans2"]:
        if key in r:
            ans_text = r[key].strip().lower()
            if normalized == ans_text:
                labels = r.get("answer_info", {}).get(key, [])
                if "unknown" in labels:
                    return True
    return False

# -----------------------------
# Categorize Mistakes
# -----------------------------
def categorize_mistakes(results):
    counts = {
        'dis': {'unknown': 0, 'not_unknown_stereo': 0, 'not_unknown_nonstereo': 0},
        'amb': {'not_unknown_stereo': 0, 'not_unknown_nonstereo': 0}
    }
    for r in results:
        is_dis = not r.get("ambiguous", False)
        incorrect = not r["is_correct"]
        unknown = is_unknown(r)
        stereotype = r.get("stereotype_alignment", False)
        if incorrect:
            if is_dis:
                if unknown:
                    counts['dis']['unknown'] += 1
                elif stereotype:
                    counts['dis']['not_unknown_stereo'] += 1
                else:
                    counts['dis']['not_unknown_nonstereo'] += 1
            else:
                if not unknown and stereotype:
                    counts['amb']['not_unknown_stereo'] += 1
                elif not unknown and not stereotype:
                    counts['amb']['not_unknown_nonstereo'] += 1
    return counts

# -----------------------------
# Aggregate by Category
# -----------------------------
def compute_mistake_summary(folder_path):
    summary = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('_results_merged.json') and filename.startswith('bbq_'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results = data["results"]
            for r in results:
                cat = r["category"]
                if cat not in summary:
                    summary[cat] = {'dis': {'unknown': 0, 'not_unknown_stereo': 0, 'not_unknown_nonstereo': 0},
                                    'amb': {'not_unknown_stereo': 0, 'not_unknown_nonstereo': 0}}
                cat_counts = categorize_mistakes([r])
                for key in ['dis', 'amb']:
                    for subkey in cat_counts[key]:
                        summary[cat][key][subkey] += cat_counts[key][subkey]
    return summary

# -----------------------------
# Plot Mistakes as Proportions
# -----------------------------
def plot_mistakes_proportion(summary, output_file="mistakes_proportion.png"):
    categories = list(summary.keys())
    x = range(len(categories))
    width = 0.25

    # Colors
    color_map = {
        'unknown': '#7F7F7F',            # gray (neutral)
        'not_unknown_stereo': '#FF0000', # vivid red (worst mistakes)
        'not_unknown_nonstereo': '#2CA02C' # green (safe)
    }

    # Compute total mistakes per category
    dis_total = [
        summary[cat]['dis']['unknown'] +
        summary[cat]['dis']['not_unknown_stereo'] +
        summary[cat]['dis']['not_unknown_nonstereo']
        for cat in categories
    ]
    amb_total = [
        summary[cat]['amb']['not_unknown_stereo'] +
        summary[cat]['amb']['not_unknown_nonstereo']
        for cat in categories
    ]

    # Compute proportions
    dis_unknown = [summary[cat]['dis']['unknown'] / total if total>0 else 0 for cat, total in zip(categories, dis_total)]
    dis_stereo = [summary[cat]['dis']['not_unknown_stereo'] / total if total>0 else 0 for cat, total in zip(categories, dis_total)]
    dis_nonstereo = [summary[cat]['dis']['not_unknown_nonstereo'] / total if total>0 else 0 for cat, total in zip(categories, dis_total)]

    amb_stereo = [summary[cat]['amb']['not_unknown_stereo'] / total if total>0 else 0 for cat, total in zip(categories, amb_total)]
    amb_nonstereo = [summary[cat]['amb']['not_unknown_nonstereo'] / total if total>0 else 0 for cat, total in zip(categories, amb_total)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,6), sharey=True)

    # Disambiguated
    ax1.bar([p - width for p in x], dis_unknown, width=width, color=color_map['unknown'], label='Unknown')
    ax1.bar(x, dis_stereo, width=width, color=color_map['not_unknown_stereo'], label='Not unknown & stereotype')
    ax1.bar([p + width for p in x], dis_nonstereo, width=width, color=color_map['not_unknown_nonstereo'], label='Not unknown & non-stereotype')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.set_title('Disambiguated Mistakes (Proportion)')
    ax1.set_ylabel('Proportion of mistakes')
    ax1.set_ylim(0,1)
    ax1.legend()

    # Ambiguous
    ax2.bar([p - width/2 for p in x], amb_stereo, width=width, color=color_map['not_unknown_stereo'], label='Not unknown & stereotype')
    ax2.bar([p + width/2 for p in x], amb_nonstereo, width=width, color=color_map['not_unknown_nonstereo'], label='Not unknown & non-stereotype')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.set_title('Ambiguous Mistakes (Proportion)')
    ax2.set_ylim(0,1)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Mistake proportion plot saved to {output_file}")
    plt.close()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    #folder_path = "outputs/qwen_8B_full_prompt"  # update to your folder
    folder_path = 'outputs/qwen_full_8B_simple_prompt/20250827_163953'
    summary = compute_mistake_summary(folder_path)
    plot_mistakes_proportion(summary, output_file="mistakes_proportion_8B_simple_prompt.png")

    folder_path = "outputs/qwen_8B_full_prompt" 
    summary = compute_mistake_summary(folder_path)
    plot_mistakes_proportion(summary, output_file="mistakes_proportion_8B_full_prompt.png")
