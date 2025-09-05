import json
import os
import matplotlib.pyplot as plt
import argparse

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
                    summary[cat] = {
                        'dis': {'unknown': 0, 'not_unknown_stereo': 0, 'not_unknown_nonstereo': 0, 'total': 0},
                        'amb': {'not_unknown_stereo': 0, 'not_unknown_nonstereo': 0, 'total': 0}
                    }
                # Increment totals for all answers
                is_dis = not r.get("ambiguous", False)
                if is_dis:
                    summary[cat]['dis']['total'] += 1
                else:
                    summary[cat]['amb']['total'] += 1
                
                # Count mistakes as before
                cat_counts = categorize_mistakes([r])
                for key in ['dis', 'amb']:
                    for subkey in cat_counts[key]:
                        summary[cat][key][subkey] += cat_counts[key][subkey]
    return summary

# -----------------------------
# Plot Mistakes as Proportions of All Answers
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

    # Totals: all answers
    dis_total = [summary[cat]['dis']['total'] for cat in categories]
    amb_total = [summary[cat]['amb']['total'] for cat in categories]

    # Compute proportions relative to all answers
    dis_unknown = [summary[cat]['dis']['unknown'] / total if total>0 else 0 for cat, total in zip(categories, dis_total)]
    dis_stereo = [summary[cat]['dis']['not_unknown_stereo'] / total if total>0 else 0 for cat, total in zip(categories, dis_total)]
    dis_nonstereo = [summary[cat]['dis']['not_unknown_nonstereo'] / total if total>0 else 0 for cat, total in zip(categories, dis_total)]

    amb_stereo = [summary[cat]['amb']['not_unknown_stereo'] / total if total>0 else 0 for cat, total in zip(categories, amb_total)]
    amb_nonstereo = [summary[cat]['amb']['not_unknown_nonstereo'] / total if total>0 else 0 for cat, total in zip(categories, amb_total)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,6), sharey=True)

    # Disambiguated
    ax1.bar([p - width for p in x], dis_unknown, width=width, color=color_map['unknown'], label='Incorrect – Unknown')
    ax1.bar(x, dis_stereo, width=width, color=color_map['not_unknown_stereo'], label='Incorrect – Stereotype')
    ax1.bar([p + width for p in x], dis_nonstereo, width=width, color=color_map['not_unknown_nonstereo'], label='Incorrect – Non-stereotype')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.set_title('Disambiguated Mistakes (Proportion of All Answers)')
    ax1.set_ylabel('Proportion of All Answers')
    ax1.set_ylim(0, 0.6)  # <-- cap y-axis at 0.4
    ax1.legend()

    # Ambiguous
    ax2.bar([p - width/2 for p in x], amb_stereo, width=width, color=color_map['not_unknown_stereo'], label='Incorrect – Stereotype')
    ax2.bar([p + width/2 for p in x], amb_nonstereo, width=width, color=color_map['not_unknown_nonstereo'], label='Incorrect – Non-stereotype')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.set_title('Ambiguous Mistakes (Proportion of All Answers)')
    ax2.set_ylim(0,0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Mistake proportion plot saved to {output_file}")
    plt.close()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot BBQ mistake proportions for given folders.")
    parser.add_argument(
        '--folders',
        type=str,
        nargs='+',
        required=True,
        help='Folder(s) containing BBQ results to summarize and plot'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file name (any supported extension). If not specified, generated from folder name.'
    )

    args = parser.parse_args()

    for folder_path in args.folders:
        print(f"Processing folder: {folder_path}")
        summary = compute_mistake_summary(folder_path)

        if args.output:
            output_file = args.output
            # If multiple folders, append folder basename to avoid overwriting
            if len(args.folders) > 1:
                base_name = os.path.basename(os.path.normpath(folder_path))
                name, ext = os.path.splitext(output_file)
                output_file = f"{name}_{base_name}{ext}"
        else:
            output_file = f"mistakes_proportion_{os.path.basename(os.path.normpath(folder_path))}.png"

        plot_mistakes_proportion(summary, output_file=output_file)