import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# File paths
# ---------------------------
baseline_file = "reasoning_eval/llm_judge_samples/test_set/baseline/llm_eval_claude-opus-4-1-20250805_llama70B_gt_temp0.6_seed42_max_tokens2048.json"
labels_file   = "reasoning_eval/llm_judge_samples/test_set/our_labels/llm_eval_claude-opus-4-1-20250805_detailed_example_clarification_opt_temp0.6_seed42_max_tokens2048_reasoning.json"

# Output file
output_file = "error_label_summary.png"

# ---------------------------
# Load JSONs
# ---------------------------
with open(baseline_file) as f:
    baseline_data = json.load(f)

with open(labels_file) as f:
    label_data = json.load(f)

# ---------------------------
# Merge traces by sample_id
# ---------------------------
baseline_scores = {r['sample_id']: r['judge_output']['bias_label'] for r in baseline_data['results']}
label_scores = {r['sample_id']: r['judge_output'] for r in label_data['results']}

# ---------------------------
# Metrics calculation
# ---------------------------
baseline_score_counts = Counter(baseline_scores.values())

coverage_per_score = {}
profiles_per_score = {}
blind_spots_count = 0
total_traces = len(label_scores)

for score in sorted(set(baseline_scores.values())):
    traces = [sid for sid, s in baseline_scores.items() if s == score]
    # Coverage: proportion with ≥1 error label
    coverage = sum(1 for sid in traces if any(label_scores[sid].values())) / len(traces)
    coverage_per_score[score] = coverage
    # Distinct error profiles
    profiles = set()
    for sid in traces:
        labels_tuple = tuple(sorted((k,v) for k,v in label_scores[sid].items()))
        profiles.add(labels_tuple)
    profiles_per_score[score] = len(profiles)
    # Blind spots
    if score == 0:
        blind_spots_count = sum(1 for sid in traces if any(label_scores[sid].values()))

# ---------------------------
# Visualization
# ---------------------------
fig, ax1 = plt.subplots(figsize=(10,6))

scores = sorted(coverage_per_score.keys())
coverage_values = [coverage_per_score[s]*100 for s in scores]  # %
profile_values = [profiles_per_score[s] for s in scores]

# Coverage bar plot
ax1.bar(scores, coverage_values, color='skyblue', alpha=0.7, label='Coverage (% traces with ≥1 error)')
ax1.set_xlabel("Baseline Score")
ax1.set_ylabel("Coverage (%)", color='blue')
ax1.set_ylim(0, 110)
ax1.tick_params(axis='y', labelcolor='blue')

# Distinct profiles line plot
ax2 = ax1.twinx()
ax2.plot(scores, profile_values, color='red', marker='o', label='Distinct error profiles')
ax2.set_ylabel("Distinct Error Profiles", color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Annotate blind spots
ax1.text(0, coverage_values[0]+5, f"Blind spots: {blind_spots_count}", color='purple', fontweight='bold')

# Title and legend
plt.title("Summary of Error Label Analysis vs Baseline Score")
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()

# ---------------------------
# Save figure
# ---------------------------
plt.savefig('error_label_summary_0_1.png', dpi=300)
print(f"Figure saved as {output_file}")
plt.close()
1



# ---------------------------
# Prepare data for stacked bar
# ---------------------------
scores = sorted(set(baseline_scores.values()))
error_labels = list(label_scores[next(iter(label_scores))].keys())

# Initialize counts: {baseline_score: {label: count}}
label_counts_by_score = {score: Counter() for score in scores}

for sid, score in baseline_scores.items():
    label_counts_by_score[score].update(label_scores[sid])

# Convert counts to list for plotting
counts_matrix = []
for label in error_labels:
    counts_matrix.append([label_counts_by_score[score][label] for score in scores])


# ---------------------------
# Plot stacked bar
# ---------------------------
fig, ax = plt.subplots(figsize=(10,6))
bottom = [0]*len(scores)

colors = plt.get_cmap('tab20').colors  # nice palette

for i, label in enumerate(error_labels):
    ax.bar(scores, counts_matrix[i], bottom=bottom, color=colors[i % len(colors)], label=label)
    bottom = [bottom[j] + counts_matrix[i][j] for j in range(len(scores))]

ax.set_xlabel("Baseline Score")
ax.set_ylabel("Number of Traces with Error Label")
ax.set_title("Error Label Distribution per Baseline Score (Stacked)")
ax.legend(loc='upper left', bbox_to_anchor=(1,1))  # legend outside plot
plt.tight_layout()

# Save figure
plt.savefig('stacked_bar_0_1.png', dpi=300)
plt.close()
print(f"Stacked bar figure saved as {output_file}")
