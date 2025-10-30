import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV = "reasoning_eval/ground_truth_samples/sample_traces_full_annotated_test.csv"
SAVE_FIGS = True

# Load CSV
df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip().str.lower()

# Ensure boolean columns
for col in ["ambiguous", "is_correct", "stereotype_aligned"]:
    df[col] = df.get(col, False).astype(bool)

# Normalize prompt type
if "prompt_type" not in df.columns:
    raise KeyError("CSV must have a 'prompt_type' column.")
df["prompt_type"] = df["prompt_type"].astype(str).str.strip()
df = df[df["prompt_type"] != ""]

# Error categories
error_features = [
    "group_assumption",
    "bias_acknowledgement",
    "meta_reflection",
    "outside_demo_knowledge",
    "outside_topical_knowledge",
    "unresolved",
    "overthinking",
    "missing_logic"
]
for col in error_features:
    if col not in df.columns:
        df[col] = 0
    else:
        df[col] = df[col].fillna(0).astype(int)

# Melt for plotting
melted = df.melt(
    id_vars=["prompt_type", "ambiguous", "is_correct", "stereotype_aligned"],
    value_vars=error_features,
    var_name="error_type",
    value_name="count"
)

# Aggregate counts
agg = melted.groupby(
    ["stereotype_aligned", "is_correct", "ambiguous", "prompt_type", "error_type"]
)["count"].sum().reset_index()

if agg.empty:
    raise ValueError("Aggregated data is empty. Check your columns.")

# --- Plot separate figures for stereotype_aligned True/False ---
for st_aligned in [True, False]:
    data = agg[agg["stereotype_aligned"] == st_aligned]
    title = f"Stereotype Aligned = {st_aligned}"
    g = sns.catplot(
        data=data,
        x="prompt_type",
        y="count",
        hue="error_type",
        row="is_correct",
        col="ambiguous",
        kind="bar",
        palette="Set2",
        height=4,
        aspect=1.2
    )
    g.set_titles("Correct: {row_name} | Ambiguous: {col_name}")
    g.set_axis_labels("Prompt Type", "Count")
    g.set_xticklabels(rotation=25)
    if g._legend:
        g._legend.set_title("Error Type")
    plt.tight_layout()
    filename = f"facet_error_distribution_stereotype_{st_aligned}.png"
    if SAVE_FIGS:
        plt.savefig(filename, dpi=300)
    plt.show()
    print(f"📊 Saved: {filename}")
