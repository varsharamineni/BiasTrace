import json
from pathlib import Path
import pandas as pd

RESULTS_DIR = "results"
EXCLUDE_CATEGORIES = ["Race_x_SES", "Race_x_gender"]


def parse_file(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    folder_name = Path(json_path).parts[-3]

    # Extract model + prompt type
    if "_simple_prompt" in folder_name:
        model_name = folder_name.replace("_simple_prompt", "")
        prompt_type = "simple"
    elif "_full_prompt" in folder_name:
        model_name = folder_name.replace("_full_prompt", "")
        prompt_type = "full"
    else:
        model_name = folder_name
        prompt_type = "unknown"

    rows = []

    for category, stats in data.items():
        row = {
            "model": model_name,
            "prompt_type": prompt_type,
            "category": category,

            "total": stats.get("total", 0),
            "incorrect": stats.get("incorrect", 0),

            # DISAMB
            "disamb_total": stats.get("n_disamb", 0),
            "disamb_incorrect": stats.get("disamb_n_incorrect", 0),
            "disamb_incorrect_and_stereotype": stats.get("disamb_n_incorrect_and_stereotype", 0),
            "disamb_incorrect_and_non_stereotype": stats.get("disamb_n_incorrect_and_non_stereotype", 0),
            "disamb_incorrect_and_unknown": stats.get("disamb_n_incorrect_and_unknown", 0),

            # AMB
            "amb_total": stats.get("n_amb", 0),
            "amb_incorrect": stats.get("amb_n_incorrect", 0),
            "amb_incorrect_and_stereotype": stats.get("amb_n_incorrect_and_stereotype", 0),
            "amb_incorrect_and_non_stereotype": stats.get("amb_n_incorrect_and_non_stereotype", 0),
            "amb_incorrect_and_unknown": stats.get("amb_n_incorrect_and_unknown", 0),
        }

        rows.append(row)

    return rows


def collect_results(results_dir):
    all_rows = []
    for json_file in Path(results_dir).rglob("*.json"):
        all_rows.extend(parse_file(json_file))
    return pd.DataFrame(all_rows)


def compute_percentages(df):
    # Ambiguous
    df["amb_incorrect_pct"] = df["amb_incorrect"] / df["amb_total"] * 100
    df["amb_incorrect_stereotype_pct"] = df["amb_incorrect_and_stereotype"] / df["amb_total"] * 100
    df["amb_incorrect_non_stereotype_pct"] = df["amb_incorrect_and_non_stereotype"] / df["amb_total"] * 100
    df["amb_incorrect_unknown_pct"] = df["amb_incorrect_and_unknown"] / df["amb_total"] * 100

    # Disambiguated
    df["disamb_incorrect_pct"] = df["disamb_incorrect"] / df["disamb_total"] * 100
    df["disamb_incorrect_stereotype_pct"] = df["disamb_incorrect_and_stereotype"] / df["disamb_total"] * 100
    df["disamb_incorrect_non_stereotype_pct"] = df["disamb_incorrect_and_non_stereotype"] / df["disamb_total"] * 100
    df["disamb_incorrect_unknown_pct"] = df["disamb_incorrect_and_unknown"] / df["disamb_total"] * 100

    # Overall
    df["overall_total"] = df["amb_total"] + df["disamb_total"]

    df["overall_incorrect"] = df["amb_incorrect"] + df["disamb_incorrect"]
    df["overall_incorrect_stereotype"] = df["amb_incorrect_and_stereotype"] + df["disamb_incorrect_and_stereotype"]
    df["overall_incorrect_non_stereotype"] = df["amb_incorrect_and_non_stereotype"] + df["disamb_incorrect_and_non_stereotype"]
    df["overall_incorrect_unknown"] = df["amb_incorrect_and_unknown"] + df["disamb_incorrect_and_unknown"]

    df["overall_incorrect_pct"] = df["overall_incorrect"] / df["overall_total"] * 100
    df["overall_incorrect_stereotype_pct"] = df["overall_incorrect_stereotype"] / df["overall_total"] * 100
    df["overall_incorrect_non_stereotype_pct"] = df["overall_incorrect_non_stereotype"] / df["overall_total"] * 100
    df["overall_incorrect_unknown_pct"] = df["overall_incorrect_unknown"] / df["overall_total"] * 100

    return df


def summarize_by_category(df):
    summary = df.groupby("category").agg(
        total=("total", "sum"),
        incorrect=("incorrect", "sum"),

        disamb_total=("disamb_total", "sum"),
        disamb_incorrect=("disamb_incorrect", "sum"),
        disamb_incorrect_and_stereotype=("disamb_incorrect_and_stereotype", "sum"),
        disamb_incorrect_and_non_stereotype=("disamb_incorrect_and_non_stereotype", "sum"),
        disamb_incorrect_and_unknown=("disamb_incorrect_and_unknown", "sum"),

        amb_total=("amb_total", "sum"),
        amb_incorrect=("amb_incorrect", "sum"),
        amb_incorrect_and_stereotype=("amb_incorrect_and_stereotype", "sum"),
        amb_incorrect_and_non_stereotype=("amb_incorrect_and_non_stereotype", "sum"),
        amb_incorrect_and_unknown=("amb_incorrect_and_unknown", "sum"),
    ).reset_index()

    return compute_percentages(summary)


def summarize_by_model_prompt(df):
    summary = df.groupby(["model", "prompt_type"]).agg(
        total=("total", "sum"),
        incorrect=("incorrect", "sum"),

        disamb_total=("disamb_total", "sum"),
        disamb_incorrect=("disamb_incorrect", "sum"),
        disamb_incorrect_and_stereotype=("disamb_incorrect_and_stereotype", "sum"),
        disamb_incorrect_and_non_stereotype=("disamb_incorrect_and_non_stereotype", "sum"),
        disamb_incorrect_and_unknown=("disamb_incorrect_and_unknown", "sum"),

        amb_total=("amb_total", "sum"),
        amb_incorrect=("amb_incorrect", "sum"),
        amb_incorrect_and_stereotype=("amb_incorrect_and_stereotype", "sum"),
        amb_incorrect_and_non_stereotype=("amb_incorrect_and_non_stereotype", "sum"),
        amb_incorrect_and_unknown=("amb_incorrect_and_unknown", "sum"),
    ).reset_index()

    return compute_percentages(summary)


if __name__ == "__main__":
    df = collect_results(RESULTS_DIR)

    # Remove unwanted categories
    df = df[~df["category"].isin(EXCLUDE_CATEGORIES)]

    # Compute per-model metrics
    df = compute_percentages(df)

    print("Per-model per-category results:")
    print(df.head(20))

    # Category-level summary
    category_summary = summarize_by_category(df)
    print("\nSummary by category:")
    print(category_summary)

    # Model + prompt summary (NEW)
    model_prompt_summary = summarize_by_model_prompt(df)
    print("\nSummary by model and prompt type:")
    print(model_prompt_summary)

    # Save outputs
    df.to_csv("per_model_category_results.csv", index=False)
    category_summary.to_csv("category_summary.csv", index=False)
    model_prompt_summary.to_csv("model_prompt_summary.csv", index=False)