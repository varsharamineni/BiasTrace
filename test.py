from datasets import load_dataset

for config in ["age", "nationality", "religion"]:
    dataset = load_dataset("bbq", config)
    print(f"Category: {config}, test set size: {len(dataset['test'])}")