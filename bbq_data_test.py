from datasets import load_from_disk

dataset_path = "/leonardo_work/EUHPC_D19_099/vraminen/datasets/bbq_dataset"
dataset = load_from_disk(dataset_path)

print(dataset)  # shows info about splits, features

test_data = dataset["test"]
print(test_data)  # summary of the test split
test_data.to_json("bbq_test.json")
