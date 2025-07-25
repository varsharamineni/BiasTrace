from datasets import load_dataset

dataset = load_dataset("heegyu/bbq")
dataset.save_to_disk("/leonardo_work/EUHPC_D19_099/vraminen/datasets/bbq_dataset")
print("Dataset saved!")