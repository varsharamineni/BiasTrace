from datasets import load_dataset
import pandas as pd
import json

class EvalData:

    def __init__(self, category: str = "Age"):
        self.category = category

    def load_BBQ_dataset(self, dataset_path: str = "heegyu/bbq"):
        # Example: load the Age.json file inside the "data" folder
        dataset = load_dataset("json", data_files=f"data/{self.category}.jsonl", repo_id="heegyu/bbq")
        dataset = dataset['test']
        print(f"INFO:: Loaded BBQ dataset with {len(dataset)} samples.")
        return dataset
    
    def load_BBQ_metadata(self, metadata_path: str = "../datasets/bbq_additional_metadata.csv"):
        # load metadata
        metadata = pd.read_csv(metadata_path)
        metadata = metadata[metadata['category'] == self.category]
        print(f"INFO:: Loaded BBQ metadata with {len(metadata)} samples.")
        return metadata
    
    def load_BBQ_templates(self):
        template_path = f"../datasets/bbq_templates/new_templates - {self.category}.csv"
        template_df = pd.read_csv(template_path)
        return template_df
    
    def load_reasoning_data(reasoning_data_path: str = "../outputs/qwen_8B_full/bbq_Age_results_merged.json"):
        with open(reasoning_data_path, "r") as f:
            data = json.load(f)
        reasoning_data = data['results']
        print(f"INFO:: Loaded BBQ reasoning data with {len(reasoning_data)} samples.")
        return reasoning_data

