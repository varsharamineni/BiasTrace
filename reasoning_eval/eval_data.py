from datasets import load_dataset
import pandas as pd
import json

class EvalData:

    def __init__(self, category: str = "Age"):
        self.category = category

    def load_BBQ_dataset(self, dataset_path: str = "../datasets/bbq_dataset_all_cat/data"):
        """
        # Load a single JSON file
        # The file is just a single .jsonl with no predefined splits. In that case, load_dataset("json", ...) still wraps it 
        # into a DatasetDict with a default key "train" (even though there is no actual train/test distinction)
        """
        dataset = load_dataset("json", data_files=f"{dataset_path}/{self.category}.jsonl", split="train")
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

