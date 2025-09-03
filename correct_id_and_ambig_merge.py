import json
import csv
import os

def process_and_merge_data_by_order(results_file, original_data_file, output_file, meta_data_file=None):
    try:
        # Load results file
        with open(results_file, 'r') as f:
            bbq_data = json.load(f)
            results_list = bbq_data['results']

        # Load original dataset
        original_data_list = []
        with open(original_data_file, 'r') as f:
            for line in f:
                original_data_list.append(json.loads(line))

        # Sanity check
        if len(results_list) != len(original_data_list):
            print("Error: Number of entries in results and original dataset do not match. Cannot merge by order.")
            return

        # Step 1: Merge results with original dataset by order
        for i in range(len(results_list)):
            result = results_list[i]
            original_item = original_data_list[i]

            # Add example_id from original dataset
            example_id = original_item.get('example_id')
            result['example_id'] = example_id

            # Update ambiguous flag based on context_condition
            context_condition = original_item.get('context_condition')
            result['ambiguous'] = (context_condition == 'ambig')

            # Add other fields from original dataset
            result['question_polarity'] = original_item.get('question_polarity')
            result['answer_info'] = original_item.get('answer_info')

            additional_metadata = original_item.get('additional_metadata', {})
            result['version'] = additional_metadata.get('version')
            result['subcategory'] = additional_metadata.get('subcategory')
            result['answer_info'] = original_item.get('answer_info')

        # Step 2: Merge meta data by example_id (if provided)
        if meta_data_file:
            # Detect category from results_file name
            category = None
            base_name = os.path.basename(results_file)
            if "bbq_" in base_name and "_results" in base_name:
                category = base_name.split("bbq_")[1].split("_results")[0]
            print(f"Detected category: {category}")

            # Load meta data CSV filtered to this category
            meta_data_dict = {}
            with open(meta_data_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # filter by category
                    if row.get('category') == category and 'example_id' in row:
                        # normalize example_id to int for matching
                        meta_data_dict[int(row['example_id'])] = row

            # Merge metadata into results
            for result in results_list:
                example_id = result.get('example_id')
                if example_id in meta_data_dict:
                    for k, v in meta_data_dict[example_id].items():
                        if k not in result:  # only fill missing keys
                            result[k] = v

        # Write output
        with open(output_file, 'w') as f:
            json.dump(bbq_data, f, indent=2)

        print(f"Successfully updated data and saved to {output_file}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure the files exist in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def merge_all_categories(base_folder, meta_data_file):
    """
    Merge BBQ results with original data and optional metadata for all categories
    in the given base folder.
    
    base_folder: path to folder containing bbq_*_results.json files
    meta_data_file: path to the additional metadata CSV
    """
    # Detect all categories by looking for *_results.json files
    for filename in os.listdir(base_folder):
        if filename.endswith('_results.json') and filename.startswith('bbq_'):
            category = filename.split('bbq_')[1].split('_results')[0]

            results_file = os.path.join(base_folder, f'bbq_{category}_results.json')
            original_data_file = os.path.join('datasets/bbq_dataset_all_cat/data', f'{category}.jsonl')
            output_file = os.path.join(base_folder, f'bbq_{category}_results_merged.json')

            print(f"Processing category: {category}")
            process_and_merge_data_by_order(
                results_file,
                original_data_file,
                output_file,
                meta_data_file=meta_data_file
            )


if __name__ == "__main__":

    meta_data_file = 'datasets/bbq_additional_metadata.csv'

    base_folder = 'outputs/qwen_full_14B_simple_prompt/20250828_215719/'    
    merge_all_categories(base_folder, meta_data_file)

    base_folder = 'outputs/qwen_full_8B_simple_prompt/20250827_163953'
    merge_all_categories(base_folder, meta_data_file)

    base_folder = 'outputs/qwen_8B_full'
    merge_all_categories(base_folder, meta_data_file)


