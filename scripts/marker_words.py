import json
import re
from collections import Counter
from typing import List, Tuple

def extract_candidate_markers_from_json(
    json_path: str,
    top_k: int = 50,
    max_words: int = 3
) -> List[Tuple[str, int]]:
    """
    Extract candidate subthought transition markers from the starts of sentences
    in generated reasoning traces in a JSON file.

    Args:
        json_path (str): Path to the JSON file.
        top_k (int): Number of top markers to return.
        max_words (int): Max number of words to consider in a candidate marker.
    
    Returns:
        List[Tuple[str, int]]: Top-k most common starting phrases with counts.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    starters = []

    for example in data:
        generated = example.get("model_output", "")
        
        # Split into sentences using punctuation followed by whitespace
        sentences = re.split(r'(?<=[.?!])\s+', generated)
        
        for sentence in sentences:
            # Tokenize and get first n words (e.g., "The question is")
            tokens = sentence.strip().split()
            if tokens:
                phrase = ' '.join(tokens[:max_words])
                starters.append(phrase)
    
    return Counter(starters).most_common(top_k)




markers = extract_candidate_markers_from_json(
    "outputs/deepseek_bbq_test/bbq_Nationality_results.json",
    top_k=100,
    max_words=3
)

# Print the results
for phrase, count in markers:
    print(f"{phrase}: {count}")