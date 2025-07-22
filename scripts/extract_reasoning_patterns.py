import argparse
import json
import os
from pathlib import Path
from collections import Counter
from sentence_transformers import SentenceTransformer
import re
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk


# Download NLTK data if not already done
nltk.download('punkt_tab')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('averaged_perceptron_tagger_eng')

def load_traces(json_path):
    with open(json_path, "r") as f:
        return json.load(f)
    
    
ENTITY_MAP = {
    'PERSON': '[PERSON]',
    'GPE': '[LOCATION]',
    'LOC': '[LOCATION]',
    'ORG': '[ORG]',
    'ORGANIZATION': '[ORG]',
    'NORP': '[GROUP]',
    'FACILITY': '[FACILITY]',
    'EVENT': '[EVENT]',
    'PRODUCT': '[PRODUCT]',
    'WORK_OF_ART': '[WORK]',
    'LAW': '[LAW]',
    'LANGUAGE': '[LANGUAGE]',
    'DATE': '[DATE]',
    'TIME': '[TIME]',
    'MONEY': '[MONEY]',
    'PERCENT': '[PERCENT]',
    'QUANTITY': '[QUANTITY]',
    'ORDINAL': '[ORDINAL]',
    'CARDINAL': '[CARDINAL]'
}

def generalize_entities(sentence):
    """
    Replace named entities in the sentence with general placeholders using NLTK NE chunking.
    """
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)
    chunks = ne_chunk(pos_tags, binary=False)

    generalized_tokens = []
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            placeholder = ENTITY_MAP.get(chunk.label(), None)
            if placeholder:
                # Replace entire entity chunk with placeholder
                generalized_tokens.append(placeholder)
            else:
                # Keep original tokens if entity type not in ENTITY_MAP
                generalized_tokens.extend([token for token, pos in chunk.leaves()])
        else:
            generalized_tokens.append(chunk[0])

    return ' '.join(generalized_tokens)

def extract_think_sentences(data):
    """
    Extracts the content between <think> and </think> tags from generated_text field.
    Generalizes named entities.
    """
    reasoning_sentences = []
    for ex in data:
        text = ex["generated_text"]
        # Extract up to the word 'think' (case insensitive)
        match = re.search(r'\bthink\b', text, re.IGNORECASE)
        if match:
            extracted = text[:match.start()].strip()
        else:
            # If no 'think' found, use whole text
            extracted = text.strip()
        # Split extracted text into sentences, keeping only sentences longer than 5 chars
        sentences = [s.strip() for s in re.split(r'[.?!]\s+', extracted) if len(s.strip()) > 5]

        # Generalize entities in each sentence
        generalized = [generalize_entities(sentence) for sentence in sentences]
        reasoning_sentences.extend(generalized)
    return reasoning_sentences

def cluster_sentences(sentences, n_clusters=5, model_name="all-MiniLM-L6-v2"):
    """
    Groups sentences into a fixed number of clusters using embeddings and Agglomerative Clustering.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, convert_to_tensor=False, normalize_embeddings=True)
    embeddings_np = np.array(embeddings)
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    cluster_labels = clustering_model.fit_predict(embeddings_np)
    clusters = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(sentences[idx])
    return clusters

def save_clusters(clusters, out_path="output/reasoning_clusters.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Save as list of dicts with count and example sentences
    cluster_dict = [{"count": len(c), "examples": c[:5]} for c in sorted(clusters, key=len, reverse=True)]
    with open(out_path, "w") as f:
        json.dump(cluster_dict, f, indent=2)
    print(f"\n💾 Clustered reasoning sentences saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the reasoning trace JSON file")
    parser.add_argument("--output", type=str, default="output/reasoning_clusters.json", help="Path to save clustered output")
    parser.add_argument("--num_clusters", type=int, default=5, help="Number of clusters to form")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return

    print(f"📥 Loading reasoning traces from {args.input}")
    data = load_traces(args.input)

    print("🔍 Extracting and generalizing reasoning sentences...")
    all_sentences = extract_think_sentences(data)
    print(f"🧠 Collected {len(all_sentences)} reasoning sentences.")

    counter = Counter(all_sentences)
    unique_sentences = list(counter.keys())

    print(f"🔗 Clustering sentences into {args.num_clusters} clusters...")
    clustered = cluster_sentences(unique_sentences, n_clusters=args.num_clusters)

    print("\n🧩 Top clusters:")
    for i, cluster in enumerate(sorted(clustered, key=len, reverse=True)):
        print(f"\nCluster #{i+1} ({len(cluster)} items):")
        for line in cluster[:5]:
            print(f"  - {line}")
        if len(cluster) > 5:
            print(f"  ... and {len(cluster) - 5} more")

    save_clusters(clustered, args.output)

if __name__ == "__main__":
    main()
