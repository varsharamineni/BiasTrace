import os
import time
from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

def is_model_fully_cached(local_dir, expected_files):
    if not os.path.isdir(local_dir):
        return False
    # Check if all expected shard files exist
    for f in expected_files:
        if not os.path.isfile(os.path.join(local_dir, f)):
            return False
    return True

def download_with_retries(repo_id, local_dir, expected_files, max_retries=5, wait_sec=60):
    attempt = 0
    while attempt < max_retries:
        if is_model_fully_cached(local_dir, expected_files):
            print(f"Model already fully cached at {local_dir}")
            return
        try:
            print(f"Attempt {attempt+1} to download {repo_id} into {local_dir}")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                allow_patterns=["*.safetensors", "config.json", "tokenizer*", "README.md", "generation_config.json"],
                ignore_patterns=["*checkpoint*"]  # optional, skip checkpoint files if any
            )
            if is_model_fully_cached(local_dir, expected_files):
                print(f"Successfully cached {repo_id} at {local_dir}")
                return
            else:
                print("Download incomplete, retrying...")
        except RepositoryNotFoundError:
            print(f"Repository {repo_id} not found. Check repo ID.")
            break
        except Exception as e:
            print(f"Error during download: {e}")
        attempt += 1
        print(f"Waiting {wait_sec} seconds before retrying...")
        time.sleep(wait_sec)
    print(f"Failed to download the full model after {max_retries} attempts.")

if __name__ == "__main__":
    repo_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    local_dir = "/leonardo_work/EUHPC_D19_099/vraminen/models/deepseek-70B"
    expected_files = [f"model-{'%05d' % i}-of-000017.safetensors" for i in range(1, 18)]
    # Note: Adjust expected_files naming format if your shards have different names!

    download_with_retries(repo_id, local_dir, expected_files)
