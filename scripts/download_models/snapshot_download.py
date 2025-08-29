import os
from huggingface_hub import snapshot_download, list_repo_files

def is_model_complete(local_dir, repo_id):
    """
    Check if all files in the Hugging Face repo are present in local_dir.
    """
    if not os.path.isdir(local_dir):
        return False

    try:
        # List all files in the HF repo
        repo_files = list_repo_files(repo_id)
    except Exception as e:
        print(f"Error listing files for {repo_id}: {e}")
        return False

    # Check that each repo file exists locally
    missing_files = []
    for f in repo_files:
        local_path = os.path.join(local_dir, f)
        if not os.path.isfile(local_path):
            missing_files.append(f)

    if missing_files:
        print(f"Missing files for {repo_id}: {missing_files}")
        return False

    return True


# Example models to check
models = {
    "qwen3-32B": "Qwen/Qwen3-32B"
}

base_cache_dir = "/home/vramineni/models"

for name, repo_id in models.items():
    local_path = os.path.join(base_cache_dir, name)
    if not is_model_complete(local_path, repo_id):
        print(f"Caching or completing model {name} from {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=1
        )
        print(f"Cached {name} at {local_path}")
    else:
        print(f"Model {name} is complete at {local_path}, no download needed.")
