import os
from huggingface_hub import snapshot_download

def is_model_cached(local_dir):
    # Check if directory exists and contains files
    return os.path.isdir(local_dir) and any(os.scandir(local_dir))

models = {
    "deepseek-70B": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "qwen3-32B": "Qwen/Qwen3-32B",
    "qwen3-14B": "Qwen/Qwen3-14B"}

base_cache_dir = "/leonardo_work/EUHPC_D19_099/vraminen/models"

for name, repo_id in models.items():
    local_path = f"{base_cache_dir}/{name}"
    if not is_model_cached(local_path):
        print(f"Caching model {name} from {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=2
        )
        print(f"Cached {name} at {local_path}")
    else:
        print(f"Model {name} already cached at {local_path}, skipping download.")