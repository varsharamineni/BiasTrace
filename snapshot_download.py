from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    local_dir="/leonardo_work/EUHPC_D19_099/vraminen/models/deepseek-8B",
    local_dir_use_symlinks=False
)


models = {

    "qwen3-4b": "Qwen/Qwen3-4B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "phi-4": "microsoft/Phi-4"
}

base_cache_dir = "/leonardo_work/EUHPC_D19_099/vraminen/models"

for name, repo_id in models.items():
    print(f"Caching model {name} from {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=f"{base_cache_dir}/{name}",
        local_dir_use_symlinks=False
    )
    print(f"Cached {name} at {base_cache_dir}/{name}")