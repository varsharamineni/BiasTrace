from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="heegyu/bbq",           # the full dataset repo
    repo_type="dataset",            # specify it's a dataset
    local_dir="/home/vramineni/bias-reasoning-LLM/datasets/bbq_dataset_all_cat",
    local_dir_use_symlinks=False
)
