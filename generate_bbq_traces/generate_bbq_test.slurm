#!/bin/bash
#SBATCH --job-name=bbq-gen-deepseek8b
#SBATCH --output=bbq-gen-%j.out
#SBATCH --error=bbq-gen-%j.err
#SBATCH --partition=boost_usr_prod  # GPU partition with 8 GPUs total
#SBATCH --nodes=1                     # Use 1 node (adjust if you want more nodes)
#SBATCH --gres=gpu:4                  # Request 4 GPUs on this node (tensor parallelism)
#SBATCH --cpus-per-task=8             # CPUs to support data loading, batching, etc.
#SBATCH --mem=64G                    # More RAM if available to cache dataset and support GPU
#SBATCH --time=04:00:00               # 4 hour max runtime (adjust to your job)

source ~/envs/reasoning_pattern_env/bin/activate

export HF_HOME=/leonardo_work/EUHPC_D19_099/vraminen/hf_home
export VLLM_CACHE_DIR=$WORK/vllm_cache

python scripts/generate_bbq_outputs_vllm.py   \
  --model_path /leonardo_work/EUHPC_D19_099/vraminen/models/deepseek-8B \
  --num_samples 32 \
  --output_dir /leonardo_work/EUHPC_D19_099/vraminen/outputs/bbq_results_test \
  --batch_size 16
