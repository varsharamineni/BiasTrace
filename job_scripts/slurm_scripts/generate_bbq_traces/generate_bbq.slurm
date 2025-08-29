#!/bin/bash
#SBATCH --job-name=bbq-gen
#SBATCH --output=bbq-gen-%j.out
#SBATCH --error=bbq-gen-%j.err
#SBATCH --partition=boost_usr_prod  # GPU partition with 8 GPUs total
#SBATCH --nodes=1                     # Use 1 node (adjust if you want more nodes)
#SBATCH --gres=gpu:4                  # Request 4 GPUs on this node (tensor parallelism)
#SBATCH --cpus-per-task=8             # CPUs to support data loading, batching, etc.
#SBATCH --mem=256G                   # More RAM if available to cache dataset and support GPU
#SBATCH --time=24:00:00               # 4 hour max runtime (adjust to your job)


  # Activate your virtual environment
source ~/envs/reasoning_pattern_env/bin/activate

# Set Hugging Face and vLLM cache directories
export HF_HOME=$WORK/hf_home
export VLLM_CACHE_DIR=$WORK/vllm_cache

# Use positional arguments
model_name="$1"
batch_size="$2"
temperature="$3"
top_p="$4"
top_k="$5"
max_length="$6"

# Run the script with passed arguments
python scripts/generate_bbq_outputs_vllm.py \
  --model_path "$WORK/vraminen/models/${model_name}" \
  --output_dir "$WORK/vraminen/outputs/bbq_results_${model_name}" \
  --batch_size "${batch_size}" \
  --categories Disability_status Religion Nationality Gender \
  --temperature "${temperature}" \
  --top_p "${top_p}" \
  --top_k "${top_k}" \
  --max_length "${max_length}"
