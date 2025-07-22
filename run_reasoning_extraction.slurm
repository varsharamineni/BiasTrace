#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --job-name=reasoning_extract
#SBATCH --output=$WORK/reasoning-towards-fairness/logs/reasoning_extract_%j.out
#SBATCH --error=$WORK/reasoning-towards-fairness/logs/reasoning_extract_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=30G
#SBATCH --gres=tmpfs:10G
#SBATCH --cpus-per-task=4

module purge
module load python/3.10

echo "🚀 SLURM job started on $(hostname)"
source ~/envs/reasoning_pattern_env/bin/activate


# Define input file (assuming it is still in $HOME)
INPUT_FILE="$HOME/Reasoing-Towards-Fairness/outputs/processed_bbq_checkpoint_results/only_correct/checkpoint_3177/nationality/checkpoint_3177_final_copy.json"

#INPUT_FILE="$HOME/Reasoing-Towards-Fairness/outputs/processed_bbq_checkpoint_results/incorrect/checkpoint_356/nationality/checkpoint_356_final_copy.json"


# Define output file inside project folder in $WORK
OUTPUT_FILE="$WORK/reasoning-towards-fairness/results/reasoning_clusters.json"

#OUTPUT_FILE="$HOME/Reasoing-Towards-Fairness/results_incorrect.json"

echo "Running job $SLURM_JOB_ID"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"

python scripts/extract_reasoning_patterns.py \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE"


echo "✅ Job finished. Output saved in $OUTPUT_FILE"
