#!/bin/bash
# Test mode script for BBQ generation with GPT-OSS-20B
# Runs only 10 samples per category for quick testing

# Configuration
MODEL="openai/gpt-oss-20b"
OUTPUT_BASE="outputs/gpt-oss-20B_test"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/run.log"
TENSOR_PARALLEL=2
BATCH_SIZE=8 # Smaller batch for test mode
SEED=42
REASONING_LEVEL="medium" # options: low, medium, high
TEMPERATURE=0.7
TOP_P=0.95
TOP_K=20
MAX_LENGTH=2048

# Categories to test
CATEGORIES="Age Nationality Religion"

# Create output directory
mkdir -p $OUTPUT_DIR

# Function to print colored output
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1" | tee -a $LOG_FILE
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1" | tee -a $LOG_FILE
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1" | tee -a $LOG_FILE
}

# Start logging
print_info "Starting BBQ TEST MODE generation at $(date)"
print_info "Configuration:"
echo "  Model: $MODEL" | tee -a $LOG_FILE
echo "  Output: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "  Log: $LOG_FILE" | tee -a $LOG_FILE
echo "  Categories: $CATEGORIES" | tee -a $LOG_FILE
echo "  Tensor Parallel: $TENSOR_PARALLEL" | tee -a $LOG_FILE
echo "  Batch Size: $BATCH_SIZE" | tee -a $LOG_FILE
echo "  Seed: $SEED" | tee -a $LOG_FILE
echo "  Reasoning Level: $REASONING_LEVEL" | tee -a $LOG_FILE
echo "  Temperature: $TEMPERATURE" | tee -a $LOG_FILE
echo "  Top-p: $TOP_P" | tee -a $LOG_FILE
echo "  Top-k: $TOP_K" | tee -a $LOG_FILE
echo "  Max Tokens: $MAX_LENGTH" | tee -a $LOG_FILE
echo "  Mode: TEST (10 samples per category)" | tee -a $LOG_FILE
echo "================================" | tee -a $LOG_FILE

# Run the generation
print_info "Starting generation..."

poetry run python scripts/generate_bbq_outputs_vllm_gpt-oss_simple.py \
    --model $MODEL \
    --output_dir $OUTPUT_DIR \
    --tensor_parallel_size $TENSOR_PARALLEL \
    --batch_size $BATCH_SIZE \
    --categories $CATEGORIES \
    --seed $SEED \
    --test_mode \
    --reasoning_level $REASONING_LEVEL \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --max_length $MAX_LENGTH \
    2>&1 | tee -a $LOG_FILE

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_success "Generation completed successfully!"
    
    # Show summary of outputs
    echo "" | tee -a $LOG_FILE
    print_info "Output files:"
    ls -lh $OUTPUT_DIR/*.json 2>/dev/null | tee -a $LOG_FILE
    
    # Extract accuracy from evaluation stats if available
    if [ -f "$OUTPUT_DIR/evaluation_stats.json" ]; then
        echo "" | tee -a $LOG_FILE
        print_info "Quick accuracy summary:"
        python - <<PYTHON | tee -a $LOG_FILE
import json
with open('$OUTPUT_DIR/evaluation_stats.json', 'r') as f:
    stats = json.load(f)
print(f"  Overall: {stats['overall']['accuracy']:.2f}%")
for cat, data in stats['categories'].items():
    print(f"  {cat}: {data['accuracy']:.2f}%")
PYTHON
    fi
else
    print_error "Generation failed! Check the log for details."
    exit 1
fi

echo "================================" | tee -a $LOG_FILE
print_success "Completed at $(date)"
print_info "Full log saved to: $LOG_FILE"
print_info "Results saved to: $OUTPUT_DIR"

# Optional: How to monitor in screen
echo "" | tee -a $LOG_FILE
echo "Tip: To run in background with screen:" | tee -a $LOG_FILE
echo "  screen -S bbq_test bash $0" | tee -a $LOG_FILE
echo "  (Detach with Ctrl+A, D)" | tee -a $LOG_FILE
echo "  screen -r bbq_test  # to reattach" | tee -a $LOG_FILE
