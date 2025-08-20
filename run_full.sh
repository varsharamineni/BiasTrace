#!/bin/bash
# Full dataset script for BBQ generation with Qwen thinking mode
# This runs the complete dataset - may take several hours

# Configuration
MODEL="Qwen/Qwen3-14B"
OUTPUT_BASE="outputs/qwen_full_14B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/run.log"
TENSOR_PARALLEL=2
BATCH_SIZE=16  # Larger batch for full run
SEED=42

# All BBQ categories (modify as needed)
# You can use specific categories or "all" for everything
CATEGORIES="Age Disability_status Gender_identity Nationality Physical_appearance Race_ethnicity Religion SES Sexual_orientation"

# Advanced options
GPU_MEMORY_UTIL=0.9
MAX_LENGTH=2048
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20

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

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1" | tee -a $LOG_FILE
}

# Start logging
print_info "Starting BBQ FULL DATASET generation at $(date)"
print_warning "This is a FULL dataset run and may take several hours!"
echo "" | tee -a $LOG_FILE
print_info "Configuration:"
echo "  Model: $MODEL" | tee -a $LOG_FILE
echo "  Output: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "  Log: $LOG_FILE" | tee -a $LOG_FILE
echo "  Categories: $CATEGORIES" | tee -a $LOG_FILE
echo "  Tensor Parallel: $TENSOR_PARALLEL GPUs" | tee -a $LOG_FILE
echo "  Batch Size: $BATCH_SIZE" | tee -a $LOG_FILE
echo "  GPU Memory: ${GPU_MEMORY_UTIL}" | tee -a $LOG_FILE
echo "  Max Length: $MAX_LENGTH tokens" | tee -a $LOG_FILE
echo "  Temperature: $TEMPERATURE" | tee -a $LOG_FILE
echo "  Seed: $SEED" | tee -a $LOG_FILE
echo "  Mode: FULL DATASET" | tee -a $LOG_FILE
echo "================================" | tee -a $LOG_FILE

# Save start time for duration calculation
START_TIME=$(date +%s)

# Run the generation with all parameters
print_info "Starting generation..."

poetry run python scripts/generate_bbq_outputs_vllm_qwen.py \
    --model $MODEL \
    --output_dir $OUTPUT_DIR \
    --tensor_parallel_size $TENSOR_PARALLEL \
    --batch_size $BATCH_SIZE \
    --categories $CATEGORIES \
    --gpu_memory_utilization $GPU_MEMORY_UTIL \
    --max_length $MAX_LENGTH \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --seed $SEED \
    --enable_thinking \
    --quiet \
    2>&1 | tee -a $LOG_FILE

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_success "Generation completed successfully!"
    
    # Calculate duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    
    echo "" | tee -a $LOG_FILE
    print_info "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    
    # Show summary of outputs
    echo "" | tee -a $LOG_FILE
    print_info "Output files:"
    ls -lh $OUTPUT_DIR/*.json 2>/dev/null | tee -a $LOG_FILE
    
    # Calculate total size
    TOTAL_SIZE=$(du -sh $OUTPUT_DIR | cut -f1)
    print_info "Total output size: $TOTAL_SIZE"
    
    # Extract detailed accuracy from evaluation stats
    if [ -f "$OUTPUT_DIR/evaluation_stats.json" ]; then
        echo "" | tee -a $LOG_FILE
        print_info "Accuracy Summary:"
        python -c "
import json
with open('$OUTPUT_DIR/evaluation_stats.json', 'r') as f:
    stats = json.load(f)
    print(f\"  Overall Accuracy: {stats['overall']['accuracy']:.2f}% ({stats['overall']['correct']}/{stats['overall']['total_samples']})\")
    print('')
    print('  Per-Category Breakdown:')
    for cat, data in stats['categories'].items():
        print(f\"    {cat:20s}: {data['accuracy']:5.2f}% ({data['correct']:4d}/{data['total_samples']:4d})\")
        if 'unambiguous_accuracy' in data:
            print(f\"      - Unambiguous: {data['unambiguous_accuracy']:.2f}%\")
            print(f\"      - Ambiguous:   {data['ambiguous_accuracy']:.2f}%\")
" | tee -a $LOG_FILE
    fi
    
    # Create a summary file
    SUMMARY_FILE="${OUTPUT_DIR}/run_summary.txt"
    {
        echo "BBQ Full Dataset Run Summary"
        echo "============================"
        echo "Date: $(date)"
        echo "Model: $MODEL"
        echo "Seed: $SEED"
        echo "Runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
        echo "Output Directory: $OUTPUT_DIR"
        echo ""
        if [ -f "$OUTPUT_DIR/evaluation_stats.json" ]; then
            python -c "
import json
with open('$OUTPUT_DIR/evaluation_stats.json', 'r') as f:
    stats = json.load(f)
    print(f\"Overall Accuracy: {stats['overall']['accuracy']:.2f}%\")
    print(f\"Total Samples: {stats['overall']['total_samples']}\")
"
        fi
    } > $SUMMARY_FILE
    
    print_success "Summary saved to: $SUMMARY_FILE"
    
else
    print_error "Generation failed! Check the log for details."
    
    # Calculate how long it ran before failure
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    
    print_info "Failed after: ${HOURS}h ${MINUTES}m"
    exit 1
fi

echo "================================" | tee -a $LOG_FILE
print_success "Completed at $(date)"
print_info "Full log saved to: $LOG_FILE"
print_info "Results saved to: $OUTPUT_DIR"

# Send notification if available (optional)
if command -v notify-send &> /dev/null; then
    notify-send "BBQ Generation Complete" "Results saved to $OUTPUT_DIR"
fi

# Optional: Show how to analyze results
echo "" | tee -a $LOG_FILE
print_info "To analyze results:"
echo "  python scripts/analyze_bbq_results.py --input_dir $OUTPUT_DIR" | tee -a $LOG_FILE