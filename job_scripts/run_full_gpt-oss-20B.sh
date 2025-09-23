#!/bin/bash
# Full dataset script for BBQ generation using GPT-OSS-20B vLLM deployment
# Supports automated resumption per category and configurable reasoning level

# -------------------------
# Configuration
# -------------------------
MODEL="openai/gpt-oss-20b"
OUTPUT_BASE="outputs/gpt-oss-20B_full_simple_prompt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/20250922_133414"
LOG_FILE="${OUTPUT_DIR}/run.log"
BATCH_SIZE=32
SEED=42

# All BBQ categories
CATEGORIES="Age Disability_status Gender_identity Nationality Physical_appearance Race_ethnicity Religion SES Sexual_orientation"

# Sampling / reasoning options (default)
MAX_LENGTH=2048
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
REASONING_LEVEL="${REASONING_LEVEL:-medium}"  # Can be overridden via environment variable

# API URL for vLLM deployment
API_URL="https://gpt-oss-20b.nvidia-oci.saturnenterprise.io"  # Replace with your server IP

# -------------------------
# Create output directory
# -------------------------
mkdir -p "$OUTPUT_DIR"

# -------------------------
# Logging helpers
# -------------------------
print_info() { echo -e "\033[1;34m[INFO]\033[0m $1" | tee -a "$LOG_FILE"; }
print_success() { echo -e "\033[1;32m[SUCCESS]\033[0m $1" | tee -a "$LOG_FILE"; }
print_error() { echo -e "\033[1;31m[ERROR]\033[0m $1" | tee -a "$LOG_FILE"; }
print_warning() { echo -e "\033[1;33m[WARNING]\033[0m $1" | tee -a "$LOG_FILE"; }

# -------------------------
# Start logging
# -------------------------
print_info "Starting BBQ FULL DATASET generation at $(date)"
print_warning "This is a FULL dataset run and may take several hours!"
print_info "Configuration:"
echo "  Model: $MODEL" | tee -a "$LOG_FILE"
echo "  Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "  Categories: $CATEGORIES" | tee -a "$LOG_FILE"
echo "  Tensor Parallel: $TENSOR_PARALLEL" | tee -a "$LOG_FILE"
echo "  Batch Size: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  GPU Memory: $GPU_MEMORY_UTIL" | tee -a "$LOG_FILE"
echo "  Max Length: $MAX_LENGTH tokens" | tee -a "$LOG_FILE"
echo "  Temperature: $TEMPERATURE" | tee -a "$LOG_FILE"
echo "  Top-p: $TOP_P" | tee -a "$LOG_FILE"
echo "  Top-k: $TOP_K" | tee -a "$LOG_FILE"
echo "  Seed: $SEED" | tee -a "$LOG_FILE"
echo "  Reasoning Level: $REASONING_LEVEL" | tee -a "$LOG_FILE"
echo "  Mode: FULL DATASET" | tee -a "$LOG_FILE"
echo "================================" | tee -a "$LOG_FILE"

START_TIME=$(date +%s)

# -------------------------
# Loop over categories with automated resumption
# -------------------------
for category in $CATEGORIES; do
    CATEGORY_FILE="${OUTPUT_DIR}/bbq_${category}_results.json"
    if [ -f "$CATEGORY_FILE" ]; then
        print_info "Skipping $category (already exists)"
        continue
    fi

    print_info "Processing category: $category"

    poetry run python scripts/generate_bbq_outputs_vllm_gpt-oss_simple.py \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --categories "$category" \
        --max_length "$MAX_LENGTH" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --top_k "$TOP_K" \
        --seed "$SEED" \
        --reasoning_level "$REASONING_LEVEL" \
        --api_url "$API_URL" \
        --quiet \
        2>&1 | tee -a "$LOG_FILE"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        print_error "Generation failed for $category! Check logs."
        break
    fi
done

# -------------------------
# Duration
# -------------------------
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))
print_info "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"

# -------------------------
# Output summary
# -------------------------
print_info "Output files:"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null | tee -a "$LOG_FILE"
TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
print_info "Total output size: $TOTAL_SIZE"

# Accuracy summary
if [ -f "$OUTPUT_DIR/evaluation_stats.json" ]; then
    print_info "Accuracy Summary:"
    python - <<PYTHON | tee -a "$LOG_FILE"
import json
with open('$OUTPUT_DIR/evaluation_stats.json', 'r') as f:
    stats = json.load(f)
    print(f"Overall Accuracy: {stats['overall']['accuracy']:.2f}% ({stats['overall']['correct']}/{stats['overall']['total_samples']})\n")
    print("Per-Category Breakdown:")
    for cat, data in stats['categories'].items():
        print(f"  {cat:20s}: {data['accuracy']:5.2f}% ({data['correct']:4d}/{data['total_samples']:4d})")
        if 'unambiguous_accuracy' in data:
            print(f"    - Unambiguous: {data['unambiguous_accuracy']:.2f}%")
            print(f"    - Ambiguous:   {data['ambiguous_accuracy']:.2f}%")
PYTHON
fi

# -------------------------
# Summary file
# -------------------------
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
        python - <<PYTHON
import json
with open('$OUTPUT_DIR/evaluation_stats.json','r') as f:
    stats = json.load(f)
    print(f"Overall Accuracy: {stats['overall']['accuracy']:.2f}%")
    print(f"Total Samples: {stats['overall']['total_samples']}")
PYTHON
    fi
} > "$SUMMARY_FILE"
print_success "Summary saved to: $SUMMARY_FILE"

print_success "Completed at $(date)"
print_info "Full log saved to: $LOG_FILE"
print_info "Results saved to: $OUTPUT_DIR"

# Optional notification
if command -v notify-send &> /dev/null; then
    notify-send "BBQ Generation Complete" "Results saved to $OUTPUT_DIR"
fi

# Analysis hint
print_info "To analyze results:"
echo "  python scripts/analyze_bbq_results.py --input_dir $OUTPUT_DIR" | tee -a "$LOG_FILE"
