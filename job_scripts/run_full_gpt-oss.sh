#!/bin/bash
# Full dataset script for BBQ generation using GPT-OSS vLLM deployment
# Supports automated resumption per category and configurable reasoning level

# -------------------------
# Configuration
# -------------------------
MODEL="${MODEL:-openai/gpt-oss-120b}"        # Model name (e.g., openai/gpt-oss-120b)
REASONING_LEVEL="${REASONING_LEVEL:-medium}" # default "medium", or choose "low", "high"
API_URL="${API_URL:-https://gpt-oss-120b-bbq.nvidia-oci.saturnenterprise.io/v1}" # default API URL

# Prompt style: simple, full, or custom
PROMPT_STYLE="${PROMPT_STYLE:-simple}"    

# Optional custom prompt file or raw string
CUSTOM_PROMPT_FILE="${CUSTOM_PROMPT_FILE:-}"  # Path to custom prompt file
CUSTOM_PROMPT="${CUSTOM_PROMPT:-}"            # Raw custom prompt string

BATCH_SIZE=32
SEED=42
MAX_LENGTH=2048
TEMPERATURE=1.0
TOP_P=1.0

# All BBQ categories
CATEGORIES="Age Disability_status Gender_identity Nationality Physical_appearance Race_ethnicity Religion SES Sexual_orientation"

# -------------------------
# Extract model short name (e.g., gpt-oss-20B)
# -------------------------
MODEL_SHORT=$(echo "$MODEL" | awk -F/ '{print $2}' | sed 's/-b/B/')

# -------------------------
# Handle timestamped output directory with optional resume
# -------------------------
if [ -n "$RESUME_DIR" ]; then
    OUTPUT_DIR="$RESUME_DIR"
    mkdir -p "$OUTPUT_DIR"
    echo "[INFO] Resuming in existing directory: $OUTPUT_DIR"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="outputs/${MODEL_SHORT}_${PROMPT_STYLE}_prompt_${REASONING_LEVEL}_reasoning/${TIMESTAMP}"
    mkdir -p "$OUTPUT_DIR"
    echo "[INFO] Starting new run in directory: $OUTPUT_DIR"
fi

LOG_FILE="${OUTPUT_DIR}/run.log"

# -------------------------
# Generation script (can override with GEN_SCRIPT env variable)
# -------------------------
GEN_SCRIPT="${GEN_SCRIPT:-scripts/generate_bbq_outputs_vllm_gpt-oss.py}"

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
echo "  Batch Size: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  Max Length: $MAX_LENGTH tokens" | tee -a "$LOG_FILE"
echo "  Temperature: $TEMPERATURE" | tee -a "$LOG_FILE"
echo "  Top-p: $TOP_P" | tee -a "$LOG_FILE"
echo "  Seed: $SEED" | tee -a "$LOG_FILE"
echo "  Reasoning Level: $REASONING_LEVEL" | tee -a "$LOG_FILE"
echo "  Prompt Style: $PROMPT_STYLE" | tee -a "$LOG_FILE"
echo "================================" | tee -a "$LOG_FILE"

START_TIME=$(date +%s)

poetry run python "$GEN_SCRIPT" \
        --model "$MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --categories $CATEGORIES \
        --max_output_tokens "$MAX_LENGTH" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --seed "$SEED" \
        --reasoning_level "$REASONING_LEVEL" \
        --api_url "$API_URL" \
        --prompt_type "$PROMPT_STYLE" \
        ${CUSTOM_PROMPT_FILE:+--custom_prompt_file "$CUSTOM_PROMPT_FILE"} \
        ${CUSTOM_PROMPT:+--custom_prompt "$CUSTOM_PROMPT"} \
        --quiet \
        2>&1 | tee -a "$LOG_FILE"

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

