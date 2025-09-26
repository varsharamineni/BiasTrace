#!/bin/bash
# Test mode for BBQ generation with remote GPT-OSS-20B vLLM deployment

MODEL="openai/gpt-oss-120b"
OUTPUT_BASE="outputs/gpt-oss-120B_test_client_new"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/run.log"
BATCH_SIZE=8
SEED=42
REASONING_LEVEL="low"
TEMPERATURE=1.0
TOP_P=1.0
MAX_LENGTH=2048
CATEGORIES="Age"
API_URL="https://gpt-oss-120b-bbq.nvidia-oci.saturnenterprise.io/v1"  # Replace with your deployment URL

mkdir -p $OUTPUT_DIR

print_info() { echo -e "\033[1;34m[INFO]\033[0m $1" | tee -a $LOG_FILE; }
print_success() { echo -e "\033[1;32m[SUCCESS]\033[0m $1" | tee -a $LOG_FILE; }
print_error() { echo -e "\033[1;31m[ERROR]\033[0m $1" | tee -a $LOG_FILE; }

print_info "Starting BBQ TEST MODE generation at $(date)"

poetry run python scripts/generate_bbq_outputs_vllm_gpt-oss.py \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --categories $CATEGORIES \
    --seed $SEED \
    --test_mode \
    --reasoning_level $REASONING_LEVEL \
    --api_url $API_URL \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --max_output_tokens $MAX_LENGTH \
    --model $MODEL \
    2>&1 | tee -a $LOG_FILE

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_success "Generation completed successfully!"
    if [ -f "$OUTPUT_DIR/evaluation_stats.json" ]; then
        python - <<PYTHON | tee -a $LOG_FILE
import json
with open('$OUTPUT_DIR/evaluation_stats.json') as f:
    stats = json.load(f)
print(f"Overall accuracy: {stats['overall']['accuracy']:.2f}%")
for cat, val in stats['categories'].items():
    print(f"{cat}: {val['accuracy']:.2f}%")
PYTHON
    fi
else
    print_error "Generation failed!"
fi
