#!/bin/bash
# Full dataset script for BBQ Best-of-N generation (Qwen thinking mode + LLM judge)
# Generates N candidates per question, judges each, keeps the best.

# --- Judge API credentials (same as reasoning_eval setup) ---
export LLM_BASE_URL="https://api.deepseek.com/v1"
# export LLM_API_KEY="..."   # set before running

# Configuration
MODEL="/home/vramineni/models/qwen3-32B"
OUTPUT_BASE="outputs/qwen_full_32B_bon"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/run.log"
TENSOR_PARALLEL=2
BATCH_SIZE=16
BEST_OF_N=8
SEED=42

CATEGORIES="Age Disability_status Gender_identity Nationality Physical_appearance Race_ethnicity Race_x_SES Race_x_gender Religion SES Sexual_orientation"

# Generation options
GPU_MEMORY_UTIL=0.8
MAX_LENGTH=2048
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20

# Judge options
JUDGE_MODEL="deepseek-chat"
JUDGE_PROMPT="reasoning_eval/prompts/new_prompt_bias_pathways_simple.txt"
SCORE_FIELD="bias_label"     # judge emits {"bias_label": 0|1}
PASS_SCORE=0                 # with --invert_score, an unbiased (bias_label 0) trace scores 0
JUDGE_WORKERS=8

mkdir -p $OUTPUT_DIR

print_info()    { echo -e "\033[1;34m[INFO]\033[0m $1"    | tee -a $LOG_FILE; }
print_success() { echo -e "\033[1;32m[SUCCESS]\033[0m $1" | tee -a $LOG_FILE; }
print_error()   { echo -e "\033[1;31m[ERROR]\033[0m $1"   | tee -a $LOG_FILE; }
print_warning() { echo -e "\033[1;33m[WARNING]\033[0m $1" | tee -a $LOG_FILE; }

if [ -z "$LLM_API_KEY" ]; then
    print_error "LLM_API_KEY is not set. Export it before running."
    exit 1
fi

print_info "Starting BBQ BEST-OF-N generation at $(date)"
print_warning "N=${BEST_OF_N}: ~${BEST_OF_N}x generation tokens and exactly ${BEST_OF_N} judge calls per question (filtered majority vote judges all candidates)."
echo "" | tee -a $LOG_FILE
print_info "Configuration:"
echo "  Model: $MODEL"                | tee -a $LOG_FILE
echo "  Best-of-N: $BEST_OF_N"        | tee -a $LOG_FILE
echo "  Judge: $JUDGE_MODEL ($JUDGE_PROMPT)" | tee -a $LOG_FILE
echo "  Score field: $SCORE_FIELD (inverted; candidates pass at >= $PASS_SCORE)" | tee -a $LOG_FILE
echo "  Output: $OUTPUT_DIR"          | tee -a $LOG_FILE
echo "  Categories: $CATEGORIES"      | tee -a $LOG_FILE
echo "  Tensor Parallel: $TENSOR_PARALLEL GPUs" | tee -a $LOG_FILE
echo "  Batch Size: $BATCH_SIZE"      | tee -a $LOG_FILE
echo "  Seed: $SEED"                  | tee -a $LOG_FILE
echo "================================" | tee -a $LOG_FILE

START_TIME=$(date +%s)
print_info "Starting generation..."

poetry run python scripts/generate_bbq_outputs_vllm_bon.py \
    --model $MODEL \
    --output_dir $OUTPUT_DIR \
    --categories $CATEGORIES \
    --best_of_n $BEST_OF_N \
    --judge_model $JUDGE_MODEL \
    --judge_prompt $JUDGE_PROMPT \
    --score_field $SCORE_FIELD \
    --invert_score \
    --pass_score $PASS_SCORE \
    --judge_temperature 0.0 \
    --judge_max_workers $JUDGE_WORKERS \
    --judge_on reasoning \
    --save_all_candidates \
    --tensor_parallel_size $TENSOR_PARALLEL \
    --batch_size $BATCH_SIZE \
    --gpu_memory_utilization $GPU_MEMORY_UTIL \
    --max_length $MAX_LENGTH \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --seed $SEED \
    --enable_thinking \
    --quiet \
    2>&1 | tee -a $LOG_FILE

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_success "Generation completed successfully!"

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600)); MINUTES=$(((DURATION % 3600) / 60)); SECONDS=$((DURATION % 60))
    echo "" | tee -a $LOG_FILE
    print_info "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"

    echo "" | tee -a $LOG_FILE
    print_info "Output files:"
    ls -lh $OUTPUT_DIR/*.json 2>/dev/null | tee -a $LOG_FILE
    TOTAL_SIZE=$(du -sh $OUTPUT_DIR | cut -f1)
    print_info "Total output size: $TOTAL_SIZE"

    if [ -f "$OUTPUT_DIR/evaluation_stats.json" ]; then
        echo "" | tee -a $LOG_FILE
        print_info "Accuracy Summary:"
        python -c "
import json
with open('$OUTPUT_DIR/evaluation_stats.json', 'r') as f:
    stats = json.load(f)
    o = stats['overall']
    n = stats['best_of_n']
    print(f\"  {'Selection method':<34}{'Accuracy':>10}\")
    print('  ' + '-'*44)
    print(f\"  {'Single sample':<34}{o['baseline_accuracy_first_sample']:>9.2f}%\")
    print(f\"  {'Majority vote over all N':<34}{o['majority_vote_accuracy_all']:>9.2f}%\")
    print(f\"  {'Majority vote over judge-passed':<34}{o['accuracy_filtered_majority']:>9.2f}%\")
    print(f\"  {'Oracle (pass@' + str(n) + ')':<34}{o['oracle_accuracy_pass_at_n']:>9.2f}%\")
    print('')
    print(f\"  Candidate pass rate: {o['candidate_pass_rate']:.1f}% | questions with >=1 pass: {o['questions_with_a_passing_candidate']:.1f}%\")
    print(f\"  Fallback used: {o['fallback_used_pct']:.1f}% | filtering changed answer: {o['filtering_changed_answer']:.1f}%\")
    print(f\"  Judge API calls: {stats['total_judge_api_calls']}\")
    print('')
    print('  Per-Category (filtered / all-majority / single / oracle):')
    for cat, d in stats['categories'].items():
        print(f\"    {cat:22s}: {d['accuracy']:5.2f}% / {d['majority_vote_accuracy_all']:5.2f}% / {d['baseline_accuracy_first_sample']:5.2f}% / {d['oracle_accuracy_pass_at_n']:5.2f}%\")
        print(f\"      - Unambiguous: {d['unambiguous_accuracy']:.2f}% | Ambiguous: {d['ambiguous_accuracy']:.2f}%\")
" | tee -a $LOG_FILE
    fi

    SUMMARY_FILE="${OUTPUT_DIR}/run_summary.txt"
    {
        echo "BBQ Best-of-N Run Summary"
        echo "========================="
        echo "Date: $(date)"
        echo "Model: $MODEL"
        echo "Best-of-N: $BEST_OF_N"
        echo "Judge: $JUDGE_MODEL / $JUDGE_PROMPT"
        echo "Seed: $SEED"
        echo "Runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
        echo "Output Directory: $OUTPUT_DIR"
    } > $SUMMARY_FILE
    print_success "Summary saved to: $SUMMARY_FILE"
else
    print_error "Generation failed! Check the log for details."
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600)); MINUTES=$(((DURATION % 3600) / 60))
    print_info "Failed after: ${HOURS}h ${MINUTES}m"
    exit 1
fi

echo "================================" | tee -a $LOG_FILE
print_success "Completed at $(date)"
print_info "Next steps:"
echo "  python scripts/process_bbq_results.py --base_folders $OUTPUT_DIR --meta_file datasets/bbq_additional_metadata.csv" | tee -a $LOG_FILE