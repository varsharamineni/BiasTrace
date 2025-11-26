#!/bin/bash
# Example script to run LLM judge with local API server
# Configuration is loaded from reasoning_eval/lm_config.py by default

# Paths
PROMPT_PATH="tests/judge_optimized_prompt.json"
DATA_PATH="reasoning_eval/data_to_label/sample_traces_inital.json"
OUTPUT_DIR="reasoning_eval/llm_judge_samples/"

echo "🚀 Running LLM judge evaluation..."
echo "📄 Prompt: $PROMPT_PATH"
echo "📊 Data: $DATA_PATH"
echo "💾 Output: $OUTPUT_DIR"
echo ""

# Run the judge script (uses defaults from lm_config.py)
.venv/bin/python reasoning_eval/llm_judge_script.py \
  --prompt_path "$PROMPT_PATH" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_samples 5

# Optional: Override defaults by uncommenting and modifying:
# .venv/bin/python reasoning_eval/llm_judge_script.py \
#   --model "different-model" \
#   --prompt_path "$PROMPT_PATH" \
#   --data_path "$DATA_PATH" \
#   --output_dir "$OUTPUT_DIR" \
#   --api_base "http://localhost:8001/v1" \
#   --temperature 0.5 \
#   --max_samples 5

echo ""
echo "✅ Judge evaluation complete!"
echo "📊 Results saved to: $OUTPUT_DIR"
echo ""
echo "💡 To customize settings, edit: reasoning_eval/lm_config.py"

