import os
import json
from datasets import load_dataset
from vllm import LLM, SamplingParams

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "Qwen/Qwen3-1.7B"   # or Qwen/Qwen3-4B if memory allows
NUM_EXAMPLES = 5
ENABLE_THINKING = True

# Force GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ----------------------------
# LOAD STEREOSET (intrasentence)
# ----------------------------
dataset = load_dataset(
    "stereoset",
    "intrasentence",
    split="validation"
)

examples = dataset.select(range(NUM_EXAMPLES))
print(f"Loaded {len(examples)} examples from StereoSet (intrasentence)")

# ----------------------------
# BUILD PROMPTS
# ----------------------------
messages_batch = []

for ex in examples:
    context = ex["context"]
    target = ex["target"]

    sentences = ex["sentences"]["sentence"]
    gold_labels = ex["sentences"]["gold_label"]

    prompt = f"Context: {context}\n"
    if "BLANK" in context:
        prompt += "Fill in the blank by choosing the best option.\n\n"
    else:
        prompt += "\nChoose the most appropriate continuation:\n\n"

    for i, s in enumerate(sentences):
        prompt += f"{i+1}. {s}\n"

    prompt += "\nAnswer with the option number only."

    messages_batch.append([
        {"role": "user", "content": prompt}
    ])

# ----------------------------
# INIT vLLM
# ----------------------------
llm = LLM(
    model=MODEL_NAME,
    dtype="float16",          # correct dtype for vLLM
    tensor_parallel_size=1,
    disable_log_stats=True
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048
)

# ----------------------------
# RUN MODEL (THINKING MODE)
# ----------------------------
outputs = llm.chat(
    messages_batch,
    sampling_params,
    chat_template_kwargs={"enable_thinking": ENABLE_THINKING},
    use_tqdm=False
)

# ----------------------------
# PRINT RESULTS
# ----------------------------
for i, out in enumerate(outputs):
    print("=" * 80)
    print(f"Example {i}")
    print(out.outputs[0].text)
