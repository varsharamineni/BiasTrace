from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B"  # or "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True)

prompt = "Once upon a time in a distant galaxy, "
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))