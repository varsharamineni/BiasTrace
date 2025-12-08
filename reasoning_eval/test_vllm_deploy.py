from openai import OpenAI
import os

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")  # optional if your deployment doesn’t require authentication
)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    messages=[{"role": "user", "content": "Hello! Summarize this in one sentence."}],
    max_tokens=50
)

# Correct access
print("✅ Server is running. Response:")
print(response.choices[0].message.content)