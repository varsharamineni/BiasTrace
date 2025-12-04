#!/usr/bin/env python
import os
from openai import OpenAI

# ===============================
# Set environment variables
# ===============================
# ===============================
# Create client
# ===============================
client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

# ===============================
# Test request
# ===============================
try:
    response = client.chat.completions.create(
        model="DeepSeek-R1-Distill-Qwen-70B",
        messages=[{"role": "user", "content": "Hello, DeepSeek! Can you summarize this in one sentence?"}],
        max_tokens=50
    )

    print("✅ Server is running. Response:")
    print(response.choices[0].message["content"])

except Exception as e:
    print("❌ Error connecting to vLLM server:")
    print(e)
