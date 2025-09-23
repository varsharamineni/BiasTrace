#!/usr/bin/env python
import os
import requests

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "https://gpt-oss-20b.nvidia-oci.saturnenterprise.io"  # Your Saturn endpoint
SATURN_TOKEN = os.environ.get("OPENAI_API_KEY")  # Set this to your Saturn token

if not SATURN_TOKEN:
    raise ValueError("Please set the OPENAI_API_KEY environment variable with your Saturn token")

# -----------------------------
# TEST REQUEST
# -----------------------------
payload = {
    "messages": [{"role": "user", "content": "Hello, can you respond briefly?"}],
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 50
}

headers = {"Authorization": f"Bearer {SATURN_TOKEN}", "Content-Type": "application/json"}

try:
    response = requests.post(f"{API_URL}/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    print("✅ Deployment works! Model response:")
    print(data["choices"][0]["message"]["content"])
except requests.exceptions.HTTPError as e:
    print(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
except requests.exceptions.RequestException as e:
    print(f"❌ Request failed: {e}")
