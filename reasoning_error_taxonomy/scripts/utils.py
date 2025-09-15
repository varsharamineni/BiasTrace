from anthropic import Anthropic
import os
import re  
import json


client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def call_claude(prompt, model="claude-opus-4-1-20250805", max_tokens=2000, temperature=0.0, thinking=False):
    """
    Sends a prompt to Claude and returns a single string containing the model's completion.
    'thinking' is optional; if True, enables extended reasoning steps.
    """
    messages = [{"role": "user", "content": prompt}]

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    if thinking:
        kwargs["thinking"] = {"enabled": True, "type": "default"}  # only add if True

    response = client.messages.create(**kwargs)

    # concatenate all text blocks into a single string
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


def extract_json_array(text):
    """
    Attempt to parse JSON array even if it's truncated.
    Returns a Python list.
    """
    import json, re

    # Remove markdown fences
    text = re.sub(r"```json|```", "", text)

    # Match the first array
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        json_text = match.group(0)

        # Try to close any unclosed brackets/braces (simple fix)
        open_brackets = json_text.count("[") - json_text.count("]")
        if open_brackets > 0:
            json_text += "]" * open_brackets
        open_braces = json_text.count("{") - json_text.count("}")
        if open_braces > 0:
            json_text += "}" * open_braces

        # Fix trailing commas
        json_text = re.sub(r",\s*}", "}", json_text)
        json_text = re.sub(r",\s*]", "]", json_text)

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print("Partial JSON could not be loaded:", e)
            return None
    return None