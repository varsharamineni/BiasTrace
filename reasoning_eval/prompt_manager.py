import json
from string import Template

class PromptManager:
    def __init__(self, prompt_file: str):
        # Load prompts from a JSON file
        with open(prompt_file, "r") as f:
            self.prompts = json.load(f)
    
    def get_prompt(self, prompt_name: str, **kwargs):
        """
        Fill placeholders in the selected prompt.
        Usage: get_prompt("summarize", text="Your text here")
        """
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found.")
        
        # Using str.format
        prompt_template = self.prompts[prompt_name]
        prompt_filled = prompt_template.format(**kwargs)
        return prompt_filled