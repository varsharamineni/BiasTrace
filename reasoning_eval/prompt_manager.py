import os

class PromptManager:

    def __init__(self, prompt_source: str):
        """
        Load prompts either from a JSON file or a directory of .txt files.
        - If prompt_source is a JSON file, loads as before.
        - If prompt_source is a directory, loads each .txt file as a prompt.
        """
        self.prompts = {}
        if os.path.isfile(prompt_source) and prompt_source.endswith(".json"):
            import json
            with open(prompt_source, "r", encoding="utf-8") as f:
                self.prompts = json.load(f)
        elif os.path.isdir(prompt_source):
            for filename in os.listdir(prompt_source):
                if filename.endswith(".txt"):
                    prompt_name = os.path.splitext(filename)[0]
                    with open(os.path.join(prompt_source, filename), "r", encoding="utf-8") as f:
                        self.prompts[prompt_name] = f.read()
        else:
            raise ValueError(f"Prompt source '{prompt_source}' is neither a JSON file nor a directory.")

    def get_prompt(self, prompt_name: str, **kwargs):
        """
        Fill placeholders in the selected prompt.
        Usage: get_prompt('summarize', text='Your text here')
        """
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found.")
        
        prompt_template = self.prompts[prompt_name]
        prompt_filled = prompt_template.format(**kwargs)
        return prompt_filled
