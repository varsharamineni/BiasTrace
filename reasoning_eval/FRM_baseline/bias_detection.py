import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FairnessPRM:
    def __init__(self, model_id="zarahall/fairness-reward-model"):
        self.model_id = model_id
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16
        ).eval()

        # padding fix
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if isinstance(self.model.config.eos_token_id, list):
            self.model.config.pad_token_id = self.model.config.eos_token_id[0]
        else:
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def split_steps(self, text):
        return text.split("\n\n")

    def score_trace(self, question: str, trace: str):
        """
        Returns:
            step_scores: List[float]
            overall_score: float
        """
        steps = self.split_steps(trace)

        texts = [
            f"{question} {step}" if i == 0 else step
            for i, step in enumerate(steps)
        ]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(-1)
            step_scores = torch.sigmoid(logits).tolist()

        # overall score = mean (you can change this)
        overall_score = sum(step_scores) / len(step_scores)

        return step_scores, overall_score