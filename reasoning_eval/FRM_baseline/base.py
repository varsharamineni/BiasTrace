from typing import List, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from sal.config import Config

class PRM:
    def __init__(self, search_config: Config, **model_kwargs):
        self.search_config = search_config
        self.model, self.tokenizer = self.load_model_and_tokenizer(**model_kwargs)
        self.weight = 1.0  # Default weight

    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        raise NotImplementedError

    def score(
        self, questions: List[str], outputs: List[List[str]]
    ) -> List[List[float]]:
        """Score the outputs and apply the PRM's weight."""
        raw_scores = self._score(questions, outputs)
        return [[score * self.weight for score in output_scores] for output_scores in raw_scores]

    def _score(
        self, questions: List[str], outputs: List[List[str]]
    ) -> List[List[float]]:
        """Internal scoring method to be implemented by subclasses."""
        raise NotImplementedError 