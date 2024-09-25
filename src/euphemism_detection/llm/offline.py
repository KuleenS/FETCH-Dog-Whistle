import torch

from typing import Iterable, List, Dict, Any

from transformers import pipeline

from tqdm import tqdm


class OfflineLLM:
    def __init__(self, model_name: str, temperature: float = 1, max_tokens: int = 1):
        """HF offline model initializer

        :param model_name: name of model
        :type model_name: str
        :param temperature: temperature of model when generating, defaults to 1
        :type temperature: float, optional
        :param max_tokens: maximum number of tokens generated, defaults to 5
        :type max_tokens: int, optional
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.pipeline_model = pipeline(
            "text-generation", model=self.model_name, device_map="auto", batch_size=4
        )

        if isinstance(self.pipeline_model.model.config.eos_token_id, list):
            self.pipeline_model.tokenizer.pad_token_id = (
                self.pipeline_model.model.config.eos_token_id[0]
            )
        else:
            self.pipeline_model.tokenizer.pad_token_id = (
                self.pipeline_model.model.config.eos_token_id
            )

        self.pipeline_model.tokenizer.padding_side = "left"

    def format_response(self, response: str, prompt: str) -> str:
        """Clean up response from Offline HF model and return generated string

        :param response: response from Offline HF model
        :type response: Dict[str, Any]
        :return: generated string
        :rtype: str
        """
        text = response[len(prompt) :].replace("\n", " ").strip().lower()
        return text

    def generate_from_prompts(self, examples: Iterable[str]) -> List[str]:
        """Send all examples to offline HF model and get its responses

        :param examples: list of prompts
        :type examples: Iterable[str]
        :return: list of cleaned responses
        :rtype: List[str]
        """
        with torch.inference_mode():
            responses = self.pipeline_model(examples, max_new_tokens=self.max_tokens)

        responses = [x[0]["generated_text"] for x in responses]

        responses = [
            self.format_response(x, prompt) for x, prompt in zip(responses, examples)
        ]

        del self.pipeline_model

        torch.cuda.empty_cache()

        return responses
