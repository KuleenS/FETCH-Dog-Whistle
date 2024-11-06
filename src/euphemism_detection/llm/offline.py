from typing import Iterable, List, Dict, Any

from jsonformer import Jsonformer

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


class OfflineLLM:
    def __init__(self, model_name: str, max_tokens: int = 1):
        """HF offline model initializer

        :param model_name: name of model
        :type model_name: str
        :param temperature: temperature of model when generating, defaults to 1
        :type temperature: float, optional
        :param max_tokens: maximum number of tokens generated, defaults to 5
        :type max_tokens: int, optional
        """
        self.model_name = model_name
        self.max_tokens = max_tokens

        self.dogwhistle_schema = {
            "dogwhistles": {"type": "array", "items": {"type": "string"}}
        }

        self.prediction_schema = {
            "predictions": {"type": "array", "items": {"type": "string"}}
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, use_cache=True, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, use_cache=True
        )

        if isinstance(self.model.config.eos_token_id, list):
            self.tokenizer.pad_token_id = self.model.config.eos_token_id[0]
        else:
            self.tokenizer.pad_token_id = self.model.config.eos_token_id

        self.tokenizer.padding_side = "left"

    def generate_from_prompts(self, examples: Iterable[str], schema: str) -> List[str]:
        """Send all examples to offline HF model and get its responses

        :param examples: list of prompts
        :type examples: Iterable[str]
        :return: list of cleaned responses
        :rtype: List[str]
        """

        responses = []

        schema = (
            self.dogwhistle_schema if schema == "dogwhistle" else self.prediction_schema
        )

        for example in examples:

            jsonformer = Jsonformer(
                model=self.model,
                tokenizer=self.tokenizer,
                json_schema=schema,
                prompt=example,
                max_string_token_length=self.max_tokens,
                max_array_length=self.max_tokens,
            )

            generated_data = jsonformer()

            responses.append(generated_data)

        return responses
