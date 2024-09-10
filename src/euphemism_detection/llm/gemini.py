import os

from typing import Iterable, List, Dict, Any

import google.generativeai as genai

from tqdm import tqdm


class GeminiLLM:
    def __init__(self, model_name: str, temperature: float = 1, max_tokens: int = 5):
        """Gemini model initializer

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

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        self.model = genai.GenerativeModel(self.model_name)

    def get_response(self, prompts: Iterable[str]) -> str:
        """Get response from Gemini model with prompt batch

        :param prompt: prompt to send to model
        :type prompt: Iterable[str]
        :return: response text from API
        :rtype: str
        """
        response = self.model.generate_content(prompts, generation_config=genai.types.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            ))

        return response.text

    def generate_from_prompts(self, examples: Iterable[str]) -> List[str]:
        """Send all examples to offline HF model and get its responses

        :param examples: list of prompts
        :type examples: Iterable[str]
        :return: list of cleaned responses
        :rtype: List[str]
        """
        responses = []

        for i in tqdm(range(0, len(examples), ncols=0)):
            response = self.get_response(examples[i])

            responses.extend(response)

        return responses