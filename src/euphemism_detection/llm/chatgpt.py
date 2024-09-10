import os

from typing import Iterable, List, Dict, Any

from openai import OpenAI

from tqdm import tqdm

class ChatGPTLLM:
    def __init__(self, model_name: str, temperature: float = 1, max_tokens: int = 5):
        """ChatGPT initializer

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

        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def get_response(self, prompt: str) -> Dict[str, Any]:
        """Send request to ChatGPT API with prompt

        :param prompt: prompt to send to model
        :type prompt: str
        :return: response of API endpoint
        :rtype: Dict[str, Any]
        """
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        return response

    def generate_from_prompts(self, examples: Iterable[str]) -> List[str]:
        """Send all examples to chatGPT and get its responses

        :param examples: list of prompts
        :type examples: Iterable[str]
        :return: list of cleaned responses
        :rtype: List[str]
        """
        lines_length = len(examples)

        responses = []

        # loop through examples
        for example in tqdm(examples):
            # try to get response
            # catch any errors that happen
            try:
                response = self.get_response(example)
                responses.append(response.choices[0].message.content.replace("\n", " ").strip())
            except Exception as e:
                print(e)
                responses.append("")
                print(f"Failure of {example}")

        return responses