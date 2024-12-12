import os
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("The 'openai' library is required. Please install it using 'pip install openai'.")

from pippo.configs.base_llm_config import BaseLlmConfig
from pippo.llms.base import LLMBase


class OpenAILLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "gpt-4o-mini"

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def _parse_response(self, response):
        return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
    ):
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }


        if response_format:
            params["response_format"] = response_format

        response = self.client.chat.completions.create(**params)
        return self._parse_response(response)