import os
from typing import Dict, List, Optional

try:
    from groq import Groq
except ImportError:
    raise ImportError("The 'groq' library is required. Please install it using 'pip install groq'.")

from pippo.configs.base_llm_config import BaseLlmConfig
from pippo.llms.base import LLMBase


class GroqLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "llama3-70b-8192"

        api_key = self.config.api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)

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