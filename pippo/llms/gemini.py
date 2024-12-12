import os
from typing import Dict, List, Optional

try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
except ImportError:
    raise ImportError(
        "The 'google-generativeai' library is required. Please install it using 'pip install google-generativeai'."
    )

from pippo.configs.base_llm_config import BaseLlmConfig
from pippo.llms.base import LLMBase


class GeminiLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "gemini-1.5-flash-latest"

        api_key = self.config.api_key or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.client = GenerativeModel(model_name=self.config.model)

    def _parse_response(self, response):
        return response.candidates[0].content.parts[0].text

    def _reformat_messages(self, messages: List[Dict[str, str]]):
        new_messages = []

        for message in messages:
            if message["role"] == "system":
                content = "THIS IS A SYSTEM PROMPT. YOU MUST OBEY THIS: " + message["content"]

            else:
                content = message["content"]

            new_messages.append({"parts": content, "role": "model" if message["role"] == "model" else "user"})

        return new_messages

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
    ):
        params = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        if response_format:
            params["response_mime_type"] = "application/json"
            params["response_schema"] = list[response_format]

        response = self.client.generate_content(
            contents=self._reformat_messages(messages),
            generation_config=genai.GenerationConfig(**params),
        )

        return self._parse_response(response)