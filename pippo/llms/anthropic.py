import os
from typing import Dict, List, Optional

try:
    import anthropic
except ImportError:
    raise ImportError(
        "The 'anthropic' library is required. Please install it using 'pip install anthropic'."
    )

from pippo.configs.base_llm_config import BaseLlmConfig
from pippo.llms.base import LLMBase


class AnthropicLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "claude-3-5-sonnet-20240620"

        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
    ):

        system_message = ""
        filtered_messages = []
        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            else:
                filtered_messages.append(message)

        params = {
            "model": self.config.model,
            "messages": filtered_messages,
            "system": system_message,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        response = self.client.messages.create(**params)
        return response.content[0].text