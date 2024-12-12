from typing import Dict, List, Optional

try:
    from ollama import Client
except ImportError:
    raise ImportError("The 'ollama' library is required. Please install it using 'pip install ollama'.")

from pippo.configs.base_llm_config import BaseLlmConfig
from pippo.llms.base import LLMBase


class OllamaLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "llama3.2:3b"
        self.client = Client(host=self.config.ollama_base_url)
        self._ensure_model_exists()

    def _ensure_model_exists(self):
        local_models = self.client.list()["models"]
        if not any(model.get("name") == self.config.model for model in local_models):
            self.client.pull(self.config.model)

    def _parse_response(self, response):
        return response["message"]["content"]

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
    ):
        params = {
            "model": self.config.model,
            "messages": messages,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p,
            },
        }
        if response_format:
            params["format"] = "json"

        response = self.client.chat(**params)
        return self._parse_response(response)