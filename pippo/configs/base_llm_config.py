from abc import ABC
from typing import Optional


class BaseLlmConfig(ABC):
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0,
        api_key: Optional[str] = None,
        max_tokens: int = 3000,
        top_p: float = 0,
        top_k: int = 1,
        # Ollama specific
        ollama_base_url: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k

        # Ollama specific
        self.ollama_base_url = ollama_base_url
