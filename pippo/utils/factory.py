import importlib

from pippo.configs.base_llm_config import BaseLlmConfig


def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LlmFactory:
    provider_to_class = {
        "ollama": "pippo.llms.ollama.OllamaLLM",
        "openai": "pippo.llms.openai.OpenAILLM",
        "groq": "pippo.llms.groq.GroqLLM",
        "azure_openai": "pippo.llms.azure_openai.AzureOpenAILLM",
        "gemini": "pippo.llms.gemini.GeminiLLM",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            llm_instance = load_class(class_type)
            base_config = BaseLlmConfig(**config)
            return llm_instance(base_config)
        else:
            raise ValueError(f"Unsupported Llm provider: {provider_name}")
