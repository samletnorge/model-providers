"""Provider abstraction package for LLM and Embedding models.

Exports factory helpers:
  get_llm_provider() -> ResolvedProvider(provider, model_name)
  get_embedding_provider() -> ResolvedEmbedding(provider, model_name, dimensions)

Environment overrides:
  LLM_PROVIDER=ollama|azure|grok|groq
  LLM_MODEL=<model name>
  LLM_MAX_TOKENS=<int>
  LLM_CONTEXT_WINDOW=<int>
  EMBEDDING_PROVIDER=ollama|azure|grok|groq
  EMBEDDING_MODEL=<model name>
  EMBEDDING_DIMENSIONS=<int>

Azure:
  AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
  AZURE_OPENAI_API_KEY=... (or use token-based auth via az login)
  AZURE_USE_TOKEN_AUTH=true (default, uses DefaultAzureCredential)
  AZURE_OPENAI_KEYVAULT_URL=https://<kv>.vault.azure.net (if API key mode)
  AZURE_OPENAI_SECRET_NAME=azure-openai-api-key
  AZURE_OPENAI_EMBED_DEPLOYMENT=<embedding deployment>
  AZURE_OPENAI_API_VERSION=2024-08-01-preview

Grok (x.ai):
  GROK_API_KEY=...

Groq:
  GROQ_API_KEY=...

Ollama:
  OLLAMA_BASE_URL=http://localhost:11434/v1

Extension:
  Add new providers by subclassing BaseLLMProvider / BaseEmbeddingProvider
  and adding entries to LLM_PROVIDERS / EMBEDDING_PROVIDERS dicts.
"""

from .llm import (
    get_llm_provider,
    get_available_llm_models,
    get_available_llm_providers,
    LLMProviderConfig,
    ResolvedProvider,
    LLM_PROVIDERS,
    BaseLLMProvider,
)
from .embeddings import (
    get_embedding_provider,
    get_available_embedding_models,
    get_available_embedding_providers,
    EmbeddingProviderConfig,
    ResolvedEmbedding,
    EMBEDDING_PROVIDERS,
    BaseEmbeddingProvider,
)

__all__ = [
    "get_llm_provider",
    "get_available_llm_models",
    "get_available_llm_providers",
    "ResolvedProvider",
    "LLMProviderConfig",
    "LLM_PROVIDERS",
    "BaseLLMProvider",
    "get_embedding_provider",
    "get_available_embedding_models",
    "get_available_embedding_providers",
    "EmbeddingProviderConfig",
    "ResolvedEmbedding",
    "EMBEDDING_PROVIDERS",
    "BaseEmbeddingProvider",
]
