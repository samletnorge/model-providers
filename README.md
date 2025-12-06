# Provider Architecture (Simple OOP)

## Overview
Clean inheritance-based provider system with no decorators. Just straightforward classes and a registry dict.

## LLM Providers

### Structure
```python
class BaseLLMProvider:
    @classmethod
    def create(cls, cfg: LLMProviderConfig) -> ResolvedProvider:
        raise NotImplementedError

class OllamaProvider(BaseLLMProvider):
    @classmethod
    def create(cls, cfg):
        # Build and return ResolvedProvider
        ...

# Registry dict
LLM_PROVIDERS = {
    "ollama": OllamaProvider,
    "azure": AzureProvider,
    "grok": GrokProvider,
    "groq": GroqProvider,
}
```

### Available Providers
- **OllamaProvider**: Self-hosted or Ollama Cloud (default: `qwen3-vl:32b`)
- **AzureProvider**: Azure OpenAI with token auth (az login) or API key (default: `gpt-4o-2`)
- **GrokProvider**: x.ai Grok models (default: `grok-2-latest`)
- **GroqProvider**: Groq cloud (default: `llama-3.3-70b-versatile`)

### Factory Usage
```python
from model_providers import get_llm_provider, LLMProviderConfig

# Use environment config
resolved = get_llm_provider()

# Or explicit config
cfg = LLMProviderConfig(provider="azure", model_name="gpt-4o-2")
resolved = get_llm_provider(cfg)

# Build chat model
from pydantic_ai.models.openai import OpenAIChatModel
model = OpenAIChatModel(
    model_name=resolved.model_name,
    provider=resolved.provider,
)
```

## Embedding Providers

### Structure
```python
class BaseEmbeddingProvider:
    def __init__(self, cfg: EmbeddingProviderConfig):
        self.cfg = cfg

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    def embed(self, texts):
        # Implementation
        ...

# Registry dict
EMBEDDING_PROVIDERS = {
    "ollama": OllamaEmbeddingProvider,
    "azure": AzureEmbeddingProvider,
}
```

### Available Providers
- **OllamaEmbeddingProvider**: Ollama embeddings (default: `nomic-embed-text`)
- **AzureEmbeddingProvider**: Azure OpenAI embeddings (default: `text-embedding-3-large`)

### Factory Usage
```python
from model_providers import get_embedding_provider

resolved = get_embedding_provider()
embeddings = resolved.provider.embed(["hello", "world"])
```

## Adding New Providers

### 1. Create Provider Class
```python
# In llm.py
class MyCustomProvider(BaseLLMProvider):
    @classmethod
    def create(cls, cfg: LLMProviderConfig) -> ResolvedProvider:
        from pydantic_ai.providers.openai import OpenAIProvider
        api_key = os.getenv("MYCUSTOM_API_KEY")
        provider = OpenAIProvider(
            api_key=api_key,
            base_url="https://api.mycustom.com/v1"
        )
        return ResolvedProvider(provider=provider, model_name=cfg.model_name)
```

### 2. Register in Dict
```python
LLM_PROVIDERS = {
    "ollama": OllamaProvider,
    "azure": AzureProvider,
    "grok": GrokProvider,
    "groq": GroqProvider,
    "mycustom": MyCustomProvider,  # Add here
}
```

### 3. Update SUPPORTED_PROVIDERS
```python
SUPPORTED_PROVIDERS = {"ollama", "azure", "grok", "groq", "mycustom"}
```

### 4. Use It
```bash
export LLM_PROVIDER=mycustom
export MYCUSTOM_API_KEY=your-key
export LLM_MODEL=model-name
```

## Environment Variables

### LLM Provider Selection
- `LLM_PROVIDER` = ollama|azure|grok|groq (default: ollama)
- `LLM_MODEL` = override default model/deployment name
- `LLM_MAX_TOKENS` = override default max tokens
- `LLM_CONTEXT_WINDOW` = override default context window
- `LLM_TIMEOUT` = request timeout in seconds (default: 604800)

### Azure Specific
- `AZURE_OPENAI_ENDPOINT` = https://your-resource.openai.azure.com/
- `AZURE_OPENAI_API_VERSION` = 2024-08-01-preview
- `AZURE_OPENAI_DEPLOYMENT` = deployment name (default: gpt-4o-2)
- `AZURE_USE_TOKEN_AUTH` = true (default, uses az login) | false (use API key)
- `AZURE_OPENAI_API_KEY` = API key (if AZURE_USE_TOKEN_AUTH=false)
- `AZURE_OPENAI_KEYVAULT_URL` = Key Vault URL for API key retrieval
- `AZURE_OPENAI_SECRET_NAME` = secret name (default: azure-openai-api-key)
- `AZURE_COGNITIVE_SERVICE_SCOPE` = token scope (default: https://cognitiveservices.azure.com/.default)

### Embedding Provider Selection
- `EMBEDDING_PROVIDER` = ollama|azure (default: ollama)
- `EMBEDDING_MODEL` = override default embedding model
- `EMBEDDING_DIMENSIONS` = embedding dimension override
- `EMBED_TIMEOUT` = request timeout (default: 60)

### Azure Embeddings
- `AZURE_OPENAI_EMBED_DEPLOYMENT` = deployment name (default: text-embedding-3-large)
- `AZURE_OPENAI_EMBED_DIMENSIONS` = dimensions override (Azure-specific)

### Other Providers
- `OLLAMA_BASE_URL` = https://ollama.valiantlynx.com/v1 (default)
- `GROK_API_KEY` = x.ai API key
- `GROQ_API_KEY` = Groq API key

## Default Models per Provider

| Provider | Default LLM Model | Max Tokens | Context Window |
|----------|-------------------|------------|----------------|
| ollama   | qwen3-vl:32b     | 131072     | 131072         |
| azure    | gpt-4o-2         | 8192       | 8192           |
| grok     | grok-2-latest    | 8192       | 8192           |
| groq     | llama-3.3-70b-versatile | 32768 | 32768      |

| Provider | Default Embedding Model |
|----------|-------------------------|
| ollama   | nomic-embed-text       |
| azure    | text-embedding-3-large |

## Architecture Benefits

1. **No Magic**: Just classes, no decorators or metaclasses
2. **Explicit Registry**: See all providers in one dict
3. **Easy Extension**: Subclass + add to dict = done
4. **Simple Testing**: Mock provider classes directly
5. **Clear Dependencies**: Import tree is straightforward
6. **IDE Friendly**: Autocomplete and navigation work perfectly

## Migration Notes

If you were using the old decorator pattern:
- `@register_llm_provider("name")` → Just add class to `LLM_PROVIDERS` dict
- `@register_embedding_provider("name")` → Add to `EMBEDDING_PROVIDERS` dict
- Imports still work the same: `from model_providers import get_llm_provider`
