"""Embedding provider factory.

Currently pydantic-ai might not yet expose a unified EmbeddingModel abstraction comparable
to chat models; we supply a lightweight shim with pluggable backends for future use.

Environment variables:
  EMBEDDING_PROVIDER=ollama|azure|grok|groq (default: ollama)
  EMBEDDING_MODEL=<model name override>

Provider-specific variables:
Azure:
  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_EMBED_DEPLOYMENT
Grok:
  GROK_API_KEY
Groq:
  GROQ_API_KEY
Ollama:
  OLLAMA_BASE_URL (optional)

The shim implements an `embed(texts: list[str]) -> list[list[float]]` method.
Each backend call is deferred; for now only Ollama is implemented concretely.
Others raise NotImplementedError placeholders to avoid silent failure.
"""

from __future__ import annotations
from dataclasses import dataclass
import os
from typing import List, Optional, Dict, Type
from loguru import logger  # type: ignore
import httpx

from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # type: ignore
from azure.keyvault.secrets import SecretClient  # type: ignore

DEFAULT_EMBED_MODEL_OLLAMA = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
DEFAULT_EMBED_MODEL_AZURE = os.getenv(
    "EMBEDDING_MODEL",
    os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large"),
)
DEFAULT_EMBED_MODEL_GROK = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
DEFAULT_EMBED_MODEL_GROQ = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

SUPPORTED = {"ollama", "azure", "grok", "groq"}


@dataclass
class EmbeddingProviderConfig:
    provider: str
    model_name: str
    dimensions: Optional[int] = None
    timeout: float = float(os.getenv("EMBED_TIMEOUT", "60"))

    @classmethod
    def from_env(cls) -> "EmbeddingProviderConfig":
        provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
        if provider not in SUPPORTED:
            logger.warning(
                f"EMBEDDING_PROVIDER '{provider}' not recognized; falling back to 'ollama'."
            )
            provider = "ollama"
        model = os.getenv("EMBEDDING_MODEL") or (
            DEFAULT_EMBED_MODEL_OLLAMA
            if provider == "ollama"
            else DEFAULT_EMBED_MODEL_AZURE
            if provider == "azure"
            else DEFAULT_EMBED_MODEL_GROK
            if provider == "grok"
            else DEFAULT_EMBED_MODEL_GROQ
        )
        dims_env = os.getenv("EMBEDDING_DIMENSIONS")
        dimensions = int(dims_env) if dims_env and dims_env.isdigit() else None
        return cls(provider=provider, model_name=model, dimensions=dimensions)


@dataclass
class ResolvedEmbedding:
    provider: "BaseEmbeddingProvider"
    model_name: str
    dimensions: Optional[int]


class BaseEmbeddingProvider:
    """Base class for embedding providers."""

    def __init__(self, cfg: EmbeddingProviderConfig):
        self.cfg = cfg

    def embed(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover
        raise NotImplementedError


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    def embed(self, texts: List[str]) -> List[List[float]]:
        base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.valiantlynx.com/v1")
        url = f"{base_url}/embeddings"
        payload = {"model": self.cfg.model_name, "input": texts}
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            # Ollama returns { embeddings: [ [..], [..] ] } (OpenAI-like), unify shape
            embeddings = data.get("data") or data.get("embeddings")
            if isinstance(embeddings, list):
                if (
                    embeddings
                    and isinstance(embeddings[0], dict)
                    and "embedding" in embeddings[0]
                ):
                    return [e["embedding"] for e in embeddings]
                return embeddings  # already list of vectors
            raise ValueError("Unexpected embedding response shape")
        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
            raise

    def _fetch_azure_api_key_from_keyvault(self) -> Optional[str]:
        keyvault_url = os.getenv("AZURE_OPENAI_KEYVAULT_URL")
        secret_name = os.getenv("AZURE_OPENAI_SECRET_NAME", "azure-openai-api-key")
        if not keyvault_url or not DefaultAzureCredential or not SecretClient:
            return None
        try:  # pragma: no cover (network)
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=keyvault_url, credential=credential)
            secret = client.get_secret(secret_name)
            logger.info(
                f"Retrieved Azure OpenAI key from Key Vault for embeddings secret '{secret_name}'."
            )
            return secret.value
        except Exception as e:  # pragma: no cover
            logger.warning(
                f"Failed to fetch secret '{secret_name}' from Key Vault: {e}"
            )
            return None


class AzureEmbeddingProvider(BaseEmbeddingProvider):
    def _fetch_azure_api_key_from_keyvault(self) -> Optional[str]:
        keyvault_url = os.getenv("AZURE_OPENAI_KEYVAULT_URL")
        secret_name = os.getenv("AZURE_OPENAI_SECRET_NAME", "azure-openai-api-key")
        if not keyvault_url or not DefaultAzureCredential or not SecretClient:
            return None
        try:  # pragma: no cover
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=keyvault_url, credential=credential)
            secret = client.get_secret(secret_name)
            logger.info(
                f"Retrieved Azure OpenAI key from Key Vault for embeddings secret '{secret_name}'."
            )
            return secret.value
        except Exception as e:  # pragma: no cover
            logger.warning(
                f"Failed to fetch secret '{secret_name}' from Key Vault: {e}"
            )
            return None

    def embed(self, texts: List[str]) -> List[List[float]]:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        deployment = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", self.cfg.model_name)
        
        # Check if user wants token-based auth (matches LLM provider)
        use_token_auth = os.getenv("AZURE_USE_TOKEN_AUTH", "true").lower() == "true"
        
        # Build request URL
        url = f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version={api_version}"
        payload = {"input": texts}
        
        # Handle dimensions override
        dims = os.getenv("AZURE_OPENAI_EMBED_DIMENSIONS") or (
            str(self.cfg.dimensions) if self.cfg.dimensions else None
        )
        if dims:
            try:
                payload["dimensions"] = int(dims)
            except ValueError:
                logger.warning("Invalid AZURE_OPENAI_EMBED_DIMENSIONS; ignoring.")
        
        # Set up authentication headers
        headers = {}
        
        if not use_token_auth:
            # Try API key authentication
            api_key = (
                os.getenv("AZURE_OPENAI_API_KEY")
                or self._fetch_azure_api_key_from_keyvault()
            )
            if api_key:
                headers["api-key"] = api_key
            else:
                raise RuntimeError(
                    "Azure embeddings API key unavailable (env or Key Vault)."
                )
        else:
            # Use token-based authentication (az login / managed identity)
            if not get_bearer_token_provider or not DefaultAzureCredential:
                raise RuntimeError(
                    "Azure token authentication unavailable: missing azure-identity library."
                )
            
            credential = DefaultAzureCredential()
            scope = os.getenv("AZURE_COGNITIVE_SERVICE_SCOPE", "https://cognitiveservices.azure.com/.default")
            token_provider = get_bearer_token_provider(credential, scope)
            # Get the token synchronously for httpx request
            token = token_provider()
            headers["Authorization"] = f"Bearer {token}"
        
        try:  # pragma: no cover (network)
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            data_items = data.get("data")
            if not isinstance(data_items, list):
                raise ValueError("Unexpected Azure embedding response shape")
            return [item.get("embedding") for item in data_items]
        except Exception as e:
            logger.error(f"Azure embedding request failed: {e}")
            raise


# Registry mapping provider names to their classes
EMBEDDING_PROVIDERS: Dict[str, Type[BaseEmbeddingProvider]] = {
    "ollama": OllamaEmbeddingProvider,
    "azure": AzureEmbeddingProvider,
}


def get_available_embedding_models(provider: str) -> list[str]:
    """Get list of available embedding models for a given provider.
    
    Args:
        provider: Provider name (ollama, azure, grok, groq)
        
    Returns:
        List of available embedding model names
    """
    provider = provider.lower()
    
    if provider == "ollama":
        # Query Ollama API for available models
        import httpx
        try:
            base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.valiantlynx.com/v1")
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{base_url}/models")
                resp.raise_for_status()
                data = resp.json()
                # Filter for embedding models
                models = [model.get("id", model.get("name", "")) for model in data.get("data", [])]
                # Common embedding model patterns
                embedding_models = [m for m in models if any(kw in m.lower() for kw in ["embed", "nomic", "mxbai"])]
                return sorted(embedding_models) if embedding_models else [DEFAULT_EMBED_MODEL_OLLAMA]
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama embedding models: {e}")
            return [DEFAULT_EMBED_MODEL_OLLAMA, "mxbai-embed-large", "nomic-embed-text"]
    
    elif provider == "azure":
        # Azure embedding models
        return [
            DEFAULT_EMBED_MODEL_AZURE,
            "text-embedding-3-small",
            "text-embedding-ada-002"
        ]
    
    elif provider == "grok":
        # Grok embedding models
        return [DEFAULT_EMBED_MODEL_GROK, "text-embedding-3-small"]
    
    elif provider == "groq":
        # Groq embedding models
        return [DEFAULT_EMBED_MODEL_GROQ, "text-embedding-3-small"]
    
    return [DEFAULT_EMBED_MODEL_OLLAMA]


def get_available_embedding_providers() -> list[str]:
    """Get list of available embedding providers."""
    return list(SUPPORTED)


def get_embedding_provider(
    cfg: EmbeddingProviderConfig | None = None,
) -> ResolvedEmbedding:
    if cfg is None:
        cfg = EmbeddingProviderConfig.from_env()
    provider_cls = EMBEDDING_PROVIDERS.get(cfg.provider)
    if not provider_cls:
        raise RuntimeError(f"Unsupported embedding provider: {cfg.provider}")
    provider = provider_cls(cfg)
    logger.info(
        f"Resolved embedding provider='{cfg.provider}' model='{cfg.model_name}' dims={cfg.dimensions}"
    )
    return ResolvedEmbedding(
        provider=provider, model_name=cfg.model_name, dimensions=cfg.dimensions
    )
