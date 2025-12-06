"""LLM provider factory using pydantic-ai.

Adds a DRY abstraction over different backends:
- Ollama (default)
- Azure OpenAI
- Grok (x.ai) via OpenAI-compatible API
- Groq (distinct from Grok, optional)

Environment variables control selection:
  LLM_PROVIDER=ollama|azure|grok|groq (default: ollama)
  LLM_MODEL=<model name override>

Provider-specific env vars:
Azure:
  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
  AZURE_OPENAI_API_KEY=...
  AZURE_OPENAI_DEPLOYMENT=<chat deployment>
Grok (x.ai):
  GROK_API_KEY=...
Groq:
  GROQ_API_KEY=...
Ollama:
  OLLAMA_BASE_URL=... (optional)

The factory returns an OpenAIChatModel instance (even for Ollama etc.) because Ollama/Grok/Groq expose OpenAI-compatible endpoints.
"""

from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Optional, Callable, Dict, Type
from loguru import logger  # type: ignore
from pydantic_ai.providers.ollama import OllamaProvider as PydanticOllamaProvider
from azure.identity import DefaultAzureCredential  # type: ignore
from azure.keyvault.secrets import SecretClient  # type: ignore
from pydantic_ai.providers.openai import OpenAIProvider  # Azure + OpenAI-compatible
from pydantic_ai.providers.azure import AzureProvider
from azure.identity import get_bearer_token_provider  # type: ignore
from openai import AsyncAzureOpenAI  # type: ignore
from dotenv import load_dotenv
load_dotenv()  # load from .env if present

DEFAULT_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "https://ollama.valiantlynx.com/v1")
DEFAULT_OLLAMA_MODEL = "gpt-oss:latest"
DEFAULT_GROK_MODEL = "grok-2-latest"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-2")

SUPPORTED_PROVIDERS = {"ollama", "azure", "grok", "groq"}

@dataclass
class ResolvedProvider:
    """Container for a resolved raw provider and its chosen model name."""

    provider: object
    model_name: str


class BaseLLMProvider:
    """Abstract provider; subclasses implement create(cfg)."""

    @classmethod
    def create(cls, cfg: "LLMProviderConfig") -> ResolvedProvider:  # pragma: no cover
        raise NotImplementedError


@dataclass
class LLMProviderConfig:
    provider: str
    model_name: str
    temperature: float = 0.1
    timeout: float = float(os.getenv("LLM_TIMEOUT", "604800"))  # one week fallback
    vision: bool = True  # enable vision by default
    max_tokens: int = 0
    context_window: int = 0

    @classmethod
    def from_env(cls) -> "LLMProviderConfig":
        provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        if provider not in SUPPORTED_PROVIDERS:
            logger.warning(
                f"LLM_PROVIDER '{provider}' not recognized; falling back to 'ollama'."
            )
            provider = "ollama"
        # Choose default model per provider if not explicitly set
        env_model = os.getenv("LLM_MODEL")
        if env_model:
            model = env_model
        else:
            if provider == "ollama":
                model = DEFAULT_OLLAMA_MODEL
            elif provider == "azure":
                model = DEFAULT_AZURE_DEPLOYMENT  # Azure uses deployment name
            elif provider == "grok":
                model = DEFAULT_GROK_MODEL
            elif provider == "groq":
                model = DEFAULT_GROQ_MODEL
            else:
                model = DEFAULT_OLLAMA_MODEL
        # max_tokens/context_window defaults per provider, overrideable via env
        default_limits = {
            "ollama": (131072, 131072),
            "azure": (8192, 8192),
            "grok": (8192, 8192),
            "groq": (32768, 32768),
        }
        mt_env = os.getenv("LLM_MAX_TOKENS")
        cw_env = os.getenv("LLM_CONTEXT_WINDOW")
        mt_default, cw_default = default_limits.get(provider, (8192, 8192))
        max_tokens = int(mt_env) if mt_env and mt_env.isdigit() else mt_default
        context_window = int(cw_env) if cw_env and cw_env.isdigit() else cw_default
        return cls(
            provider=provider,
            model_name=model,
            max_tokens=max_tokens,
            context_window=context_window,
        )


class OllamaProvider(BaseLLMProvider):
    @classmethod
    def create(cls, cfg: "LLMProviderConfig") -> ResolvedProvider:
        provider = PydanticOllamaProvider(base_url=DEFAULT_OLLAMA_BASE)
        return ResolvedProvider(provider=provider, model_name=cfg.model_name)


def _fetch_azure_api_key_from_keyvault() -> Optional[str]:
    keyvault_url = os.getenv("AZURE_OPENAI_KEYVAULT_URL", "https://kv-dev-localnews-ai.vault.azure.net/")
    secret_name = os.getenv("AZURE_OPENAI_SECRET_NAME", "azure-openai-api-key")
    if not keyvault_url or not DefaultAzureCredential or not SecretClient:
        return None
    try:  # pragma: no cover (network)
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=keyvault_url, credential=credential)
        secret = client.get_secret(secret_name)
        logger.info(
            f"Retrieved Azure OpenAI key from Key Vault secret '{secret_name}'."
        )
        return secret.value
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to fetch secret '{secret_name}' from Key Vault: {e}")
        return None


class AzureProvider(BaseLLMProvider):
    @classmethod
    def create(cls, cfg: "LLMProviderConfig") -> ResolvedProvider:
        if not AzureProvider:
            raise RuntimeError(
                "AzureProvider not available; upgrade pydantic-ai for Azure support."
            )
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://openai-new-version.openai.azure.com/")
        if not endpoint:
            raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT for Azure backend.")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        
        # Check if user wants token-based auth (az login) - matches setup.py pattern
        use_token_auth = os.getenv("AZURE_USE_TOKEN_AUTH", "true").lower() == "true"
        
        if not use_token_auth:
            # Try API key authentication
            api_key = os.getenv("AZURE_OPENAI_API_KEY") or _fetch_azure_api_key_from_keyvault()
            if api_key:
                provider = AzureProvider(
                    azure_endpoint=endpoint,
                    api_version=api_version,
                    api_key=api_key,
                )
                return ResolvedProvider(provider=provider, model_name=cfg.model_name)
        
        # Default: use token-based authentication (az login / managed identity) via custom client
        if not get_bearer_token_provider or not DefaultAzureCredential or not AsyncAzureOpenAI:
            raise RuntimeError("Azure token authentication unavailable: missing azure-identity or openai library.")
        
        credential = DefaultAzureCredential()
        scope = os.getenv("AZURE_COGNITIVE_SERVICE_SCOPE", "https://cognitiveservices.azure.com/.default")
        token_provider = get_bearer_token_provider(credential, scope)
        
        # Use AsyncAzureOpenAI client with token provider, then wrap in OpenAIProvider
        # This matches the pydantic-ai docs pattern for token-based Azure auth
        azure_client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            azure_ad_token_provider=token_provider,
        )
        provider = OpenAIProvider(openai_client=azure_client)
        
        return ResolvedProvider(provider=provider, model_name=cfg.model_name)


class GrokProvider(BaseLLMProvider):
    @classmethod
    def create(cls, cfg: "LLMProviderConfig") -> ResolvedProvider:
        if not OpenAIProvider:
            raise RuntimeError(
                "OpenAIProvider not available; upgrade pydantic-ai for Grok support."
            )
        api_key = os.getenv("GROK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GROK_API_KEY for Grok backend.")
        provider = OpenAIProvider(api_key=api_key, base_url="https://api.x.ai/v1")
        return ResolvedProvider(provider=provider, model_name=cfg.model_name)


class GroqProvider(BaseLLMProvider):
    @classmethod
    def create(cls, cfg: "LLMProviderConfig") -> ResolvedProvider:
        if not OpenAIProvider:
            raise RuntimeError(
                "OpenAIProvider not available; upgrade pydantic-ai for Groq support."
            )
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY for Groq backend.")
        provider = OpenAIProvider(
            api_key=api_key, base_url="https://api.groq.com/openai/v1"
        )
        return ResolvedProvider(provider=provider, model_name=cfg.model_name)


# Registry mapping provider names to their classes
LLM_PROVIDERS: Dict[str, Type[BaseLLMProvider]] = {
    "ollama": OllamaProvider,
    "azure": AzureProvider,
    "grok": GrokProvider,
    "groq": GroqProvider,
}


def get_available_llm_models(provider: str) -> list[str]:
    """Get list of available models for a given provider.
    
    Args:
        provider: Provider name (ollama, azure, grok, groq)
        
    Returns:
        List of available model names
    """
    provider = provider.lower()
    
    if provider == "ollama":
        # Query Ollama API for available models
        import httpx
        try:
            base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.valiantlynx.com/v1")
            # Use OpenAI-compatible list models endpoint
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{base_url}/models")
                resp.raise_for_status()
                data = resp.json()
                # Extract model names from response
                models = [model.get("id", model.get("name", "")) for model in data.get("data", [])]
                return sorted(models) if models else [DEFAULT_OLLAMA_MODEL]
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama models: {e}")
            return [DEFAULT_OLLAMA_MODEL, "llama3.2-vision:latest", "qwen2.5-coder:latest", "deepseek-r1:latest"]
    
    elif provider == "azure":
        # Azure uses deployment names - return common ones
        return [
            DEFAULT_AZURE_DEPLOYMENT,
            "gpt-4o",
            "gpt-4",
            "gpt-35-turbo"
        ]
    
    elif provider == "grok":
        # Grok models from x.ai
        return [
            DEFAULT_GROK_MODEL,
            "grok-2-mini",
            "grok-beta"
        ]
    
    elif provider == "groq":
        # Groq models
        return [
            DEFAULT_GROQ_MODEL,
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
    
    return [DEFAULT_OLLAMA_MODEL]


def get_available_llm_providers() -> list[str]:
    """Get list of available LLM providers."""
    return list(SUPPORTED_PROVIDERS)


def get_llm_provider(cfg: Optional[LLMProviderConfig] = None) -> ResolvedProvider:
    """Return raw provider object + model name, without constructing a chat model.

    This lets downstream code decide how to assemble models (e.g., different settings per Agent).
    """
    if cfg is None:
        cfg = LLMProviderConfig.from_env()
    provider_cls = LLM_PROVIDERS.get(cfg.provider)
    if not provider_cls:
        raise RuntimeError(f"Unsupported LLM provider: {cfg.provider}")
    resolved = provider_cls.create(cfg)
    logger.info(
        f"Resolved provider='{cfg.provider}' model='{resolved.model_name}' tokens={cfg.max_tokens} ctx={cfg.context_window}"
    )
    return resolved
