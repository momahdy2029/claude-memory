"""Embedding service with pluggable provider support.

Supports multiple embedding backends (Ollama, sentence-transformers) via
a provider abstraction layer. Includes health checks, graceful degradation,
and model switching capabilities.
"""
import os
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Conditional ollama import
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

logger = logging.getLogger(__name__)

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")  # Ollama default; sentence-transformers uses config.py
HEALTH_CHECK_TIMEOUT = float(os.getenv("OLLAMA_HEALTH_TIMEOUT", "5.0"))
HEALTH_CACHE_TTL = float(os.getenv("OLLAMA_HEALTH_CACHE_TTL", "30.0"))


class EmbeddingError(Enum):
    """Distinguishable error codes for embedding failures."""
    NONE = "none"                    # No error
    EMPTY_TEXT = "empty_text"        # Input text was empty or whitespace
    OLLAMA_OFFLINE = "ollama_offline"  # Ollama service not reachable
    MODEL_NOT_LOADED = "model_not_loaded"  # Model not available in Ollama
    TIMEOUT = "timeout"              # Embedding generation timed out
    DEGRADED_MODE = "degraded_mode"  # Service in degraded mode, not retrying yet
    UNKNOWN = "unknown"              # Unexpected error


@dataclass
class EmbeddingResult:
    """Result from embedding generation with error context.

    Allows callers to distinguish failure modes and take appropriate action:
    - EMPTY_TEXT: Skip embedding, store memory without it
    - OLLAMA_OFFLINE: Queue for later re-embedding
    - TIMEOUT: Retry with smaller text or different model
    - DEGRADED_MODE: Wait for auto-recovery
    """
    embedding: Optional[List[float]]
    error: EmbeddingError = EmbeddingError.NONE
    error_message: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.embedding is not None and self.error == EmbeddingError.NONE


# Model configurations: model_name -> dimension
MODEL_CONFIGS = {
    "nomic-embed-text": {"dimension": 768, "description": "General purpose, fast"},
    "mxbai-embed-large": {"dimension": 1024, "description": "Higher quality, larger"},
    "all-minilm": {"dimension": 384, "description": "Lightweight, fast"},
    "snowflake-arctic-embed": {"dimension": 1024, "description": "High quality, multilingual"},
    "bge-m3": {"dimension": 1024, "description": "Multilingual, dense retrieval"},
    "gte-large-en-v1.5": {"alias_for": "Alibaba-NLP/gte-large-en-v1.5"},
    "Alibaba-NLP/gte-large-en-v1.5": {"dimension": 1024, "description": "High quality, best STS scores"},
    "all-MiniLM-L6-v2": {"dimension": 384, "description": "Lightweight, fast, sentence-transformers"},
    "BAAI/bge-base-en-v1.5": {"dimension": 768, "description": "Good balance, sentence-transformers"},
    "default": {"alias_for": "nomic-embed-text"},
}


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text. Runs synchronously."""
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts. Runs synchronously."""
        ...

    @abstractmethod
    def check_health(self) -> dict:
        """Check provider health. Returns dict with 'healthy', 'error' keys."""
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name."""
        ...


class OllamaProvider(EmbeddingProvider):
    """Embedding provider backed by a local Ollama instance."""

    def __init__(self, host: str = OLLAMA_HOST, model: str = DEFAULT_MODEL):
        if not HAS_OLLAMA:
            raise RuntimeError(
                "ollama package not installed. Run: pip install ollama"
            )
        self.host = host
        self.model = model
        self.client = ollama.Client(host=host)

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Ollama has no native batch endpoint; call embed() sequentially
        return [self.embed(t) for t in texts]

    def check_health(self) -> dict:
        try:
            models = self.client.list()
            model_names = [
                m.get("name", m.get("model", ""))
                for m in models.get("models", [])
            ]
            model_loaded = any(self.model in name for name in model_names)
            return {
                "healthy": True,
                "model": self.model,
                "model_loaded": model_loaded,
                "provider": "ollama",
                "host": self.host,
                "error": None,
                "available_models": model_names,
            }
        except Exception as e:
            return {
                "healthy": False,
                "model": self.model,
                "provider": "ollama",
                "host": self.host,
                "error": str(e),
            }

    def get_dimension(self) -> int:
        config = MODEL_CONFIGS.get(self.model, {})
        if "alias_for" in config:
            config = MODEL_CONFIGS.get(config["alias_for"], {})
        return config.get("dimension", 768)

    def get_model_name(self) -> str:
        return self.model


class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using the sentence-transformers library."""

    def __init__(self, model: str = "Alibaba-NLP/gte-large-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers package not installed. "
                "Run: pip install sentence-transformers"
            )

        self.model_name = model
        self._model = SentenceTransformer(model, trust_remote_code=True)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, batch_size=32
        )
        return embeddings.tolist()

    def check_health(self) -> dict:
        return {
            "healthy": True,
            "model": self.model_name,
            "provider": "sentence-transformers",
            "error": None,
        }

    def get_dimension(self) -> int:
        return self._dimension

    def get_model_name(self) -> str:
        return self.model_name


# ---------------------------------------------------------------------------
# EmbeddingService  (public API unchanged)
# ---------------------------------------------------------------------------

class EmbeddingService:
    """Service for generating embeddings with pluggable provider backends.

    Features:
    - Multiple model support with automatic dimension handling
    - Health check with caching to avoid hammering the provider
    - Graceful degradation: returns None when provider unavailable
    - Timeout handling for unresponsive backends
    - Model switching without data loss
    """

    def __init__(
        self,
        provider_type: str = "sentence-transformers",
        model: Optional[str] = None,
    ):
        self.provider_type = provider_type

        # Determine model
        if model:
            self.model = self._resolve_model(model)
        elif provider_type == "sentence-transformers":
            self.model = "Alibaba-NLP/gte-large-en-v1.5"
        else:
            self.model = DEFAULT_MODEL  # nomic-embed-text

        # Guard: Ollama-only model names don't exist on HuggingFace
        OLLAMA_ONLY_MODELS = {"nomic-embed-text", "mxbai-embed-large", "all-minilm",
                              "snowflake-arctic-embed", "bge-m3"}
        if provider_type == "sentence-transformers" and self.model in OLLAMA_ONLY_MODELS:
            logger.warning(
                f"Model '{self.model}' is an Ollama-only model but provider is "
                f"sentence-transformers. Falling back to Alibaba-NLP/gte-large-en-v1.5. "
                f"Update EMBEDDING_MODEL in your .env to fix this."
            )
            self.model = "Alibaba-NLP/gte-large-en-v1.5"

        self._model_config = self._get_model_config(self.model)

        # Create provider
        if provider_type == "ollama":
            self._provider = OllamaProvider(host=OLLAMA_HOST, model=self.model)
            self.host = OLLAMA_HOST
        elif provider_type == "sentence-transformers":
            self._provider = SentenceTransformerProvider(model=self.model)
            self.host = "local"
        else:
            raise ValueError(
                f"Unknown provider: {provider_type}. "
                "Use 'ollama' or 'sentence-transformers'"
            )

        # Health check caching
        self._health_status: Optional[bool] = None
        self._health_last_check: float = 0
        self._health_cache_ttl = HEALTH_CACHE_TTL
        self._health_error: Optional[str] = None

        # Degraded mode tracking
        self._degraded_mode = False
        self._degraded_since: Optional[float] = None

        # Available models cache
        self._available_models: Optional[List[str]] = None
        self._models_last_check: float = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model(self, model: str) -> str:
        """Resolve model aliases to actual model names."""
        config = MODEL_CONFIGS.get(model, {})
        if "alias_for" in config:
            return config["alias_for"]
        return model

    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get configuration for a model."""
        if model in MODEL_CONFIGS:
            return MODEL_CONFIGS[model]
        return {"dimension": 768, "description": "Unknown model"}

    def _enter_degraded_mode(self):
        """Enter degraded mode when provider is unavailable."""
        if not self._degraded_mode:
            self._degraded_mode = True
            self._degraded_since = time.time()

    def _is_local_provider(self) -> bool:
        """Return True if the provider runs locally with no remote dependency."""
        return self.provider_type == "sentence-transformers"

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def check_health(self, force: bool = False) -> Dict[str, Any]:
        """Check if the embedding provider is healthy and responsive.

        Args:
            force: If True, bypass cache and check immediately

        Returns:
            Dict with 'healthy', 'model_loaded', 'latency_ms', 'error' keys
        """
        now = time.time()

        # Return cached result if within TTL
        if not force and self._health_status is not None:
            if (now - self._health_last_check) < self._health_cache_ttl:
                return {
                    "healthy": self._health_status,
                    "cached": True,
                    "model": self.model,
                    "host": self.host,
                    "provider": self.provider_type,
                    "error": self._health_error,
                    "degraded_mode": self._degraded_mode,
                }

        start_time = time.time()
        try:
            loop = asyncio.get_running_loop()

            health_result = await asyncio.wait_for(
                loop.run_in_executor(None, self._provider.check_health),
                timeout=HEALTH_CHECK_TIMEOUT,
            )

            latency_ms = (time.time() - start_time) * 1000

            self._health_status = health_result.get("healthy", False)
            self._health_last_check = now
            self._health_error = health_result.get("error")
            self._available_models = health_result.get("available_models")
            if self._available_models is not None:
                self._models_last_check = now

            # Exit degraded mode on success
            if self._health_status and self._degraded_mode:
                self._degraded_mode = False
                self._degraded_since = None

            return {
                "healthy": self._health_status,
                "cached": False,
                "model": self.model,
                "model_loaded": health_result.get("model_loaded", True),
                "host": self.host,
                "provider": self.provider_type,
                "latency_ms": round(latency_ms, 2),
                "error": self._health_error,
                "degraded_mode": False if self._health_status else self._degraded_mode,
                "available_models": self._available_models,
            }

        except asyncio.TimeoutError:
            self._health_status = False
            self._health_last_check = now
            self._health_error = f"Timeout after {HEALTH_CHECK_TIMEOUT}s"
            self._enter_degraded_mode()

            return {
                "healthy": False,
                "cached": False,
                "model": self.model,
                "host": self.host,
                "provider": self.provider_type,
                "error": self._health_error,
                "degraded_mode": True,
            }

        except Exception as e:
            self._health_status = False
            self._health_last_check = now
            self._health_error = str(e)
            self._enter_degraded_mode()

            return {
                "healthy": False,
                "cached": False,
                "model": self.model,
                "host": self.host,
                "provider": self.provider_type,
                "error": self._health_error,
                "degraded_mode": True,
            }

    def is_degraded(self) -> bool:
        """Check if service is in degraded mode."""
        return self._degraded_mode

    def get_degraded_duration(self) -> Optional[float]:
        """Get how long service has been in degraded mode."""
        if self._degraded_since:
            return time.time() - self._degraded_since
        return None

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        fallback_on_error: bool = True,
    ) -> Optional[List[float]]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            model: Optional model override (uses default if not specified)
            fallback_on_error: If True, return None instead of raising on error

        Returns:
            List of floats (embedding) or None if provider unavailable and fallback enabled
        """
        # For local providers, skip degraded-mode gating
        if not self._is_local_provider() and self._degraded_mode:
            if time.time() - self._health_last_check >= self._health_cache_ttl:
                health = await self.check_health(force=True)
                if not health["healthy"]:
                    if fallback_on_error:
                        return None
                    raise ConnectionError(
                        f"Provider unavailable: {health['error']}"
                    )
            elif fallback_on_error:
                return None
            else:
                raise ConnectionError(
                    f"Provider unavailable (degraded mode): {self._health_error}"
                )

        try:
            loop = asyncio.get_running_loop()

            def _embed():
                return self._provider.embed(text)

            embedding = await asyncio.wait_for(
                loop.run_in_executor(None, _embed),
                timeout=30.0,
            )
            return embedding

        except asyncio.TimeoutError:
            self._enter_degraded_mode()
            self._health_error = "Embedding generation timed out"
            if fallback_on_error:
                return None
            raise

        except Exception as e:
            error_str = str(e).lower()
            if "connection" in error_str or "refused" in error_str or "timeout" in error_str:
                self._enter_degraded_mode()
                self._health_error = str(e)

            if fallback_on_error:
                return None
            raise

    async def generate_embedding_with_status(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> EmbeddingResult:
        """Generate embedding with detailed error status.

        Unlike generate_embedding() which returns None for all failures,
        this method returns an EmbeddingResult with a specific error code
        so callers can distinguish:
        - Empty input text
        - Ollama offline (connection refused)
        - Model not loaded
        - Generation timeout
        - Degraded mode (waiting for auto-recovery)

        Args:
            text: Text to embed
            model: Optional model override

        Returns:
            EmbeddingResult with embedding and error details
        """
        # Validate input
        if not text or not text.strip():
            return EmbeddingResult(
                embedding=None,
                error=EmbeddingError.EMPTY_TEXT,
                error_message="Input text is empty or whitespace-only",
            )

        # For local providers, skip degraded-mode gating
        if not self._is_local_provider() and self._degraded_mode:
            if time.time() - self._health_last_check >= self._health_cache_ttl:
                health = await self.check_health(force=True)
                if not health["healthy"]:
                    return EmbeddingResult(
                        embedding=None,
                        error=EmbeddingError.OLLAMA_OFFLINE,
                        error_message=f"Provider unavailable: {health.get('error', 'unknown')}",
                    )
            else:
                return EmbeddingResult(
                    embedding=None,
                    error=EmbeddingError.DEGRADED_MODE,
                    error_message=(
                        f"Service in degraded mode since {self._degraded_since:.0f}. "
                        f"Next retry in {self._health_cache_ttl - (time.time() - self._health_last_check):.0f}s"
                    ),
                )

        try:
            loop = asyncio.get_running_loop()

            def _embed():
                return self._provider.embed(text)

            embedding = await asyncio.wait_for(
                loop.run_in_executor(None, _embed),
                timeout=30.0,
            )

            # Exit degraded mode on success
            if self._degraded_mode:
                self._degraded_mode = False
                self._degraded_since = None

            return EmbeddingResult(embedding=embedding)

        except asyncio.TimeoutError:
            self._enter_degraded_mode()
            self._health_error = "Embedding generation timed out"
            return EmbeddingResult(
                embedding=None,
                error=EmbeddingError.TIMEOUT,
                error_message=f"Embedding generation timed out after 30s for model {self.model}",
            )

        except Exception as e:
            error_str = str(e).lower()

            if "connection" in error_str or "refused" in error_str:
                self._enter_degraded_mode()
                self._health_error = str(e)
                return EmbeddingResult(
                    embedding=None,
                    error=EmbeddingError.OLLAMA_OFFLINE,
                    error_message=f"Provider not reachable: {e}",
                )

            if "model" in error_str and (
                "not found" in error_str or "not exist" in error_str
            ):
                return EmbeddingResult(
                    embedding=None,
                    error=EmbeddingError.MODEL_NOT_LOADED,
                    error_message=f"Model '{self.model}' not available: {e}",
                )

            if "timeout" in error_str:
                self._enter_degraded_mode()
                self._health_error = str(e)
                return EmbeddingResult(
                    embedding=None,
                    error=EmbeddingError.TIMEOUT,
                    error_message=f"Timeout: {e}",
                )

            logger.warning(f"Unexpected embedding error: {e}")
            return EmbeddingResult(
                embedding=None,
                error=EmbeddingError.UNKNOWN,
                error_message=str(e),
            )

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        fallback_on_error: bool = True,
        batch_size: int = 10,
    ) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts.

        For providers with native batch support (sentence-transformers), uses
        the provider's batch method directly. Otherwise falls back to
        concurrent individual requests.

        Args:
            texts: List of texts to embed
            model: Optional model override
            fallback_on_error: If True, include None for failed embeddings
            batch_size: Number of concurrent embedding requests per batch

        Returns:
            List of embeddings (or None for failed ones if fallback enabled)
        """
        if not texts:
            return []

        # sentence-transformers has efficient native batching
        if self.provider_type == "sentence-transformers":
            try:
                loop = asyncio.get_running_loop()

                def _batch_embed():
                    return self._provider.embed_batch(texts)

                results = await asyncio.wait_for(
                    loop.run_in_executor(None, _batch_embed),
                    timeout=max(30.0, len(texts) * 2.0),
                )
                return results
            except Exception as e:
                if fallback_on_error:
                    logger.warning(f"Batch embedding failed: {e}")
                    return [None] * len(texts)
                raise

        # For other providers, use concurrent individual requests
        results: List[Optional[List[float]]] = [None] * len(texts)

        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start : batch_start + batch_size]
            batch_results = await asyncio.gather(
                *[
                    self.generate_embedding(text, model, fallback_on_error)
                    for text in batch_texts
                ],
                return_exceptions=True,
            )

            for i, result in enumerate(batch_results):
                idx = batch_start + i
                if isinstance(result, Exception):
                    if fallback_on_error:
                        results[idx] = None
                    else:
                        raise result
                else:
                    results[idx] = result

        return results

    # ------------------------------------------------------------------
    # Model / status helpers
    # ------------------------------------------------------------------

    def get_dimension(self, model: Optional[str] = None) -> int:
        """Return the embedding dimension for a model."""
        if model:
            use_model = self._resolve_model(model)
            config = self._get_model_config(use_model)
            return config.get("dimension", 768)
        return self._provider.get_dimension()

    def get_current_model(self) -> str:
        """Get the current default model."""
        return self.model

    def set_model(self, model: str):
        """Set the default model.

        Note: This only changes the default for new embeddings.
        Existing embeddings are not affected.
        """
        self.model = self._resolve_model(model)
        self._model_config = self._get_model_config(self.model)

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available embedding models with their configurations."""
        models = []
        for name, config in MODEL_CONFIGS.items():
            if "alias_for" in config:
                continue
            models.append({
                "name": name,
                "dimension": config.get("dimension", 768),
                "description": config.get("description", ""),
                "is_current": name == self.model,
                "available_in_ollama": (
                    any(name in m for m in (self._available_models or []))
                    if self._available_models
                    else None
                ),
            })
        return models

    async def get_ollama_models(self) -> List[str]:
        """Get list of models currently available in Ollama."""
        if self.provider_type != "ollama":
            return []

        if self._available_models and (time.time() - self._models_last_check) < 60:
            return self._available_models

        try:
            loop = asyncio.get_running_loop()
            provider: OllamaProvider = self._provider  # type: ignore[assignment]
            models = await loop.run_in_executor(None, provider.client.list)
            model_names = [
                m.get("name", m.get("model", ""))
                for m in models.get("models", [])
            ]
            self._available_models = model_names
            self._models_last_check = time.time()
            return model_names
        except Exception:
            return self._available_models or []

    def get_status(self) -> Dict[str, Any]:
        """Get current service status."""
        return {
            "model": self.model,
            "dimension": self.get_dimension(),
            "host": self.host,
            "provider": self.provider_type,
            "degraded_mode": self._degraded_mode,
            "degraded_since": self._degraded_since,
            "degraded_duration_seconds": self.get_degraded_duration(),
            "last_health_check": self._health_last_check,
            "last_health_status": self._health_status,
            "last_health_error": self._health_error,
            "available_models_in_ollama": self._available_models,
        }


# Global registry of embedding services per provider:model
_embedding_services: Dict[str, EmbeddingService] = {}


def get_embedding_service(
    model: Optional[str] = None,
    provider_type: Optional[str] = None,
) -> EmbeddingService:
    """Get an embedding service for a specific provider and model.

    Uses a shared instance per provider:model to maintain health check state.
    """
    provider = provider_type or os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
    model_key = model or DEFAULT_MODEL
    cache_key = f"{provider}:{model_key}"

    if cache_key not in _embedding_services:
        _embedding_services[cache_key] = EmbeddingService(
            provider_type=provider, model=model_key
        )

    return _embedding_services[cache_key]
