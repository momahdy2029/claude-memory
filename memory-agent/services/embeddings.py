"""Embedding service using Ollama with multi-model support.

Includes health checks, graceful degradation, and model switching capabilities.
"""
import os
import time
import asyncio
from typing import List, Optional, Dict, Any
import ollama
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
HEALTH_CHECK_TIMEOUT = float(os.getenv("OLLAMA_HEALTH_TIMEOUT", "2.0"))
HEALTH_CACHE_TTL = float(os.getenv("OLLAMA_HEALTH_CACHE_TTL", "30.0"))

# Model configurations: model_name -> dimension
MODEL_CONFIGS = {
    "nomic-embed-text": {"dimension": 768, "description": "General purpose, fast"},
    "mxbai-embed-large": {"dimension": 1024, "description": "Higher quality, larger"},
    "all-minilm": {"dimension": 384, "description": "Lightweight, fast"},
    "snowflake-arctic-embed": {"dimension": 1024, "description": "High quality, multilingual"},
    "bge-m3": {"dimension": 1024, "description": "Multilingual, dense retrieval"},
    "default": {"alias_for": "nomic-embed-text"},
}


class EmbeddingService:
    """Service for generating embeddings using Ollama with multi-model support.

    Features:
    - Multiple model support with automatic dimension handling
    - Health check with caching to avoid hammering Ollama
    - Graceful degradation: returns None when Ollama unavailable
    - Timeout handling for unresponsive Ollama instances
    - Model switching without data loss
    """

    def __init__(self, model: Optional[str] = None):
        self.host = OLLAMA_HOST
        self.client = ollama.Client(host=OLLAMA_HOST)

        # Resolve model (handle aliases)
        self.model = self._resolve_model(model or DEFAULT_MODEL)
        self._model_config = self._get_model_config(self.model)

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
        # Default config for unknown models
        return {"dimension": 768, "description": "Unknown model"}

    async def check_health(self, force: bool = False) -> Dict[str, Any]:
        """Check if Ollama is healthy and responsive.

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
                    "error": self._health_error,
                    "degraded_mode": self._degraded_mode
                }

        # Perform health check with timeout
        start_time = time.time()
        try:
            loop = asyncio.get_event_loop()

            def _check():
                # Try to list models to verify Ollama is responding
                models = self.client.list()
                model_names = [m.get('name', m.get('model', '')) for m in models.get('models', [])]
                # Check if our model is available
                model_loaded = any(self.model in name for name in model_names)
                return models, model_loaded, model_names

            # Run with timeout
            models, model_loaded, model_names = await asyncio.wait_for(
                loop.run_in_executor(None, _check),
                timeout=HEALTH_CHECK_TIMEOUT
            )

            latency_ms = (time.time() - start_time) * 1000

            self._health_status = True
            self._health_last_check = now
            self._health_error = None
            self._available_models = model_names
            self._models_last_check = now

            # Exit degraded mode if we were in it
            if self._degraded_mode:
                self._degraded_mode = False
                self._degraded_since = None

            return {
                "healthy": True,
                "cached": False,
                "model": self.model,
                "model_loaded": model_loaded,
                "host": self.host,
                "latency_ms": round(latency_ms, 2),
                "error": None,
                "degraded_mode": False,
                "available_models": model_names
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
                "error": self._health_error,
                "degraded_mode": True
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
                "error": self._health_error,
                "degraded_mode": True
            }

    def _enter_degraded_mode(self):
        """Enter degraded mode when Ollama is unavailable."""
        if not self._degraded_mode:
            self._degraded_mode = True
            self._degraded_since = time.time()

    def is_degraded(self) -> bool:
        """Check if service is in degraded mode."""
        return self._degraded_mode

    def get_degraded_duration(self) -> Optional[float]:
        """Get how long service has been in degraded mode."""
        if self._degraded_since:
            return time.time() - self._degraded_since
        return None

    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        fallback_on_error: bool = True
    ) -> Optional[List[float]]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            model: Optional model override (uses default if not specified)
            fallback_on_error: If True, return None instead of raising on error

        Returns:
            List of floats (embedding) or None if Ollama unavailable and fallback enabled
        """
        use_model = self._resolve_model(model) if model else self.model

        # Quick check if we're in degraded mode
        if self._degraded_mode:
            # Check if we should retry (every 30s)
            if time.time() - self._health_last_check >= self._health_cache_ttl:
                health = await self.check_health(force=True)
                if not health["healthy"]:
                    if fallback_on_error:
                        return None
                    raise ConnectionError(f"Ollama unavailable: {health['error']}")
            elif fallback_on_error:
                return None
            else:
                raise ConnectionError(f"Ollama unavailable (degraded mode): {self._health_error}")

        try:
            loop = asyncio.get_event_loop()

            def _embed():
                response = self.client.embeddings(model=use_model, prompt=text)
                return response["embedding"]

            # Run with timeout
            embedding = await asyncio.wait_for(
                loop.run_in_executor(None, _embed),
                timeout=30.0  # 30s timeout for embedding generation
            )
            return embedding

        except asyncio.TimeoutError:
            self._enter_degraded_mode()
            self._health_error = "Embedding generation timed out"
            if fallback_on_error:
                return None
            raise

        except Exception as e:
            # Check if it's a connection error
            error_str = str(e).lower()
            if "connection" in error_str or "refused" in error_str or "timeout" in error_str:
                self._enter_degraded_mode()
                self._health_error = str(e)

            if fallback_on_error:
                return None
            raise

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        fallback_on_error: bool = True
    ) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: Optional model override
            fallback_on_error: If True, include None for failed embeddings

        Returns:
            List of embeddings (or None for failed ones if fallback enabled)
        """
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text, model, fallback_on_error)
            embeddings.append(embedding)
        return embeddings

    def get_dimension(self, model: Optional[str] = None) -> int:
        """Return the embedding dimension for a model."""
        use_model = self._resolve_model(model) if model else self.model
        config = self._get_model_config(use_model)
        return config.get("dimension", 768)

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
                continue  # Skip aliases
            models.append({
                "name": name,
                "dimension": config.get("dimension", 768),
                "description": config.get("description", ""),
                "is_current": name == self.model,
                "available_in_ollama": (
                    any(name in m for m in (self._available_models or []))
                    if self._available_models else None
                )
            })
        return models

    async def get_ollama_models(self) -> List[str]:
        """Get list of models currently available in Ollama."""
        if self._available_models and (time.time() - self._models_last_check) < 60:
            return self._available_models

        try:
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(None, self.client.list)
            model_names = [m.get('name', m.get('model', '')) for m in models.get('models', [])]
            self._available_models = model_names
            self._models_last_check = time.time()
            return model_names
        except:
            return self._available_models or []

    def get_status(self) -> Dict[str, Any]:
        """Get current service status."""
        return {
            "model": self.model,
            "dimension": self.get_dimension(),
            "host": self.host,
            "degraded_mode": self._degraded_mode,
            "degraded_since": self._degraded_since,
            "degraded_duration_seconds": self.get_degraded_duration(),
            "last_health_check": self._health_last_check,
            "last_health_status": self._health_status,
            "last_health_error": self._health_error,
            "available_models_in_ollama": self._available_models
        }


# Global registry of embedding services per model
_embedding_services: Dict[str, EmbeddingService] = {}


def get_embedding_service(model: Optional[str] = None) -> EmbeddingService:
    """Get an embedding service for a specific model.

    Uses a shared instance per model to maintain health check state.
    """
    model_key = model or DEFAULT_MODEL

    if model_key not in _embedding_services:
        _embedding_services[model_key] = EmbeddingService(model_key)

    return _embedding_services[model_key]
