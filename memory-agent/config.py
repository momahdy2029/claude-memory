"""
Centralized configuration for Claude Memory Agent.

All configuration is loaded from environment variables with sensible defaults.
Use .env file for local overrides (not committed to git).

Usage:
    from config import config
    print(config.PORT)
    print(config.MEMORY_AGENT_URL)
"""
import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from agent directory
AGENT_DIR = Path(__file__).parent.resolve()
load_dotenv(AGENT_DIR / ".env")


class Config:
    """Configuration singleton with environment variable loading."""

    def __init__(self):
        # Paths (auto-detected from script location)
        self.AGENT_DIR = AGENT_DIR
        self.DATABASE_PATH = Path(os.getenv(
            "DATABASE_PATH",
            str(AGENT_DIR / "memories.db")
        ))
        self.INDEX_DIR = Path(os.getenv(
            "INDEX_DIR",
            str(AGENT_DIR / "indexes")
        ))
        self.LOG_FILE = AGENT_DIR / "memory-agent.log"
        self.LOCK_FILE = AGENT_DIR / "memory-agent.lock"
        self.PID_FILE = AGENT_DIR / "memory-agent.pid"

        # Server configuration
        self.HOST = os.getenv("HOST", "127.0.0.1")
        self.PORT = int(os.getenv("PORT", "8102"))
        self.MEMORY_AGENT_URL = os.getenv(
            "MEMORY_AGENT_URL",
            f"http://localhost:{self.PORT}"
        )

        # Embeddings
        self.EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
        self.OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Alibaba-NLP/gte-large-en-v1.5")
        self.EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
        self.OLLAMA_HEALTH_TIMEOUT = float(os.getenv("OLLAMA_HEALTH_TIMEOUT", "2.0"))
        self.OLLAMA_HEALTH_CACHE_TTL = float(os.getenv("OLLAMA_HEALTH_CACHE_TTL", "30.0"))

        # Database
        self.USE_VECTOR_INDEX = os.getenv("USE_VECTOR_INDEX", "true").lower() == "true"
        self.DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
        self.DB_TIMEOUT = float(os.getenv("DB_TIMEOUT", "30.0"))
        self.DB_MAX_RETRIES = int(os.getenv("DB_MAX_RETRIES", "3"))
        self.DB_RETRY_BASE_DELAY = float(os.getenv("DB_RETRY_BASE_DELAY", "0.1"))

        # Authentication
        self.AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
        self.AUTH_KEY_FILE = Path(os.getenv(
            "AUTH_KEY_FILE",
            str(Path.home() / ".claude" / "memory-agent-keys.json")
        ))
        self.AUTH_RATE_LIMIT = int(os.getenv("AUTH_RATE_LIMIT", "100"))
        self.AUTH_RATE_WINDOW = int(os.getenv("AUTH_RATE_WINDOW", "60"))

        # Response size limits
        self.MAX_RESPONSE_CHARS = int(os.getenv("MAX_RESPONSE_CHARS", "80000"))
        self.CONTENT_TRUNCATE_LENGTH = int(os.getenv("CONTENT_TRUNCATE_LENGTH", "300"))
        self.MIN_RESULT_COUNT = int(os.getenv("MIN_RESULT_COUNT", "3"))

        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

        # Memory decay
        self.DECAY_ARCHIVE_THRESHOLD = float(os.getenv("DECAY_ARCHIVE_THRESHOLD", "0.1"))
        self.DECAY_CHECK_INTERVAL_HOURS = int(os.getenv("DECAY_CHECK_INTERVAL_HOURS", "24"))

        # Hook timeouts
        self.API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

        # CLaRa-inspired memory tiers
        self.TIER_HOT_MAX_AGE_DAYS = int(os.getenv("TIER_HOT_MAX_AGE_DAYS", "14"))
        self.TIER_HOT_MIN_IMPORTANCE = int(os.getenv("TIER_HOT_MIN_IMPORTANCE", "7"))
        self.TIER_WARM_MAX_AGE_DAYS = int(os.getenv("TIER_WARM_MAX_AGE_DAYS", "90"))

        # Memory consolidation
        self.CONSOLIDATION_THRESHOLD = float(os.getenv("CONSOLIDATION_THRESHOLD", "0.85"))
        self.CONSOLIDATION_MIN_GROUP = int(os.getenv("CONSOLIDATION_MIN_GROUP", "3"))
        self.CONSOLIDATION_MAX_GROUP = int(os.getenv("CONSOLIDATION_MAX_GROUP", "20"))
        self.CONSOLIDATION_MAX_PER_RUN = int(os.getenv("CONSOLIDATION_MAX_PER_RUN", "5"))
        self.CONSOLIDATION_INTERVAL_HOURS = int(os.getenv("CONSOLIDATION_INTERVAL_HOURS", "12"))

        # Embedding pipeline
        self.EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "500"))
        self.EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
        self.EMBEDDING_PRECOMPUTE_INTERVAL = int(os.getenv("EMBEDDING_PRECOMPUTE_INTERVAL", "60"))

        # Adaptive search ranking
        self.DEFAULT_SEARCH_TEMPERATURE = float(os.getenv("DEFAULT_SEARCH_TEMPERATURE", "1.0"))

        # Cross-session awareness
        self.SESSION_IDLE_THRESHOLD_MINUTES = int(os.getenv("SESSION_IDLE_THRESHOLD_MINUTES", "10"))
        self.SESSION_COMPLETED_THRESHOLD_MINUTES = int(os.getenv("SESSION_COMPLETED_THRESHOLD_MINUTES", "30"))
        self.SESSION_ACTIVITY_MAX_AGE_HOURS = int(os.getenv("SESSION_ACTIVITY_MAX_AGE_HOURS", "24"))
        self.SESSION_CLEANUP_INTERVAL_SECONDS = int(os.getenv("SESSION_CLEANUP_INTERVAL_SECONDS", "300"))

        # Validate configuration
        self._validate()

    def _validate(self):
        """Validate configuration values and warn on invalid settings."""
        warnings = []

        # Critical: port must be valid
        if not (1 <= self.PORT <= 65535):
            raise ValueError(f"PORT must be 1-65535, got {self.PORT}")

        # Critical: pool size must be positive
        if self.DB_POOL_SIZE < 1:
            raise ValueError(f"DB_POOL_SIZE must be >= 1, got {self.DB_POOL_SIZE}")

        # Critical: embedding provider must be valid
        valid_providers = ("sentence-transformers", "ollama")
        if self.EMBEDDING_PROVIDER not in valid_providers:
            raise ValueError(f"EMBEDDING_PROVIDER must be one of {valid_providers}, got '{self.EMBEDDING_PROVIDER}'")

        # Warnings for non-critical misconfigurations
        if self.DB_TIMEOUT < 1.0:
            warnings.append(f"DB_TIMEOUT={self.DB_TIMEOUT}s is very low, may cause spurious timeouts")

        if self.DB_MAX_RETRIES < 0:
            warnings.append(f"DB_MAX_RETRIES={self.DB_MAX_RETRIES} is negative, setting to 0")
            self.DB_MAX_RETRIES = 0

        if self.EMBEDDING_DIM not in (384, 768, 1024):
            warnings.append(f"EMBEDDING_DIM={self.EMBEDDING_DIM} is unusual (expected 384/768/1024)")

        if not (0.0 <= self.CONSOLIDATION_THRESHOLD <= 1.0):
            warnings.append(f"CONSOLIDATION_THRESHOLD={self.CONSOLIDATION_THRESHOLD} out of range [0,1], clamping")
            self.CONSOLIDATION_THRESHOLD = max(0.0, min(1.0, self.CONSOLIDATION_THRESHOLD))

        if not (0.0 <= self.DECAY_ARCHIVE_THRESHOLD <= 1.0):
            warnings.append(f"DECAY_ARCHIVE_THRESHOLD={self.DECAY_ARCHIVE_THRESHOLD} out of range [0,1], clamping")
            self.DECAY_ARCHIVE_THRESHOLD = max(0.0, min(1.0, self.DECAY_ARCHIVE_THRESHOLD))

        if not (0.0 <= self.DEFAULT_SEARCH_TEMPERATURE <= 2.0):
            warnings.append(f"DEFAULT_SEARCH_TEMPERATURE={self.DEFAULT_SEARCH_TEMPERATURE} out of range [0,2], clamping")
            self.DEFAULT_SEARCH_TEMPERATURE = max(0.0, min(2.0, self.DEFAULT_SEARCH_TEMPERATURE))

        if self.MAX_RESPONSE_CHARS < 1000:
            warnings.append(f"MAX_RESPONSE_CHARS={self.MAX_RESPONSE_CHARS} is very low, responses may be cut off")

        if self.EMBEDDING_CACHE_SIZE < 0:
            warnings.append(f"EMBEDDING_CACHE_SIZE={self.EMBEDDING_CACHE_SIZE} is negative, setting to 0")
            self.EMBEDDING_CACHE_SIZE = 0

        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.LOG_LEVEL.upper() not in valid_log_levels:
            warnings.append(f"LOG_LEVEL='{self.LOG_LEVEL}' is invalid, using INFO")
            self.LOG_LEVEL = "INFO"

        if self.TIER_HOT_MAX_AGE_DAYS >= self.TIER_WARM_MAX_AGE_DAYS:
            warnings.append(
                f"TIER_HOT_MAX_AGE_DAYS ({self.TIER_HOT_MAX_AGE_DAYS}) >= "
                f"TIER_WARM_MAX_AGE_DAYS ({self.TIER_WARM_MAX_AGE_DAYS}), "
                f"hot tier would overlap warm tier"
            )

        for w in warnings:
            logger.warning(f"Config: {w}")

    def get_health_url(self) -> str:
        """Get the health check URL."""
        return f"{self.MEMORY_AGENT_URL}/health"

    def get_dashboard_url(self) -> str:
        """Get the dashboard URL."""
        return f"{self.MEMORY_AGENT_URL}/dashboard"

    def to_dict(self) -> dict:
        """Export configuration as dictionary (for debugging)."""
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(self).items()
            if not key.startswith("_")
        }


# Singleton instance
config = Config()


# Convenience exports for backwards compatibility
PORT = config.PORT
HOST = config.HOST
MEMORY_AGENT_URL = config.MEMORY_AGENT_URL
OLLAMA_HOST = config.OLLAMA_HOST
EMBEDDING_PROVIDER = config.EMBEDDING_PROVIDER
EMBEDDING_MODEL = config.EMBEDDING_MODEL
DATABASE_PATH = config.DATABASE_PATH
