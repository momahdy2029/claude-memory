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
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

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
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", "8102"))
        self.MEMORY_AGENT_URL = os.getenv(
            "MEMORY_AGENT_URL",
            f"http://localhost:{self.PORT}"
        )

        # Ollama / Embeddings
        self.OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
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

        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

        # Hook timeouts
        self.API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

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
EMBEDDING_MODEL = config.EMBEDDING_MODEL
DATABASE_PATH = config.DATABASE_PATH
