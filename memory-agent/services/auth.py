"""API authentication service for the Memory Agent.

Provides API key-based authentication with:
- Key generation and storage
- Request validation via X-Memory-Key header
- Rate limiting per key
- Key rotation support
"""
import os
import json
import time
import secrets
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from threading import Lock
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configuration
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"  # Default: disabled for local use
KEY_FILE = os.getenv("AUTH_KEY_FILE", str(Path.home() / ".claude" / "memory-agent-keys.json"))
DEFAULT_RATE_LIMIT = int(os.getenv("AUTH_RATE_LIMIT", "100"))  # requests per minute
RATE_LIMIT_WINDOW = int(os.getenv("AUTH_RATE_WINDOW", "60"))  # seconds

# Endpoints that don't require authentication
# This is a local-only tool, so all API endpoints are exempt by default.
# When AUTH_ENABLED=true, only /skills/call and /tasks/send require a key.
EXEMPT_ENDPOINTS = [
    "/health",
    "/health/live",
    "/ready",
    "/.well-known/agent.json",
    "/docs",
    "/openapi.json",
    "/dashboard",
    "/favicon.ico",
    "/ws",  # WebSocket
    "/a2a",  # Agent-to-Agent protocol
    "/api/",  # All dashboard and REST API endpoints
]


class RateLimiter:
    """Simple sliding window rate limiter."""

    def __init__(self):
        self._requests: Dict[str, List[float]] = {}
        self._lock = Lock()

    def is_allowed(self, key: str, limit: int = DEFAULT_RATE_LIMIT, window: int = RATE_LIMIT_WINDOW) -> Tuple[bool, int]:
        """Check if a request is allowed under rate limits.

        Args:
            key: The API key
            limit: Maximum requests per window
            window: Window size in seconds

        Returns:
            Tuple of (allowed, remaining_requests)
        """
        now = time.time()
        with self._lock:
            # Initialize or get request list
            if key not in self._requests:
                self._requests[key] = []

            # Remove expired requests
            cutoff = now - window
            self._requests[key] = [t for t in self._requests[key] if t > cutoff]

            # Check limit
            current_count = len(self._requests[key])
            if current_count >= limit:
                return False, 0

            # Record this request
            self._requests[key].append(now)
            return True, limit - current_count - 1

    def get_stats(self, key: str) -> Dict[str, Any]:
        """Get rate limit stats for a key."""
        now = time.time()
        with self._lock:
            requests = self._requests.get(key, [])
            recent = [t for t in requests if t > now - RATE_LIMIT_WINDOW]
            return {
                "current_count": len(recent),
                "limit": DEFAULT_RATE_LIMIT,
                "window_seconds": RATE_LIMIT_WINDOW,
                "remaining": max(0, DEFAULT_RATE_LIMIT - len(recent))
            }


class AuthService:
    """API key authentication service.

    Features:
    - Secure key generation
    - Key storage in JSON file
    - Multiple keys support (for different clients)
    - Rate limiting per key
    - Key rotation
    """

    def __init__(self, key_file: str = KEY_FILE):
        self.key_file = Path(key_file)
        self.enabled = AUTH_ENABLED
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._rate_limiter = RateLimiter()
        self._lock = Lock()
        self._load_keys()

    def _load_keys(self):
        """Load keys from file."""
        if self.key_file.exists():
            try:
                with open(self.key_file, 'r') as f:
                    data = json.load(f)
                    self._keys = data.get("keys", {})
            except (json.JSONDecodeError, IOError):
                self._keys = {}

        # Generate default key if none exist
        if not self._keys and self.enabled:
            self.generate_key("default", "Default API key")

    def _save_keys(self):
        """Save keys to file."""
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.key_file, 'w') as f:
            json.dump({
                "keys": self._keys,
                "updated_at": datetime.now().isoformat()
            }, f, indent=2)
        # Set restrictive permissions (owner read/write only)
        try:
            os.chmod(self.key_file, 0o600)
        except OSError:
            pass  # Windows may not support this

    def _hash_key(self, key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()

    def generate_key(self, name: str, description: str = "", rate_limit: int = DEFAULT_RATE_LIMIT) -> str:
        """Generate a new API key.

        Args:
            name: Unique name for the key
            description: Description of what this key is for
            rate_limit: Custom rate limit for this key

        Returns:
            The generated API key (only returned once!)
        """
        with self._lock:
            # Generate a secure random key
            key = f"mem_{secrets.token_urlsafe(32)}"
            key_hash = self._hash_key(key)

            self._keys[key_hash] = {
                "name": name,
                "description": description,
                "rate_limit": rate_limit,
                "created_at": datetime.now().isoformat(),
                "last_used": None,
                "use_count": 0,
                "revoked": False
            }

            self._save_keys()
            return key

    def validate_key(self, key: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validate an API key.

        Args:
            key: The API key to validate

        Returns:
            Tuple of (valid, error_message, key_info)
        """
        if not self.enabled:
            return True, None, {"name": "auth_disabled"}

        if not key:
            return False, "Missing API key", None

        key_hash = self._hash_key(key)

        with self._lock:
            if key_hash not in self._keys:
                return False, "Invalid API key", None

            key_info = self._keys[key_hash]

            if key_info.get("revoked"):
                return False, "API key has been revoked", None

            # Check rate limit
            rate_limit = key_info.get("rate_limit", DEFAULT_RATE_LIMIT)
            allowed, remaining = self._rate_limiter.is_allowed(key_hash, rate_limit)

            if not allowed:
                return False, "Rate limit exceeded", None

            # Update usage stats
            key_info["last_used"] = datetime.now().isoformat()
            key_info["use_count"] = key_info.get("use_count", 0) + 1
            self._keys[key_hash] = key_info

            return True, None, {
                "name": key_info["name"],
                "rate_remaining": remaining
            }

    def revoke_key(self, name: str) -> bool:
        """Revoke a key by name.

        Args:
            name: Name of the key to revoke

        Returns:
            True if key was found and revoked
        """
        with self._lock:
            for key_hash, info in self._keys.items():
                if info["name"] == name:
                    info["revoked"] = True
                    info["revoked_at"] = datetime.now().isoformat()
                    self._keys[key_hash] = info
                    self._save_keys()
                    return True
            return False

    def rotate_key(self, name: str) -> Optional[str]:
        """Rotate a key (revoke old, generate new with same name).

        Args:
            name: Name of the key to rotate

        Returns:
            New API key, or None if key not found
        """
        with self._lock:
            # Find and revoke old key
            old_info = None
            for key_hash, info in self._keys.items():
                if info["name"] == name and not info.get("revoked"):
                    info["revoked"] = True
                    info["revoked_at"] = datetime.now().isoformat()
                    old_info = info
                    break

            if old_info is None:
                return None

        # Generate new key with same settings
        return self.generate_key(
            name=name,
            description=old_info.get("description", ""),
            rate_limit=old_info.get("rate_limit", DEFAULT_RATE_LIMIT)
        )

    def list_keys(self) -> List[Dict[str, Any]]:
        """List all keys (without the actual key values)."""
        with self._lock:
            return [
                {
                    "name": info["name"],
                    "description": info.get("description", ""),
                    "created_at": info["created_at"],
                    "last_used": info.get("last_used"),
                    "use_count": info.get("use_count", 0),
                    "rate_limit": info.get("rate_limit", DEFAULT_RATE_LIMIT),
                    "revoked": info.get("revoked", False)
                }
                for info in self._keys.values()
            ]

    def is_exempt(self, path: str) -> bool:
        """Check if a path is exempt from authentication."""
        return any(path.startswith(exempt) for exempt in EXEMPT_ENDPOINTS)

    def get_stats(self) -> Dict[str, Any]:
        """Get authentication statistics."""
        with self._lock:
            active_keys = sum(1 for k in self._keys.values() if not k.get("revoked"))
            revoked_keys = sum(1 for k in self._keys.values() if k.get("revoked"))
            total_uses = sum(k.get("use_count", 0) for k in self._keys.values())

            return {
                "enabled": self.enabled,
                "active_keys": active_keys,
                "revoked_keys": revoked_keys,
                "total_uses": total_uses,
                "rate_limit_default": DEFAULT_RATE_LIMIT,
                "rate_limit_window": RATE_LIMIT_WINDOW,
                "key_file": str(self.key_file)
            }


# Global instance
_auth: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Get the global auth service instance."""
    global _auth
    if _auth is None:
        _auth = AuthService()
    return _auth


def validate_request(key: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Convenience function to validate a request."""
    return get_auth_service().validate_key(key)
