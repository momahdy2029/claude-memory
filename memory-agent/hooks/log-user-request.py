#!/usr/bin/env python3
"""
User Request Logger Hook for Claude Code

This script logs user requests to the session timeline automatically.
Called via UserPromptSubmit hook - logs the request, then grounding-hook injects context.

The session file stores JSON with:
- session_id: The current session ID
- current_request_id: The event ID of the current user_request (for causal chain linking)
- request_started_at: Timestamp of when the request started
"""

import os
import sys
import json
import logging
import requests
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

# Configure logging to stderr (important for Claude Code hooks)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Configuration from environment
MEMORY_AGENT_URL = os.getenv("MEMORY_AGENT_URL", "http://localhost:8102")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))


def safe_get(data: Any, *keys, default: Any = None) -> Any:
    """
    Safely navigate nested data structures (dicts and lists).

    Args:
        data: The data structure to navigate
        *keys: Keys (str for dict) or indices (int for list) to traverse
        default: Value to return if path doesn't exist

    Returns:
        The value at the path, or default if not found
    """
    for key in keys:
        if data is None:
            return default
        if isinstance(data, dict):
            data = data.get(key, default)
        elif isinstance(data, list) and isinstance(key, int):
            if 0 <= key < len(data):
                data = data[key]
            else:
                return default
        else:
            return default
    return data


def load_session_data():
    """Load session data from JSON file."""
    session_file = Path(os.getcwd()) / ".claude_session"
    if session_file.exists():
        try:
            content = session_file.read_text().strip()
            # Try JSON format first
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error, trying legacy format: {e}")
            # Fall back to legacy plain text format (just session_id)
            try:
                content = session_file.read_text().strip()
                return {"session_id": content}
            except (IOError, OSError) as read_err:
                logger.warning(f"Failed to read session file: {read_err}")
                return None
        except (IOError, OSError) as e:
            logger.warning(f"Failed to read session file: {e}")
            return None
    return None


def save_session_data(data: dict):
    """Save session data to JSON file."""
    session_file = Path(os.getcwd()) / ".claude_session"
    try:
        session_file.write_text(json.dumps(data, indent=2))
    except (IOError, OSError) as e:
        logger.warning(f"Failed to save session data: {e}")


def get_session_id():
    """Get session ID from environment or file."""
    session_id = os.getenv("CLAUDE_SESSION_ID")
    if session_id:
        return session_id

    data = load_session_data()
    return data.get("session_id") if data else None


def call_memory_agent(skill_id: str, params: dict) -> Optional[dict]:
    """Call the memory agent API."""
    try:
        response = requests.post(
            f"{MEMORY_AGENT_URL}/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "log-hook",
                "method": "tasks/send",
                "params": {
                    "message": {"parts": [{"type": "text", "text": ""}]},
                    "metadata": {
                        "skill_id": skill_id,
                        "params": params
                    }
                }
            },
            timeout=API_TIMEOUT
        )
        return response.json()
    except requests.RequestException as e:
        logger.debug(f"Memory agent request failed for skill '{skill_id}': {e}")
        return None
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to decode memory agent response for skill '{skill_id}': {e}")
        return None


def main():
    """Log the user's request to timeline."""
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse hook input JSON: {e}")
        sys.exit(0)
    except (IOError, OSError) as e:
        logger.debug(f"Failed to read stdin: {e}")
        sys.exit(0)

    # Get user message from hook input
    user_message = hook_input.get("user_prompt", "")
    if not user_message:
        # Try alternative format
        session_messages = hook_input.get("session_messages", [])
        if session_messages:
            last_msg = session_messages[-1]
            if last_msg.get("role") == "user":
                user_message = last_msg.get("content", "")

    if not user_message:
        sys.exit(0)

    # Load session data
    session_data = load_session_data()
    if not session_data:
        sys.exit(0)

    session_id = session_data.get("session_id")
    if not session_id:
        sys.exit(0)

    # Truncate long messages
    summary = user_message[:200]
    if len(user_message) > 200:
        summary += "..."

    # Log to timeline
    result = call_memory_agent("timeline_log", {
        "session_id": session_id,
        "event_type": "user_request",
        "summary": summary,
        "details": user_message if len(user_message) > 200 else None,
        "project_path": os.getcwd()
    })

    # Save the event_id as current_request_id for causal chain linking
    if result:
        # Parse result using safe_get - the memory agent returns JSON-RPC format
        # Navigate: result -> artifacts[0] -> parts[0] -> text
        artifact_text = safe_get(result, "result", "artifacts", 0, "parts", 0, "text")

        if artifact_text:
            try:
                skill_result = json.loads(artifact_text)
                event_id = skill_result.get("event_id")
                if event_id:
                    session_data["current_request_id"] = event_id
                    session_data["request_started_at"] = datetime.now().isoformat()
                    save_session_data(session_data)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse skill result JSON: {e}")

    sys.exit(0)


if __name__ == "__main__":
    main()
