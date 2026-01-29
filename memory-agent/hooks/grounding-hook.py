#!/usr/bin/env python3
"""
Grounding Hook for Claude Code - Automatic Context Injection

This script is called by Claude Code's UserPromptSubmit hook.
It fetches the current session context and outputs it to stdout,
which Claude Code automatically injects into Claude's context.

This is the REAL anti-hallucination layer - automatic, not relying on Claude to call tools.
"""

import os
import sys
import json
import logging
import requests
from pathlib import Path
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

    Example:
        safe_get(result, "result", "artifacts", 0, "parts", 0, "text")
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


def get_project_path():
    """Get current working directory as project path."""
    return os.getcwd()


def load_session_data():
    """Load session data from JSON file."""
    session_file = Path(get_project_path()) / ".claude_session"
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
    session_file = Path(get_project_path()) / ".claude_session"
    try:
        session_file.write_text(json.dumps(data, indent=2))
    except (IOError, OSError) as e:
        logger.warning(f"Failed to save session data: {e}")


def get_session_id():
    """Get or create session ID from environment or file."""
    # Try environment variable first
    session_id = os.getenv("CLAUDE_SESSION_ID")
    if session_id:
        return session_id

    # Try session file in project
    data = load_session_data()
    return data.get("session_id") if data else None


def call_memory_agent(skill_id: str, params: dict) -> Optional[dict]:
    """Call the memory agent API."""
    try:
        response = requests.post(
            f"{MEMORY_AGENT_URL}/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "grounding-hook",
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
        result = response.json()

        # Safely extract the artifact text using safe_get
        artifact_text = safe_get(result, "result", "artifacts", 0, "parts", 0, "text")
        if artifact_text:
            try:
                return json.loads(artifact_text)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse artifact text as JSON for skill '{skill_id}': {e}")
                return None
        return None

    except requests.RequestException as e:
        # Silently fail - don't break Claude Code if memory agent is down
        logger.debug(f"Memory agent request failed for skill '{skill_id}': {e}")
        return None
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to decode memory agent response for skill '{skill_id}': {e}")
        return None


def format_grounding_context(context: dict) -> str:
    """Format the grounding context for injection."""
    if not context or not context.get("success"):
        return ""

    grounding = context.get("grounding", {})

    lines = ["[GROUNDING CONTEXT - VERIFY BEFORE RESPONDING]"]

    # Current goal
    if grounding.get("current_goal"):
        lines.append(f"CURRENT GOAL: {grounding['current_goal']}")

    # Entity registry
    registry = grounding.get("entity_registry", {})
    if registry:
        lines.append("ENTITY REGISTRY (use these exact references):")
        for key, value in list(registry.items())[:5]:
            lines.append(f"  - {key}: {value}")

    # Anchors (verified facts)
    anchors = grounding.get("anchors", [])
    if anchors:
        lines.append("ANCHORS (verified facts - DO NOT CONTRADICT):")
        for anchor in anchors[:5]:
            lines.append(f"  - {anchor}")

    # Recent decisions
    decisions = grounding.get("decisions", [])
    if decisions:
        lines.append("RECENT DECISIONS:")
        for decision in decisions[:3]:
            lines.append(f"  - {decision}")

    # Recent events
    events = grounding.get("recent_events", [])
    if events:
        lines.append("RECENT EVENTS:")
        for event in events[:5]:
            lines.append(f"  - [{event.get('type', '?')}] {event.get('summary', '')}")

    # Contradictions warning
    contradictions = grounding.get("contradictions", [])
    if contradictions:
        lines.append("WARNING - POTENTIAL CONTRADICTIONS DETECTED:")
        for c in contradictions[:3]:
            lines.append(f"  - {c.get('content', '')[:100]}")

    # Pending questions
    pending = grounding.get("pending_questions", [])
    if pending:
        lines.append("PENDING QUESTIONS:")
        for q in pending[:3]:
            lines.append(f"  - {q}")

    lines.append("[/GROUNDING CONTEXT]")
    lines.append("")  # Empty line after

    return "\n".join(lines)


def main():
    """Main entry point for the hook."""
    project_path = get_project_path()
    session_id = get_session_id()

    # If no session, try to init one
    if not session_id:
        init_result = call_memory_agent("state_init_session", {
            "project_path": project_path
        })
        if init_result and init_result.get("session_id"):
            session_id = init_result["session_id"]
            # Save session data as JSON
            save_session_data({"session_id": session_id})

    if not session_id:
        # No session, no grounding - exit silently
        sys.exit(0)

    # Get grounding context
    context = call_memory_agent("context_refresh", {
        "session_id": session_id,
        "include_recent_events": 5,
        "include_state": True,
        "include_checkpoint": True,
        "check_contradictions": True
    })

    if context:
        grounding_text = format_grounding_context(context)
        if grounding_text:
            print(grounding_text)

    sys.exit(0)


if __name__ == "__main__":
    main()
