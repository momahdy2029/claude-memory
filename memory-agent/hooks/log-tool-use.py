#!/usr/bin/env python3
"""
Tool Use Logger Hook for Claude Code

This script logs tool calls to the session timeline.
Called via PostToolUse hook - logs the action after it completes.

Reads current_request_id from .claude_session to link actions to the root user request.
"""

import os
import sys
import json
import re
import logging
import requests
from pathlib import Path
from typing import Optional

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

# Tools that represent meaningful actions to track
TRACKABLE_TOOLS = {
    "Edit": "edited file",
    "Write": "wrote file",
    "Bash": "ran command",
    "Read": "read file",
    "Grep": "searched code",
    "Glob": "searched files"
}


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
    """Get session ID from file."""
    data = load_session_data()
    return data.get("session_id") if data else None


def call_memory_agent(skill_id: str, params: dict) -> Optional[dict]:
    """Call the memory agent API."""
    try:
        response = requests.post(
            f"{MEMORY_AGENT_URL}/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "tool-hook",
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


def extract_entities(tool_name: str, tool_input: dict) -> Optional[dict]:
    """Extract entity references from tool input."""
    entities = {}

    if tool_name in ["Edit", "Write", "Read"]:
        file_path = tool_input.get("file_path") or tool_input.get("path")
        if file_path:
            entities["files"] = [file_path]

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        # Extract file paths from command (simple heuristic)
        paths = re.findall(r'[\w\-./\\]+\.(py|js|ts|json|md|yaml|yml)', command)
        if paths:
            entities["files"] = paths

    if tool_name == "Grep":
        pattern = tool_input.get("pattern")
        if pattern:
            entities["patterns"] = [pattern]

    return entities if entities else None


def main():
    """Log the tool use to timeline."""
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse hook input JSON: {e}")
        sys.exit(0)
    except (IOError, OSError) as e:
        logger.debug(f"Failed to read stdin: {e}")
        sys.exit(0)

    tool_name = hook_input.get("tool_name") or hook_input.get("tool")
    if not tool_name:
        sys.exit(0)

    # Only track meaningful tools
    if tool_name not in TRACKABLE_TOOLS:
        sys.exit(0)

    # Load session data, prefer session_id from stdin
    session_data = load_session_data() or {}
    session_id = hook_input.get("session_id") or session_data.get("session_id")
    if not session_id:
        sys.exit(0)

    # Get the current request ID for causal chain linking
    root_event_id = session_data.get("current_request_id")

    # Get decision event ID (from PreToolUse hook) for proper chain linking
    # Chain: user_request → decision → action
    decision_event_id = session_data.get("current_decision_id")
    pending_tool = session_data.get("pending_tool")

    tool_input = hook_input.get("tool_input") or hook_input.get("input") or {}
    tool_output = hook_input.get("tool_output") or hook_input.get("output") or ""

    # Build summary
    action_verb = TRACKABLE_TOOLS.get(tool_name, "used tool")

    if tool_name in ["Edit", "Write", "Read"]:
        file_path = tool_input.get("file_path") or tool_input.get("path") or "unknown"
        # Get just filename
        filename = Path(file_path).name if file_path else "unknown"
        summary = f"{action_verb}: {filename}"
    elif tool_name == "Bash":
        command = tool_input.get("command", "")[:50]
        summary = f"{action_verb}: {command}"
    elif tool_name == "Grep":
        pattern = tool_input.get("pattern", "")[:30]
        summary = f"{action_verb} for: {pattern}"
    elif tool_name == "Glob":
        pattern = tool_input.get("pattern", "")[:30]
        summary = f"{action_verb}: {pattern}"
    else:
        summary = f"{action_verb}"

    # Check if successful
    success = True
    if isinstance(tool_output, str):
        if "error" in tool_output.lower() or "failed" in tool_output.lower():
            success = False

    # Extract entities
    entities = extract_entities(tool_name, tool_input)

    # Log to timeline with causal chain linking
    log_params = {
        "session_id": session_id,
        "event_type": "action",
        "summary": summary[:200],
        "details": json.dumps({"tool": tool_name, "input": tool_input})[:500] if tool_input else None,
        "entities": entities,
        "outcome": "success" if success else "failed",
        "project_path": os.getcwd()
    }

    # Add causal chain links
    # Chain: user_request → decision → action
    if root_event_id:
        log_params["root_event_id"] = root_event_id

    # Link to decision event if this is the tool that was pre-logged
    if decision_event_id and pending_tool == tool_name:
        log_params["parent_event_id"] = decision_event_id
        # Clear the pending decision after linking
        session_data.pop("current_decision_id", None)
        session_data.pop("pending_tool", None)
        save_session_data(session_data)
    elif root_event_id:
        log_params["parent_event_id"] = root_event_id

    call_memory_agent("timeline_log", log_params)

    sys.exit(0)


if __name__ == "__main__":
    main()
