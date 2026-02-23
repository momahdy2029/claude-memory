#!/usr/bin/env python3
"""
Pre-Tool Decision Hook for Claude Code

This script captures the DECISION (why) before each tool call.
Called via PreToolUse hook - logs the reasoning before action executes.

Creates a chain: user_request → decision → action
The decision event ID is stored for PostToolUse to link the action.

Inspired by A* algorithm: selective capture of important decisions,
not blind logging of everything (Dijkstra-style).
"""

import os
import sys
import json
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

# Tools worth capturing decisions for (high-signal actions)
DECISION_WORTHY_TOOLS = {
    "Edit": "editing",
    "Write": "creating",
    "Bash": "executing",
    "Task": "delegating to agent",
}

# Low-signal tools (just reading, not changing state)
READ_ONLY_TOOLS = {"Read", "Grep", "Glob", "WebFetch", "WebSearch"}


def safe_get(data, *keys, default=None):
    """Safely navigate nested data structures."""
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
            return json.loads(content)
        except (json.JSONDecodeError, IOError, OSError) as e:
            logger.debug(f"Failed to load session data: {e}")
            return None
    return None


def save_session_data(data: dict):
    """Save session data to JSON file."""
    session_file = Path(os.getcwd()) / ".claude_session"
    try:
        session_file.write_text(json.dumps(data, indent=2))
    except (IOError, OSError) as e:
        logger.warning(f"Failed to save session data: {e}")


def call_memory_agent(skill_id: str, params: dict) -> Optional[dict]:
    """Call the memory agent API."""
    try:
        response = requests.post(
            f"{MEMORY_AGENT_URL}/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "pre-tool-hook",
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
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.debug(f"Memory agent request failed: {e}")
        return None


def generate_decision_summary(tool_name: str, tool_input: dict) -> str:
    """Generate a human-readable decision summary.

    This is the WHY, not the WHAT.
    """
    verb = DECISION_WORTHY_TOOLS.get(tool_name, "using")

    if tool_name == "Edit":
        file_path = tool_input.get("file_path", "unknown")
        old_string = tool_input.get("old_string", "")[:50]
        return f"Decided to modify {Path(file_path).name}: changing '{old_string}...'"

    elif tool_name == "Write":
        file_path = tool_input.get("file_path", "unknown")
        return f"Decided to create/overwrite {Path(file_path).name}"

    elif tool_name == "Bash":
        command = tool_input.get("command", "")[:80]
        description = tool_input.get("description", "")
        if description:
            return f"Decided to run: {description}"
        return f"Decided to execute: {command}"

    elif tool_name == "Task":
        agent_type = tool_input.get("subagent_type", "unknown")
        task_desc = tool_input.get("description", "")[:50]
        return f"Decided to delegate to {agent_type} agent: {task_desc}"

    return f"Decided to use {tool_name}"


def main():
    """Capture the decision before tool execution."""
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, IOError, OSError) as e:
        logger.debug(f"Failed to parse hook input: {e}")
        sys.exit(0)

    tool_name = hook_input.get("tool_name") or hook_input.get("tool")
    if not tool_name:
        sys.exit(0)

    # Skip read-only tools (low signal, high noise)
    # This is the A* heuristic: skip nodes that don't lead to the goal
    if tool_name in READ_ONLY_TOOLS:
        sys.exit(0)

    # Skip if not a decision-worthy tool
    if tool_name not in DECISION_WORTHY_TOOLS:
        sys.exit(0)

    # Load session data, prefer session_id from stdin
    session_data = load_session_data() or {}
    session_id = hook_input.get("session_id") or session_data.get("session_id")
    if not session_id:
        sys.exit(0)

    # Get root event (user request) for causal chain
    root_event_id = session_data.get("current_request_id")

    tool_input = hook_input.get("tool_input") or hook_input.get("input") or {}

    # Generate decision summary (the WHY)
    decision_summary = generate_decision_summary(tool_name, tool_input)

    # Log the decision event
    log_params = {
        "session_id": session_id,
        "event_type": "decision",
        "summary": decision_summary[:200],
        "details": json.dumps({
            "tool": tool_name,
            "reasoning": "Pre-action decision capture"
        }),
        "project_path": os.getcwd()
    }

    # Link to causal chain
    if root_event_id:
        log_params["root_event_id"] = root_event_id
        log_params["parent_event_id"] = root_event_id

    result = call_memory_agent("timeline_log", log_params)

    # Store the decision event ID so PostToolUse can link to it
    if result:
        artifact_text = safe_get(result, "result", "artifacts", 0, "parts", 0, "text")
        if artifact_text:
            try:
                skill_result = json.loads(artifact_text)
                decision_event_id = skill_result.get("event_id")
                if decision_event_id:
                    session_data["current_decision_id"] = decision_event_id
                    session_data["pending_tool"] = tool_name
                    save_session_data(session_data)
            except json.JSONDecodeError:
                pass

    sys.exit(0)


if __name__ == "__main__":
    main()
