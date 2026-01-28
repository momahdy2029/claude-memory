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
import requests
from pathlib import Path

MEMORY_AGENT_URL = os.getenv("MEMORY_AGENT_URL", "http://localhost:8100")

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
        content = session_file.read_text().strip()
        try:
            # Try JSON format first
            return json.loads(content)
        except json.JSONDecodeError:
            # Fall back to legacy plain text format (just session_id)
            return {"session_id": content}
    return None

def get_session_id():
    """Get session ID from file."""
    data = load_session_data()
    return data.get("session_id") if data else None

def call_memory_agent(skill_id: str, params: dict) -> dict:
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
            timeout=3
        )
        return response.json()
    except:
        return None

def extract_entities(tool_name: str, tool_input: dict) -> dict:
    """Extract entity references from tool input."""
    entities = {}

    if tool_name in ["Edit", "Write", "Read"]:
        file_path = tool_input.get("file_path") or tool_input.get("path")
        if file_path:
            entities["files"] = [file_path]

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        # Extract file paths from command (simple heuristic)
        import re
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
    except:
        sys.exit(0)

    tool_name = hook_input.get("tool_name") or hook_input.get("tool")
    if not tool_name:
        sys.exit(0)

    # Only track meaningful tools
    if tool_name not in TRACKABLE_TOOLS:
        sys.exit(0)

    # Load session data (includes current_request_id for causal chain)
    session_data = load_session_data()
    if not session_data:
        sys.exit(0)

    session_id = session_data.get("session_id")
    if not session_id:
        sys.exit(0)

    # Get the current request ID for causal chain linking
    root_event_id = session_data.get("current_request_id")

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

    # Add causal chain links if we have a root event
    if root_event_id:
        log_params["root_event_id"] = root_event_id
        log_params["parent_event_id"] = root_event_id  # Direct child of user request

    call_memory_agent("timeline_log", log_params)

    sys.exit(0)

if __name__ == "__main__":
    main()
