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
import requests
from pathlib import Path
from datetime import datetime

MEMORY_AGENT_URL = os.getenv("MEMORY_AGENT_URL", "http://localhost:8100")

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

def save_session_data(data: dict):
    """Save session data to JSON file."""
    session_file = Path(os.getcwd()) / ".claude_session"
    session_file.write_text(json.dumps(data, indent=2))

def get_session_id():
    """Get session ID from environment or file."""
    session_id = os.getenv("CLAUDE_SESSION_ID")
    if session_id:
        return session_id

    data = load_session_data()
    return data.get("session_id") if data else None

def call_memory_agent(skill_id: str, params: dict) -> dict:
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
            timeout=3
        )
        return response.json()
    except:
        return None

def main():
    """Log the user's request to timeline."""
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except:
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
        try:
            # Parse result - the memory agent returns JSON-RPC format
            if isinstance(result, dict):
                # Handle JSON-RPC response
                rpc_result = result.get("result", {})
                if isinstance(rpc_result, dict):
                    artifact = rpc_result.get("artifacts", [{}])[0] if rpc_result.get("artifacts") else {}
                    # The artifact parts contain the skill result as JSON string
                    parts = artifact.get("parts", [])
                    for part in parts:
                        if part.get("type") == "text":
                            try:
                                skill_result = json.loads(part.get("text", "{}"))
                                event_id = skill_result.get("event_id")
                                if event_id:
                                    session_data["current_request_id"] = event_id
                                    session_data["request_started_at"] = datetime.now().isoformat()
                                    save_session_data(session_data)
                            except json.JSONDecodeError:
                                pass
        except Exception:
            pass

    sys.exit(0)

if __name__ == "__main__":
    main()
