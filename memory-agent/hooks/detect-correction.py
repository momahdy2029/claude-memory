#!/usr/bin/env python3
"""
Correction Detection Hook for Claude Code

This script detects when the user is correcting Claude.
Corrections are automatically marked as anchors (verified facts).

Patterns detected:
- "No, it's actually..."
- "That's wrong, ..."
- "Not X, Y"
- "You're mistaken..."
- "The correct X is Y"
"""

import os
import sys
import json
import re
import requests
from pathlib import Path

# Configuration from environment
MEMORY_AGENT_URL = os.getenv("MEMORY_AGENT_URL", "http://localhost:8102")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

# Patterns that indicate a correction
CORRECTION_PATTERNS = [
    r"^no[,.]?\s+(?:it'?s?|that'?s?|the)\s+(?:actually|really)?\s*(.+)",
    r"^that'?s?\s+(?:wrong|incorrect|not right)[,.]?\s*(.+)?",
    r"^not\s+(\w+)[,.]?\s+(?:it'?s?|but)\s+(.+)",
    r"^you'?re?\s+(?:wrong|mistaken|incorrect)[,.]?\s*(.+)?",
    r"^(?:the\s+)?correct\s+(?:\w+\s+)?(?:is|are|should be)\s+(.+)",
    r"^actually[,.]?\s+(.+)",
    r"^I\s+(?:meant|said|wanted)\s+(.+)",
    r"^wrong[,.]?\s+(?:it'?s?|the)\s+(.+)",
]

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
                "id": "correction-hook",
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
    except:
        return None

def detect_correction(text: str) -> tuple[bool, str]:
    """
    Detect if text is a correction and extract the correction content.

    Returns:
        (is_correction, correction_content)
    """
    # Normalize text
    text_lower = text.strip().lower()
    text_clean = text.strip()

    for pattern in CORRECTION_PATTERNS:
        match = re.match(pattern, text_lower, re.IGNORECASE)
        if match:
            # Get the correction content
            groups = match.groups()
            correction = " ".join(g for g in groups if g) if groups else text_clean
            return True, correction or text_clean

    return False, ""

def main():
    """Detect corrections and mark as anchors."""
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except:
        sys.exit(0)

    # Get user message
    user_message = hook_input.get("user_prompt", "")
    if not user_message:
        session_messages = hook_input.get("session_messages", [])
        if session_messages:
            last_msg = session_messages[-1]
            if last_msg.get("role") == "user":
                user_message = last_msg.get("content", "")

    if not user_message:
        sys.exit(0)

    # Check if this is a correction
    is_correction, correction_content = detect_correction(user_message)

    if not is_correction:
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

    # Mark correction as an anchor
    call_memory_agent("mark_anchor", {
        "session_id": session_id,
        "fact": f"USER CORRECTION: {correction_content[:200]}",
        "details": f"Original message: {user_message}",
        "project_path": os.getcwd()
    })

    # Also update the event as a clarification with causal chain linking
    log_params = {
        "session_id": session_id,
        "event_type": "clarification",
        "summary": f"User corrected: {correction_content[:150]}",
        "details": user_message,
        "is_anchor": True,
        "confidence": 1.0,
        "project_path": os.getcwd()
    }

    if root_event_id:
        log_params["root_event_id"] = root_event_id
        log_params["parent_event_id"] = root_event_id

    call_memory_agent("timeline_log", log_params)

    # Output a reminder for Claude
    print(f"[CORRECTION DETECTED - ANCHOR CREATED]")
    print(f"The user corrected you: {correction_content[:200]}")
    print(f"This is now a verified fact. Do not contradict it.")
    print("")

    sys.exit(0)

if __name__ == "__main__":
    main()
