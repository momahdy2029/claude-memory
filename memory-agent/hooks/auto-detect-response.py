#!/usr/bin/env python3
"""
Auto-Detect Response Hook for Claude Code

This script runs after Claude responds (Stop hook).
It analyzes Claude's response for decisions and observations,
logging them to the timeline automatically.

Also logs an 'outcome' event summarizing the result of the request.
"""

import os
import sys
import json
import re
import logging
import requests
from pathlib import Path

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

# Outcome detection patterns
OUTCOME_SUCCESS_PATTERNS = [
    r"Done[.!]",
    r"Completed[.!]",
    r"Finished[.!]",
    r"Fixed the",
    r"Resolved the",
    r"Successfully",
    r"Created",
    r"Added",
    r"Implemented",
    r"Updated",
    r"I've made the changes",
    r"The changes are complete",
    r"Here's the",
    r"I've updated",
    r"I've fixed",
    r"I've added",
]

OUTCOME_FAILED_PATTERNS = [
    r"Error:",
    r"Failed:",
    r"Could not",
    r"Unable to",
    r"I couldn't",
    r"This won't work",
    r"There's a problem",
]

OUTCOME_PARTIAL_PATTERNS = [
    r"Let me know if",
    r"Should I",
    r"Would you like me to",
    r"I can also",
    r"If you want",
    r"I need more information",
    r"Could you clarify",
]


def detect_outcome(response_text: str) -> tuple:
    """
    Detect outcome status and generate summary from response.

    Returns:
        tuple: (status, summary) where status is 'success', 'failed', or 'partial'
    """
    # Check for failure first (most specific)
    for pattern in OUTCOME_FAILED_PATTERNS:
        if re.search(pattern, response_text, re.IGNORECASE):
            # Extract a brief summary from around the match
            match = re.search(pattern + r".{0,100}", response_text, re.IGNORECASE)
            summary = match.group(0)[:150] if match else "Request encountered an error"
            return "failed", f"FAILED - {summary}"

    # Check for partial/pending
    for pattern in OUTCOME_PARTIAL_PATTERNS:
        if re.search(pattern, response_text, re.IGNORECASE):
            # Get first line or first 100 chars as summary
            first_line = response_text.split('\n')[0][:150]
            return "partial", f"PARTIAL - {first_line}"

    # Check for explicit success
    for pattern in OUTCOME_SUCCESS_PATTERNS:
        if re.search(pattern, response_text, re.IGNORECASE):
            # Get first line as success summary
            first_line = response_text.split('\n')[0][:150]
            return "success", f"SUCCESS - {first_line}"

    # Default: assume success if response is substantial
    if len(response_text) > 100:
        first_line = response_text.split('\n')[0][:150]
        return "success", f"COMPLETED - {first_line}"

    return "partial", "Response generated"


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
                "id": "detect-hook",
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
    """Analyze Claude's response and log detected events."""
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse hook input JSON: {e}")
        sys.exit(0)
    except (IOError, OSError) as e:
        logger.debug(f"Failed to read stdin: {e}")
        sys.exit(0)

    # Get Claude's response
    # The Stop hook receives the assistant's message
    response_text = ""

    # Try different possible formats
    if "assistant_message" in hook_input:
        response_text = hook_input["assistant_message"]
    elif "message" in hook_input:
        msg = hook_input["message"]
        if isinstance(msg, str):
            response_text = msg
        elif isinstance(msg, dict):
            response_text = msg.get("content", "")
    elif "transcript" in hook_input:
        # Get last assistant message from transcript
        transcript = hook_input["transcript"]
        for msg in reversed(transcript):
            if msg.get("role") == "assistant":
                response_text = msg.get("content", "")
                break

    if not response_text or len(response_text) < 50:
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

    # Build params for auto-detect
    auto_detect_params = {
        "session_id": session_id,
        "response_text": response_text,
        "project_path": os.getcwd()
    }

    # Add causal chain link if we have a root event
    if root_event_id:
        auto_detect_params["parent_event_id"] = root_event_id
        auto_detect_params["root_event_id"] = root_event_id

    # Call auto-detect to analyze response for decisions/observations
    call_memory_agent("timeline_auto_detect", auto_detect_params)

    # Log an outcome event summarizing the result
    if root_event_id:
        status, summary = detect_outcome(response_text)
        call_memory_agent("timeline_log", {
            "session_id": session_id,
            "event_type": "outcome",
            "summary": summary[:200],
            "status": status,
            "root_event_id": root_event_id,
            "parent_event_id": root_event_id,
            "project_path": os.getcwd()
        })

    sys.exit(0)


if __name__ == "__main__":
    main()
