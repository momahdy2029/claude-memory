#!/usr/bin/env python3
"""Slim grounding hook v2 - single HTTP call, compact output.

Replaces the original grounding-hook.py (4-6 HTTP calls, verbose output)
with a single POST to /api/grounding-context that aggregates everything
server-side.

Also replaces: session_start.py, problem-detector.py, memory-first-reminder.py

Output: compact [MEM] line (<150 tokens)
Timeout: 3 seconds, silent fail
"""

import os
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("grounding-v2")

MEMORY_AGENT_URL = os.getenv("MEMORY_AGENT_URL", "http://localhost:8102")
TIMEOUT = 3  # seconds


def get_session_id() -> str:
    """Get session ID from env or .claude_session file."""
    sid = os.getenv("CLAUDE_SESSION_ID", "")
    if sid:
        return sid

    session_file = Path(os.getcwd()) / ".claude_session"
    if session_file.exists():
        try:
            content = session_file.read_text().strip()
            data = json.loads(content)
            return data.get("session_id", "")
        except (json.JSONDecodeError, IOError):
            return content  # legacy plain text format
    return ""


def get_user_input() -> str:
    """Extract user input from hook stdin."""
    try:
        if not sys.stdin.isatty():
            data = sys.stdin.read()
            if data:
                hook_data = json.loads(data)
                return hook_data.get("prompt", hook_data.get("user_prompt", ""))
    except Exception:
        pass
    return ""


def main():
    session_id = get_session_id()
    project_path = os.getcwd()
    user_input = get_user_input()

    if not session_id:
        # No session - try to initialize one via A2A
        try:
            import requests
            resp = requests.post(
                f"{MEMORY_AGENT_URL}/a2a",
                json={
                    "jsonrpc": "2.0",
                    "id": "grounding-v2-init",
                    "method": "tasks/send",
                    "params": {
                        "message": {"parts": [{"type": "text", "text": ""}]},
                        "metadata": {
                            "skill_id": "state_init_session",
                            "params": {"project_path": project_path},
                        },
                    },
                },
                timeout=TIMEOUT,
            )
            if resp.status_code == 200:
                result = resp.json()
                try:
                    text = result["result"]["artifacts"][0]["parts"][0]["text"]
                    data = json.loads(text)
                    session_id = data.get("session_id", "")
                    if session_id:
                        # Save for future hooks
                        sf = Path(project_path) / ".claude_session"
                        sf.write_text(json.dumps({"session_id": session_id}))
                except (KeyError, IndexError, json.JSONDecodeError):
                    pass
        except Exception as e:
            logger.debug(f"Session init failed: {e}")

    if not session_id:
        sys.exit(0)

    # Single aggregated call
    try:
        import requests
        resp = requests.post(
            f"{MEMORY_AGENT_URL}/api/grounding-context",
            json={
                "session_id": session_id,
                "project_path": project_path,
                "user_input": user_input,
            },
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            context = data.get("context", "")
            if context:
                print(context)
    except Exception as e:
        logger.debug(f"Grounding context call failed: {e}")
        # Silent fail - don't break Claude Code

    sys.exit(0)


if __name__ == "__main__":
    main()
