#!/usr/bin/env python3
"""Slim grounding hook v2 - single HTTP call, compact output.

Replaces the original grounding-hook.py (4-6 HTTP calls, verbose output)
with a single POST to /api/grounding-context that aggregates everything
server-side.

Also replaces: session_start.py, problem-detector.py, memory-first-reminder.py

Fresh session detection:
  - Tracks last grounded session_id via .claude_session_meta
  - Fresh session -> calls /api/grounding-context/rich (~500-800 tokens)
  - Continuing session -> calls /api/grounding-context (~150 tokens)

Design constraints:
  - Uses stdlib only (no pip dependencies) -- urllib.request, not requests
  - Timeout: 3 seconds, silent fail
  - Always exits 0 -- never blocks Claude Code
"""

import os
import sys
import json
import logging
import urllib.request
import urllib.error
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("grounding-v2")

MEMORY_AGENT_URL = os.getenv("MEMORY_AGENT_URL", "http://localhost:8102")
TIMEOUT = 3  # seconds
SESSION_META_DIR = Path.home() / ".claude"
SESSION_META_FILE = SESSION_META_DIR / ".claude_session_meta"


# ---------------------------------------------------------------------------
# HTTP helper (stdlib only -- no requests dependency)
# ---------------------------------------------------------------------------

def _http_post(url: str, payload: dict, timeout: float = TIMEOUT):
    """POST JSON to url and return parsed response dict, or None on failure."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        pass
    return None


# ---------------------------------------------------------------------------
# Stdin / session ID helpers
# ---------------------------------------------------------------------------

def read_stdin_payload() -> dict:
    """Read the full JSON payload from Claude Code stdin (once)."""
    try:
        if not sys.stdin.isatty():
            data = sys.stdin.read()
            if data:
                return json.loads(data)
    except Exception:
        pass
    return {}


def get_session_id(payload: dict, project_path: str) -> str:
    """Get session ID from stdin payload, env, or .claude_session file."""
    # 1. stdin payload (Claude Code's actual format)
    sid = payload.get("session_id", "")
    if sid:
        return sid

    # 2. env var
    sid = os.getenv("CLAUDE_SESSION_ID", "")
    if sid:
        return sid

    # 3. .claude_session file
    session_file = Path(project_path) / ".claude_session"
    if session_file.exists():
        try:
            content = session_file.read_text().strip()
            data = json.loads(content)
            return data.get("session_id", "")
        except (json.JSONDecodeError, IOError):
            return content  # legacy plain text format
    return ""


# ---------------------------------------------------------------------------
# Fresh session detection
# ---------------------------------------------------------------------------

def is_fresh_session(session_id: str, project_path: str) -> bool:
    """Detect if this is a fresh/resumed session by comparing to last grounded session_id.

    Uses .claude_session_meta to track what we last grounded for this project.
    Returns True if session_id differs from last grounded (= new session or context cleared).
    """
    try:
        if SESSION_META_FILE.exists():
            meta = json.loads(SESSION_META_FILE.read_text(encoding="utf-8"))
            last_grounded = meta.get(project_path, {}).get("last_grounded_session", "")
            last_grounded_count = meta.get(project_path, {}).get("grounding_count", 0)
            if last_grounded == session_id and last_grounded_count > 0:
                return False
            return True
    except (json.JSONDecodeError, OSError):
        pass
    return True  # Default to fresh if can't read meta


def update_session_meta(session_id: str, project_path: str):
    """Record that we've grounded this session_id for this project."""
    try:
        SESSION_META_DIR.mkdir(parents=True, exist_ok=True)
        meta = {}
        if SESSION_META_FILE.exists():
            try:
                meta = json.loads(SESSION_META_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                meta = {}

        if project_path not in meta:
            meta[project_path] = {}

        current = meta[project_path]
        if current.get("last_grounded_session") == session_id:
            current["grounding_count"] = current.get("grounding_count", 0) + 1
        else:
            current["last_grounded_session"] = session_id
            current["grounding_count"] = 1

        # Prune old projects (keep 20)
        if len(meta) > 20:
            for old_key in sorted(meta.keys())[:-20]:
                del meta[old_key]

        SESSION_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Self-reflection hint (appended every N prompts to avoid noise fatigue)
# ---------------------------------------------------------------------------

LEARNING_HINT_INTERVAL = 3  # Show hint every N prompts (1st, 4th, 7th, ...)

LEARNING_HINT = (
    "\n[IMPORTANT] After completing this response, tag any non-obvious discovery "
    "(a surprising fix, codebase quirk, reusable workflow, or gotcha) by appending "
    "exactly: <!-- LEARNED: one-sentence description -->  "
    "This is silently captured — the user won't see it. Skip if nothing novel."
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _get_grounding_count(session_id: str, project_path: str) -> int:
    """Read current grounding_count from session meta (0 if unknown)."""
    try:
        if SESSION_META_FILE.exists():
            meta = json.loads(SESSION_META_FILE.read_text(encoding="utf-8"))
            entry = meta.get(project_path, {})
            if entry.get("last_grounded_session") == session_id:
                return entry.get("grounding_count", 0)
    except (json.JSONDecodeError, OSError):
        pass
    return 0


def _should_show_hint(grounding_count: int) -> bool:
    """Show learning hint on 1st prompt and every LEARNING_HINT_INTERVAL after."""
    return grounding_count % LEARNING_HINT_INTERVAL == 0


def main():
    payload = read_stdin_payload()
    project_path = payload.get("cwd", "") or os.getcwd()
    user_input = payload.get("prompt", "") or payload.get("user_prompt", "")
    session_id = get_session_id(payload, project_path)

    if not session_id:
        # No session - try to initialize one via A2A
        try:
            result = _http_post(
                f"{MEMORY_AGENT_URL}/a2a",
                {
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
            )
            if result:
                try:
                    text = result["result"]["artifacts"][0]["parts"][0]["text"]
                    data = json.loads(text)
                    session_id = data.get("session_id", "")
                    if session_id:
                        sf = Path(project_path) / ".claude_session"
                        sf.write_text(json.dumps({"session_id": session_id}))
                except (KeyError, IndexError, json.JSONDecodeError):
                    pass
        except Exception as e:
            logger.debug(f"Session init failed: {e}")

    if not session_id:
        sys.exit(0)

    # Register session as active (for cross-session awareness)
    try:
        _http_post(
            f"{MEMORY_AGENT_URL}/api/sessions/register",
            {"session_id": session_id, "project_path": project_path},
        )
    except Exception as e:
        logger.debug(f"Session register failed: {e}")

    # Detect fresh vs continuing session
    fresh = is_fresh_session(session_id, project_path)

    # Determine if learning hint should be shown this prompt
    count = _get_grounding_count(session_id, project_path)
    hint = LEARNING_HINT if _should_show_hint(count) else ""

    if fresh:
        # Fresh session: use rich grounding context (~500-800 tokens)
        # Always include hint on fresh sessions
        try:
            data = _http_post(
                f"{MEMORY_AGENT_URL}/api/grounding-context/rich",
                {"session_id": session_id, "project_path": project_path},
            )
            if data:
                context = data.get("context", "")
                if context:
                    print(context + LEARNING_HINT)
                    update_session_meta(session_id, project_path)
                    sys.exit(0)
        except Exception as e:
            logger.debug(f"Rich grounding context call failed: {e}")
            # Fall through to slim context

    # Continuing session (or rich context failed): use slim grounding context (~150 tokens)
    try:
        data = _http_post(
            f"{MEMORY_AGENT_URL}/api/grounding-context",
            {
                "session_id": session_id,
                "project_path": project_path,
                "user_input": user_input,
            },
        )
        if data:
            context = data.get("context", "")
            if context:
                print(context + hint)
    except Exception as e:
        logger.debug(f"Grounding context call failed: {e}")

    # Update meta (even for slim context, so we track this session)
    update_session_meta(session_id, project_path)

    sys.exit(0)


if __name__ == "__main__":
    main()
