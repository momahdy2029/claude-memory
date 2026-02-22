#!/usr/bin/env python3
"""
Stop hook for Claude Code.

Fires after every Claude response. Unlike PreCompact/SessionEnd hooks which
scan the full transcript, this hook analyzes ONLY the latest assistant
response for high-signal content worth persisting immediately.

Design constraints:
  - Runs after EVERY response -- must complete in < 2 seconds
  - Extracts at most 2 memories per invocation
  - Focuses only on explicit, high-confidence signals (decisions, error
    resolutions, architecture notes)
  - Shares the cursor dedup hash list with extract_memories.py so the
    heavier hooks don't re-extract the same content
  - Uses stdlib only (no pip dependencies)
  - Always exits 0 -- never blocks the user

Stdin JSON schema (provided by Claude Code):
  {
    "session_id": "...",
    "transcript_path": "...",
    "hook_event_name": "Stop",
    "cwd": "...",
    "stop_hook_active": true,
    ... (assistant's last response in transcript)
  }
"""

import os
import sys
import json
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MEMORY_AGENT_URL = os.getenv("MEMORY_AGENT_URL", "http://localhost:8102")
API_KEY = os.getenv("MEMORY_API_KEY", "")
CURSOR_DIR = Path.home() / ".claude"
CURSOR_FILE = CURSOR_DIR / "memory-agent-cursor.json"
MAX_MEMORIES_PER_STOP = 2        # Hard cap -- stay fast
MAX_CONTENT_LENGTH = 500         # Truncate for storage
API_TIMEOUT_SECONDS = 1.5        # Tight timeout for API calls
TOTAL_TIME_BUDGET = 2.0          # Total wall-clock budget

# ---------------------------------------------------------------------------
# High-signal extraction patterns (intentionally narrow)
#
# These are stricter than the ones in extract_memories.py because the Stop
# hook runs on every response and must avoid false positives.  The heavier
# PreCompact/SessionEnd hooks catch the rest.
# ---------------------------------------------------------------------------

# Explicit decisions -- strong first-person phrasing
DECISION_PATTERNS = [
    re.compile(
        r"(?:^|\n)\s*(?:I decided to|I've decided to|Let's go with|The approach will be|"
        r"We(?:'ll| will) go with|The decision is to) (.{20,}?)(?:\.|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
]

# Error resolutions -- explicit fix language
ERROR_RESOLUTION_PATTERNS = [
    re.compile(
        r"(?:^|\n)\s*(?:The fix is|The fix was|Root cause was|Root cause:|"
        r"This was caused by|The bug was|The issue was|Resolution:) (.{20,}?)(?:\.|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
]

# Architecture / convention notes
ARCHITECTURE_PATTERNS = [
    re.compile(
        r"(?:^|\n)\s*(?:The architecture|This pattern|Convention:|"
        r"The convention is|Key pattern:|Architecture note:) (.{20,}?)(?:\.|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
]


# ---------------------------------------------------------------------------
# Cursor interaction (reuses same file as extract_memories.py)
# ---------------------------------------------------------------------------

def _load_cursor_hashes(session_id: str) -> set:
    """Load the set of already-extracted content hashes for this session."""
    try:
        if CURSOR_FILE.exists():
            data = json.loads(CURSOR_FILE.read_text(encoding="utf-8"))
            session = data.get(session_id, {})
            return set(session.get("extracted_hashes", []))
    except (json.JSONDecodeError, OSError):
        pass
    return set()


def _save_cursor_hashes(session_id: str, new_hashes: List[str]):
    """Append new hashes to the session's cursor entry."""
    try:
        CURSOR_DIR.mkdir(parents=True, exist_ok=True)
        data = {}
        if CURSOR_FILE.exists():
            try:
                data = json.loads(CURSOR_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}

        session = data.get(session_id, {"byte_offset": 0, "extracted_hashes": []})
        existing = set(session.get("extracted_hashes", []))
        merged = list(existing | set(new_hashes))
        # Cap to prevent unbounded growth
        if len(merged) > 200:
            merged = merged[-200:]
        session["extracted_hashes"] = merged
        data[session_id] = session

        CURSOR_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass  # Fail silently


def _content_hash(text: str) -> str:
    """Short MD5 prefix for dedup -- matches extract_memories.content_hash."""
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------

def _get_latest_response(transcript_path: str) -> str:
    """
    Read the transcript file and return only the last assistant response.

    Claude Code transcripts are JSONL where each line is a message object.
    We read the file from the end backwards to find the last assistant turn.
    For speed we only read the trailing portion of the file (last 32 KB max).
    """
    path = Path(transcript_path)
    if not path.exists():
        return ""

    try:
        file_size = path.stat().st_size
        if file_size == 0:
            return ""

        # Read at most the last 32 KB -- the latest response should be there
        read_start = max(0, file_size - 32768)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if read_start > 0:
                f.seek(read_start)
                # Skip partial line
                f.readline()
            tail = f.read()

        if not tail.strip():
            return ""

        # Walk lines in reverse to find last assistant message
        lines = tail.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                # Claude Code JSONL format: {"role": "assistant", "content": ...}
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        # Multi-part content (text blocks)
                        parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                parts.append(part.get("text", ""))
                            elif isinstance(part, str):
                                parts.append(part)
                        return "\n".join(parts)
                    elif isinstance(content, str):
                        return content
            except (json.JSONDecodeError, TypeError):
                continue

        # Fallback: if JSONL parsing fails, return last chunk of raw text
        # (transcript might be plain text rather than JSONL)
        return tail[-8192:] if len(tail) > 8192 else tail

    except OSError:
        return ""


def _extract_high_signal(text: str, existing_hashes: set) -> List[Dict[str, Any]]:
    """
    Scan text for high-signal patterns.  Returns at most MAX_MEMORIES_PER_STOP items.
    """
    extractions: List[Dict[str, Any]] = []
    seen = set(existing_hashes)

    def _try_add(content: str, mem_type: str, importance: int, tags: List[str]):
        if len(extractions) >= MAX_MEMORIES_PER_STOP:
            return
        h = _content_hash(content)
        if h in seen:
            return
        seen.add(h)
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH] + "..."
        extractions.append({
            "content": content,
            "type": mem_type,
            "importance": importance,
            "tags": tags + ["auto-extracted", "stop-hook"],
            "hash": h,
        })

    def _context_around(match_obj, source_text: str, chars: int = 200) -> str:
        """Grab surrounding context aligned to line boundaries."""
        start = max(0, match_obj.start() - chars)
        end = min(len(source_text), match_obj.end() + chars)
        while start > 0 and source_text[start] != '\n':
            start -= 1
        while end < len(source_text) and source_text[end] != '\n':
            end += 1
        return source_text[start:end].strip()

    # --- Decisions (importance 7 -- higher than extract_memories' 6 because
    #     these patterns are narrower / higher confidence) ---
    for pat in DECISION_PATTERNS:
        for m in pat.finditer(text):
            ctx = _context_around(m, text)
            if len(ctx) > 30:
                _try_add(ctx, "decision", 7, ["decision"])

    # --- Error resolutions (importance 7) ---
    for pat in ERROR_RESOLUTION_PATTERNS:
        for m in pat.finditer(text):
            ctx = _context_around(m, text)
            if len(ctx) > 30:
                _try_add(ctx, "error", 7, ["error", "resolution"])

    # --- Architecture notes (importance 6) ---
    for pat in ARCHITECTURE_PATTERNS:
        for m in pat.finditer(text):
            ctx = _context_around(m, text)
            if len(ctx) > 30:
                _try_add(ctx, "decision", 6, ["architecture", "pattern"])

    return extractions


# ---------------------------------------------------------------------------
# API call (mirrors extract_memories.store_memory_sync, tighter timeout)
# ---------------------------------------------------------------------------

def _store_memory(extraction: Dict[str, Any], project_path: Optional[str] = None) -> bool:
    """Store a single memory via the memory agent A2A endpoint."""
    import urllib.request
    import urllib.error

    payload = {
        "jsonrpc": "2.0",
        "method": "tasks/send",
        "params": {
            "message": {"parts": [{"type": "text", "text": ""}]},
            "metadata": {
                "skill_id": "store_memory",
                "params": {
                    "content": extraction["content"],
                    "type": extraction["type"],
                    "importance": extraction["importance"],
                    "tags": extraction["tags"],
                    "project_path": project_path,
                    "agent_type": "stop-hook",
                    "outcome_status": "pending",
                    "confidence": 0.45,  # Slightly above auto-extracted (0.4)
                },
            },
        },
        "id": f"stop-{extraction['hash']}-{int(time.time())}",
    }

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-Memory-Key"] = API_KEY

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{MEMORY_AGENT_URL}/a2a",
            data=data,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=API_TIMEOUT_SECONDS) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start = time.time()

    try:
        # --- Read stdin JSON ---
        hook_data: Dict[str, Any] = {}
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
            if raw.strip():
                hook_data = json.loads(raw)

        session_id = hook_data.get("session_id", "")
        transcript_path = hook_data.get("transcript_path", "")
        project_path = hook_data.get("cwd") or hook_data.get("project_path", "")

        if not transcript_path or not session_id:
            sys.exit(0)

        # --- Load existing hashes for dedup ---
        existing_hashes = _load_cursor_hashes(session_id)

        # --- Get only the latest assistant response ---
        response_text = _get_latest_response(transcript_path)
        if not response_text or len(response_text) < 40:
            sys.exit(0)

        # --- Extract high-signal content ---
        extractions = _extract_high_signal(response_text, existing_hashes)
        if not extractions:
            sys.exit(0)

        # --- Store via API (with time budget) ---
        stored_hashes: List[str] = []
        for extraction in extractions:
            elapsed = time.time() - start
            if elapsed >= TOTAL_TIME_BUDGET:
                break
            if _store_memory(extraction, project_path):
                stored_hashes.append(extraction["hash"])

        # --- Persist new hashes to cursor file ---
        if stored_hashes:
            _save_cursor_hashes(session_id, stored_hashes)

        elapsed_total = round(time.time() - start, 3)
        print(
            f"[Stop] session={session_id} "
            f"found={len(extractions)} stored={len(stored_hashes)} "
            f"elapsed={elapsed_total}s",
            file=sys.stderr,
        )

    except Exception as e:
        elapsed = round(time.time() - start, 3)
        print(f"[Stop] Error (non-fatal): {e} [{elapsed}s]", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
