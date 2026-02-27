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
RESPONSE_COUNTER_FILE = CURSOR_DIR / "memory-agent-response-counter.json"
MAX_MEMORIES_PER_STOP = 2        # Hard cap -- stay fast
MAX_CONTENT_LENGTH = 500         # Truncate for storage
API_TIMEOUT_SECONDS = 1.5        # Tight timeout for API calls
TOTAL_TIME_BUDGET = 2.0          # Total wall-clock budget
AUTO_CHECKPOINT_INTERVAL = 15    # Fire checkpoint every N responses

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
# Soul fragment capture (lightweight regex → HTTP POST)
# ---------------------------------------------------------------------------

SOUL_PATTERNS = {
    "decision_made": [
        re.compile(
            r"(?:let's|we'll|going to|chose to|decided to)\s+(?:use|go with|implement|try)\s+(.+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:using|choosing|picked)\s+(\S+)\s+(?:because|since|for)",
            re.IGNORECASE,
        ),
    ],
    "preference_expressed": [
        re.compile(
            r"(?:I prefer|you should always|always use|never use|don't use|I like to)\s+(.+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:remember to|make sure to|don't forget to)\s+(.+)",
            re.IGNORECASE,
        ),
    ],
    "error_resolved": [
        re.compile(
            r"(?:fixed|resolved|solved|the issue was|root cause)\s*:?\s*(.+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:the (?:fix|solution) (?:was|is))\s+(.+)",
            re.IGNORECASE,
        ),
    ],
    "pattern_used": [
        re.compile(
            r"(?:same (?:approach|pattern|method) as)\s+(.+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:like we did (?:for|in|with))\s+(.+)",
            re.IGNORECASE,
        ),
    ],
    "correction_received": [
        re.compile(
            r"(?:no,?\s+(?:actually|that's wrong|not like that)|(?:don't|stop)\s+(?:do|doing)\s+that)\s*[,:]?\s*(.+)",
            re.IGNORECASE,
        ),
    ],
}


def _capture_soul_fragments(
    text: str, session_id: str, project_path: str
):
    """Extract and POST soul fragments from response text.

    Runs regex extraction, then fires HTTP POST for each fragment.
    Budget: < 200ms total. Non-blocking — failures are silent.
    """
    import urllib.request
    import urllib.error

    fragments = []
    seen = set()

    for fragment_type, patterns in SOUL_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                captured = match.group(1).strip() if match.lastindex else match.group(0).strip()
                if len(captured) < 10 or len(captured) > 300:
                    continue
                key = captured[:50].lower()
                if key in seen:
                    continue
                seen.add(key)
                fragments.append({
                    "fragment_type": fragment_type,
                    "content": captured[:300],
                })

    if not fragments:
        return

    # POST each fragment (fire-and-forget, tight timeout)
    captured_count = 0
    for frag in fragments[:5]:  # Cap at 5 fragments per response
        try:
            payload = json.dumps({
                "session_id": session_id,
                "project_path": project_path,
                "fragment_type": frag["fragment_type"],
                "content": frag["content"],
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{MEMORY_AGENT_URL}/api/soul/capture",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                if resp.status == 200:
                    captured_count += 1
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
            pass  # Silent failure — don't block the hook

    if captured_count > 0:
        print(
            f"[Stop] Soul fragments captured: {captured_count}/{len(fragments)}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Self-reported learning extraction (<!-- LEARNED: ... --> tags)
# ---------------------------------------------------------------------------

_LEARNED_TAG_RE = re.compile(
    r"<!--\s*LEARNED:\s*(.+?)\s*-->",
    re.IGNORECASE | re.DOTALL,
)


def _extract_self_reported_learnings(
    text: str, existing_hashes: set
) -> List[Dict[str, Any]]:
    """Extract <!-- LEARNED: ... --> tags that Claude self-reported.

    These are high-confidence (Claude chose to report them) so they get
    importance=8 and confidence=0.75 — higher than regex-guessed content.
    """
    extractions = []
    seen = set(existing_hashes)

    for m in _LEARNED_TAG_RE.finditer(text):
        content = m.group(1).strip()
        if len(content) < 10 or len(content) > 500:
            continue
        h = _content_hash(content)
        if h in seen:
            continue
        seen.add(h)
        extractions.append({
            "content": content,
            "type": "decision",
            "importance": 8,
            "tags": ["self-reported", "learned", "stop-hook"],
            "hash": h,
        })

    return extractions[:3]  # Cap at 3 per response


# ---------------------------------------------------------------------------
# Response counter + auto-checkpoint (fire-and-forget)
# ---------------------------------------------------------------------------

def _increment_response_counter(session_id: str) -> int:
    """Increment and return the response count for this session."""
    try:
        CURSOR_DIR.mkdir(parents=True, exist_ok=True)
        data = {}
        if RESPONSE_COUNTER_FILE.exists():
            try:
                data = json.loads(RESPONSE_COUNTER_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
        count = data.get(session_id, 0) + 1
        data[session_id] = count
        # Prune old sessions (keep 10)
        if len(data) > 10:
            for old_key in sorted(data.keys())[:-10]:
                del data[old_key]
        RESPONSE_COUNTER_FILE.write_text(json.dumps(data), encoding="utf-8")
        return count
    except OSError:
        return 0


def _fire_auto_checkpoint(session_id: str, project_path: str):
    """Fire-and-forget checkpoint creation (1s timeout, silent fail)."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": f"auto-checkpoint-{session_id}-{int(time.time())}",
        "method": "tasks/send",
        "params": {
            "message": {"parts": [{"type": "text", "text": ""}]},
            "metadata": {
                "skill_id": "checkpoint_create",
                "params": {
                    "session_id": session_id,
                    "summary": f"Auto-checkpoint (periodic, every {AUTO_CHECKPOINT_INTERVAL} responses)",
                },
            },
        },
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-Memory-Key"] = API_KEY

    try:
        req = urllib.request.Request(
            f"{MEMORY_AGENT_URL}/a2a",
            data=payload, headers=headers, method="POST",
        )
        urllib.request.urlopen(req, timeout=1.0)
    except Exception:
        pass  # Fire-and-forget


# ---------------------------------------------------------------------------
# Workflow pattern extraction (lightweight, runs on response text)
# ---------------------------------------------------------------------------

WORKFLOW_EXTRACT_PATTERNS = [
    re.compile(
        r"(?:^|\n)\s*(?:To|to) (\w[\w\s]{3,30}),\s*(?:run|do|use|execute) (.{10,}?)(?:\n|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"(?:^|\n)\s*(?:I learned|We learned|learned how to|figured out how to) (.{20,}?)(?:\.|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
]


def _extract_and_capture_workflows(text: str, project_path: str):
    """Extract workflow patterns from response text and POST to /api/workflow/capture.

    Budget: < 200ms total. Non-blocking.
    """
    import urllib.request
    import urllib.error

    for pattern in WORKFLOW_EXTRACT_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()
            if len(groups) >= 2:
                name = groups[0].strip()[:60]
                steps = [g.strip() for g in groups[1:] if g]
            elif len(groups) == 1:
                name = groups[0].strip()[:60]
                steps = [groups[0].strip()]
            else:
                continue

            if len(name) < 5:
                continue

            payload = json.dumps({
                "name": name,
                "steps": steps,
                "commands": [],
                "project_path": project_path,
            }).encode("utf-8")

            try:
                req = urllib.request.Request(
                    f"{MEMORY_AGENT_URL}/api/workflow/capture",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=0.5)
            except Exception:
                pass  # Silent failure
            break  # Only capture first workflow match per response


# ---------------------------------------------------------------------------
# LLM-judged learning extraction (fire-and-forget via memory agent)
# ---------------------------------------------------------------------------

def _fire_llm_extraction(response_text: str, session_id: str, project_path: str):
    """Fire-and-forget POST to /api/extract-learnings.

    The memory agent handles the slow OpenClaw call asynchronously.
    This function returns immediately (non-blocking, ~50ms timeout).
    """
    import urllib.request
    import urllib.error

    # Skip very short responses (not worth analyzing)
    if len(response_text) < 200:
        return

    payload = json.dumps({
        "response_text": response_text[:4000],  # Truncate for transit
        "session_id": session_id,
        "project_path": project_path,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{MEMORY_AGENT_URL}/api/extract-learnings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Very short timeout — we just want to hand off the request.
        # The server processes it asynchronously.
        urllib.request.urlopen(req, timeout=0.5)
    except Exception:
        pass  # Fire-and-forget — never block the hook


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

        # --- Increment response counter + auto-checkpoint ---
        response_count = _increment_response_counter(session_id)
        if response_count > 0 and response_count % AUTO_CHECKPOINT_INTERVAL == 0:
            _fire_auto_checkpoint(session_id, project_path)

        # --- Load existing hashes for dedup ---
        existing_hashes = _load_cursor_hashes(session_id)

        # --- Get only the latest assistant response ---
        response_text = _get_latest_response(transcript_path)
        if not response_text or len(response_text) < 40:
            sys.exit(0)

        # --- Extract self-reported learnings first (highest signal) ---
        self_reported = _extract_self_reported_learnings(response_text, existing_hashes)

        # --- Extract high-signal content (regex-based) ---
        extractions = _extract_high_signal(response_text, existing_hashes)

        # --- Store via API (with time budget) ---
        # Self-reported learnings go first (higher priority)
        stored_hashes: List[str] = []
        all_to_store = self_reported + extractions

        if all_to_store:
            for extraction in all_to_store:
                elapsed = time.time() - start
                if elapsed >= TOTAL_TIME_BUDGET:
                    break
                if _store_memory(extraction, project_path):
                    stored_hashes.append(extraction["hash"])

            # --- Persist new hashes to cursor file ---
            if stored_hashes:
                _save_cursor_hashes(session_id, stored_hashes)

        # --- Workflow extraction (lightweight, adds ~100ms) ---
        elapsed = time.time() - start
        if elapsed < TOTAL_TIME_BUDGET - 0.3:
            _extract_and_capture_workflows(response_text, project_path)

        # --- Soul fragment capture (lightweight, adds ~100ms) ---
        elapsed = time.time() - start
        if elapsed < TOTAL_TIME_BUDGET - 0.2:
            _capture_soul_fragments(
                response_text, session_id, project_path
            )

        # --- LLM-judged learning extraction (fire-and-forget to memory agent) ---
        # This calls OpenClaw asynchronously via the server; does NOT block the hook
        _fire_llm_extraction(response_text, session_id, project_path)

        elapsed_total = round(time.time() - start, 3)
        print(
            f"[Stop] session={session_id} "
            f"learned={len(self_reported)} found={len(extractions)} stored={len(stored_hashes)} "
            f"responses={response_count} "
            f"elapsed={elapsed_total}s",
            file=sys.stderr,
        )

    except Exception as e:
        elapsed = round(time.time() - start, 3)
        print(f"[Stop] Error (non-fatal): {e} [{elapsed}s]", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
