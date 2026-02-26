#!/usr/bin/env python3
"""
Extract memories from conversation transcripts.

This script reads a Claude Code conversation transcript, extracts key
decisions, errors, patterns, and facts using keyword/pattern matching,
and stores them via the memory agent's HTTP API.

It tracks what has already been extracted using a cursor file so that
repeated calls (e.g., multiple PreCompact events) do not duplicate
extracted memories.

Design constraints:
  - Must complete in under 5 seconds
  - Uses simple keyword matching, NOT an LLM call
  - Fails silently (exit 0) to never block compaction or session end
  - Idempotent: cursor tracking prevents duplicate extraction
"""

import os
import sys
import json
import re
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MEMORY_AGENT_URL = os.getenv("MEMORY_AGENT_URL", "http://localhost:8102")
API_KEY = os.getenv("MEMORY_API_KEY", "")
CURSOR_DIR = Path.home() / ".claude"
CURSOR_FILE = CURSOR_DIR / "memory-agent-cursor.json"
MAX_EXTRACTION_TIME_SECONDS = 4.0  # Leave 1s headroom under the 5s budget
MAX_MEMORIES_PER_RUN = 10  # Cap to stay fast
MAX_CONTENT_LENGTH = 500  # Truncate long content for storage

# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

DECISION_PATTERNS = [
    # Explicit decision language
    re.compile(r"(?:^|\n)\s*(?:I |We |Let's |Going to )?(?:decided|decide) (?:to |that )(.*?)(?:\.|$)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|\n)\s*(?:Let's use|Going with|Chose|Choosing|Will use|Using|Went with) (.*?)(?:\.|$)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|\n)\s*(?:The approach|The plan|The strategy|The solution) (?:is|will be) (.*?)(?:\.|$)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|\n)\s*(?:I'll implement|We'll implement|Implementing) (.*?)(?:\.|$)", re.IGNORECASE | re.MULTILINE),
]

ERROR_PATTERNS = [
    # Error/bug language
    re.compile(r"(?:^|\n)\s*(?:Error|ERROR|Bug|BUG|ISSUE|Issue|PROBLEM|Problem|CRITICAL|FATAL)[:\s]+(.*?)(?:\n|$)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|\n)\s*(?:Fixed|Fixing|Fix for|Resolved|Resolution)[:\s]+(.*?)(?:\n|$)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:Traceback|Exception|raise \w+Error)(.*?)(?:\n\n|\Z)", re.DOTALL),
    re.compile(r"(?:^|\n)\s*(?:Root cause|The bug was|The issue was|The problem was)[:\s]+(.*?)(?:\.|$)", re.IGNORECASE | re.MULTILINE),
]

PATTERN_PATTERNS = [
    # Architecture/pattern language
    re.compile(r"(?:^|\n)\s*(?:The pattern|A pattern|Pattern)[:\s]+(.*?)(?:\.|$)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|\n)\s*(?:The approach|Best practice|Convention|Architecture)[:\s]+(.*?)(?:\.|$)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|\n)\s*(?:Always|Never|Should always|Should never|Must always|Must never) (.*?)(?:\.|$)", re.IGNORECASE | re.MULTILINE),
]

# Workflow/procedure patterns
WORKFLOW_PATTERNS = [
    # "To build X, run Y" / "To deploy X, do Y"
    re.compile(r"(?:^|\n)\s*(?:To|to) (\w[\w\s]{3,30}),\s*(?:run|do|use|execute|type) (.{10,}?)(?:\n|$)", re.IGNORECASE | re.MULTILINE),
    # "learned how to..."
    re.compile(r"(?:^|\n)\s*(?:I learned|We learned|learned how to|figured out how to) (.{20,}?)(?:\.|$)", re.IGNORECASE | re.MULTILINE),
    # Step-by-step: "1. ...\n2. ...\n3. ..."
    re.compile(r"(?:^|\n)\s*1[.)]\s+(.+)\n\s*2[.)]\s+(.+)\n\s*3[.)]\s+(.+)", re.MULTILINE),
    # "The workflow is..." / "The process is..."
    re.compile(r"(?:^|\n)\s*(?:The workflow|The process|The procedure|Steps to) (?:is|are|for)[:\s]+(.*?)(?:\n\n|\Z)", re.IGNORECASE | re.DOTALL),
]

# Broader keyword triggers (used for line-level scanning)
DECISION_KEYWORDS = {"decided", "let's use", "going with", "chose", "choosing", "will use", "the plan is", "approach is", "strategy is", "i'll implement", "we'll implement"}
ERROR_KEYWORDS = {"error", "bug", "fix", "issue", "traceback", "exception", "failed", "failure", "broken", "crash", "root cause"}
PATTERN_KEYWORDS = {"pattern", "approach", "architecture", "convention", "best practice", "always", "never", "rule"}
WORKFLOW_KEYWORDS = {"workflow", "procedure", "steps to", "how to", "process for", "pipeline", "build steps", "deploy steps"}


# ---------------------------------------------------------------------------
# Cursor management - tracks what we already extracted
# ---------------------------------------------------------------------------

def load_cursor(session_id: str) -> Dict[str, Any]:
    """Load the extraction cursor for a session."""
    try:
        if CURSOR_FILE.exists():
            data = json.loads(CURSOR_FILE.read_text(encoding="utf-8"))
            return data.get(session_id, {"byte_offset": 0, "extracted_hashes": []})
    except (json.JSONDecodeError, OSError):
        pass
    return {"byte_offset": 0, "extracted_hashes": []}


def save_cursor(session_id: str, cursor: Dict[str, Any]):
    """Save the extraction cursor for a session."""
    try:
        CURSOR_DIR.mkdir(parents=True, exist_ok=True)
        data = {}
        if CURSOR_FILE.exists():
            try:
                data = json.loads(CURSOR_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}

        data[session_id] = cursor

        # Prune old sessions (keep last 20)
        if len(data) > 20:
            sorted_keys = sorted(data.keys())
            for old_key in sorted_keys[:-20]:
                del data[old_key]

        CURSOR_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass  # Fail silently


def cleanup_cursor(session_id: str):
    """Remove cursor data for a completed session."""
    try:
        if CURSOR_FILE.exists():
            data = json.loads(CURSOR_FILE.read_text(encoding="utf-8"))
            if session_id in data:
                del data[session_id]
                CURSOR_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except (json.JSONDecodeError, OSError):
        pass


def content_hash(text: str) -> str:
    """Create a short hash to deduplicate extracted content."""
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Transcript reading
# ---------------------------------------------------------------------------

def read_transcript(transcript_path: str, byte_offset: int = 0) -> Tuple[str, int]:
    """
    Read the transcript file from the given byte offset.
    Returns (new_text, new_byte_offset).
    """
    path = Path(transcript_path)
    if not path.exists():
        return "", byte_offset

    try:
        file_size = path.stat().st_size
        if file_size <= byte_offset:
            return "", byte_offset

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(byte_offset)
            text = f.read()
            new_offset = f.tell()

        return text, new_offset
    except OSError:
        return "", byte_offset


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------

def extract_context_around(text: str, match_start: int, match_end: int, context_chars: int = 200) -> str:
    """Get surrounding context for a match to make the extraction more useful."""
    start = max(0, match_start - context_chars)
    end = min(len(text), match_end + context_chars)

    # Try to align to line boundaries
    while start > 0 and text[start] != '\n':
        start -= 1
    while end < len(text) and text[end] != '\n':
        end += 1

    return text[start:end].strip()


def extract_from_text(text: str, existing_hashes: set) -> List[Dict[str, Any]]:
    """
    Extract memories from transcript text using keyword/pattern matching.
    Returns a list of extracted memory dicts.
    """
    extractions = []
    seen_hashes = set(existing_hashes)

    def add_extraction(content: str, memory_type: str, importance: int, tags: List[str]):
        """Add an extraction if not already seen."""
        if len(extractions) >= MAX_MEMORIES_PER_RUN:
            return
        h = content_hash(content)
        if h in seen_hashes:
            return
        seen_hashes.add(h)
        # Truncate content
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH] + "..."
        extractions.append({
            "content": content,
            "type": memory_type,
            "importance": importance,
            "tags": tags + ["auto-extracted", "hook"],
            "hash": h,
        })

    # --- Regex-based extraction ---

    # Decisions
    for pattern in DECISION_PATTERNS:
        for match in pattern.finditer(text):
            context = extract_context_around(text, match.start(), match.end())
            if len(context) > 30:  # Skip very short matches
                add_extraction(context, "decision", 6, ["decision"])

    # Errors
    for pattern in ERROR_PATTERNS:
        for match in pattern.finditer(text):
            context = extract_context_around(text, match.start(), match.end())
            if len(context) > 30:
                add_extraction(context, "error", 7, ["error"])

    # Patterns
    for pattern in PATTERN_PATTERNS:
        for match in pattern.finditer(text):
            context = extract_context_around(text, match.start(), match.end())
            if len(context) > 30:
                add_extraction(context, "code", 6, ["pattern"])

    # Workflows / Procedures
    for pattern in WORKFLOW_PATTERNS:
        for match in pattern.finditer(text):
            context = extract_context_around(text, match.start(), match.end(), context_chars=300)
            if len(context) > 40:
                add_extraction(context, "code", 7, ["workflow", "procedure"])

    # --- Line-level keyword scanning (fallback for cases regex misses) ---
    # Only do this if we have not yet hit our cap
    if len(extractions) < MAX_MEMORIES_PER_RUN:
        lines = text.split('\n')
        i = 0
        while i < len(lines) and len(extractions) < MAX_MEMORIES_PER_RUN:
            line_lower = lines[i].lower().strip()

            # Skip very short or empty lines
            if len(line_lower) < 20:
                i += 1
                continue

            # Check for decision keywords
            if any(kw in line_lower for kw in DECISION_KEYWORDS):
                # Grab this line plus next 2 for context
                block = '\n'.join(lines[i:i+3]).strip()
                if len(block) > 30:
                    add_extraction(block, "decision", 5, ["decision", "keyword-match"])

            # Check for error keywords
            elif any(kw in line_lower for kw in ERROR_KEYWORDS):
                block = '\n'.join(lines[i:i+3]).strip()
                if len(block) > 30:
                    add_extraction(block, "error", 6, ["error", "keyword-match"])

            # Check for pattern keywords
            elif any(kw in line_lower for kw in PATTERN_KEYWORDS):
                block = '\n'.join(lines[i:i+3]).strip()
                if len(block) > 30:
                    add_extraction(block, "code", 5, ["pattern", "keyword-match"])

            # Check for workflow keywords
            elif any(kw in line_lower for kw in WORKFLOW_KEYWORDS):
                block = '\n'.join(lines[i:i+5]).strip()  # Wider context for workflows
                if len(block) > 40:
                    add_extraction(block, "code", 6, ["workflow", "keyword-match"])

            i += 1

    return extractions


# ---------------------------------------------------------------------------
# API calls to memory agent
# ---------------------------------------------------------------------------

def store_memory_sync(extraction: Dict[str, Any], project_path: Optional[str] = None) -> bool:
    """
    Store a single extracted memory via the memory agent API.
    Uses urllib to avoid requiring httpx/requests for the hook scripts.
    """
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
                    "agent_type": "hook-extractor",
                    "outcome_status": "pending",
                    "confidence": 0.4,  # Lower confidence for auto-extracted
                }
            }
        },
        "id": f"extract-{extraction['hash']}-{int(time.time())}"
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
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        return False


# ---------------------------------------------------------------------------
# Workflow / Bash command extraction from JSONL transcript
# ---------------------------------------------------------------------------

def extract_bash_commands(transcript_path: str, byte_offset: int = 0) -> List[str]:
    """Extract successful bash commands from JSONL transcript.

    Looks for tool_use blocks with tool=Bash that were followed by success results.
    Returns deduplicated command list.
    """
    path = Path(transcript_path)
    if not path.exists():
        return []

    commands = []
    seen = set()
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if byte_offset > 0:
                f.seek(byte_offset)
                f.readline()  # skip partial line
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    content = msg.get("content", [])
                    if not isinstance(content, list):
                        continue
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "tool_use" and part.get("name") == "Bash":
                            cmd = ""
                            inp = part.get("input", {})
                            if isinstance(inp, dict):
                                cmd = inp.get("command", "")
                            if cmd and len(cmd) > 5 and cmd not in seen:
                                # Skip trivial commands
                                if not cmd.strip().startswith(("ls", "pwd", "echo", "cat ")):
                                    seen.add(cmd)
                                    commands.append(cmd)
                except (json.JSONDecodeError, TypeError):
                    continue
    except OSError:
        pass
    return commands[-20:]  # Keep last 20 commands


def capture_workflow_sync(name: str, steps: List[str], commands: List[str],
                          project_path: Optional[str] = None) -> bool:
    """POST a captured workflow to /api/workflow/capture."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "name": name,
        "steps": steps,
        "commands": commands,
        "project_path": project_path or "",
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-Memory-Key"] = API_KEY

    try:
        req = urllib.request.Request(
            f"{MEMORY_AGENT_URL}/api/workflow/capture",
            data=payload, headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_extraction(session_id: str, transcript_path: str, project_path: Optional[str] = None, is_session_end: bool = False) -> Dict[str, Any]:
    """
    Main extraction function.

    Args:
        session_id: The session identifier
        transcript_path: Path to the conversation transcript file
        project_path: Optional project path for memory context
        is_session_end: If True, clean up cursor after extraction

    Returns:
        Summary dict with extraction results
    """
    start_time = time.time()
    results = {
        "extracted": 0,
        "stored": 0,
        "skipped_duplicate": 0,
        "errors": 0,
        "elapsed_seconds": 0,
    }

    # Load cursor state
    cursor = load_cursor(session_id)
    byte_offset = cursor.get("byte_offset", 0)
    existing_hashes = set(cursor.get("extracted_hashes", []))

    # Read new transcript content
    new_text, new_offset = read_transcript(transcript_path, byte_offset)
    if not new_text:
        results["elapsed_seconds"] = time.time() - start_time
        if is_session_end:
            cleanup_cursor(session_id)
        return results

    # Extract memories from text
    extractions = extract_from_text(new_text, existing_hashes)
    results["extracted"] = len(extractions)

    # Store each extraction via API (with time budget)
    stored_hashes = []
    for extraction in extractions:
        # Check time budget
        elapsed = time.time() - start_time
        if elapsed >= MAX_EXTRACTION_TIME_SECONDS:
            break

        success = store_memory_sync(extraction, project_path)
        if success:
            results["stored"] += 1
            stored_hashes.append(extraction["hash"])
        else:
            results["errors"] += 1

    # Update cursor
    all_hashes = list(existing_hashes | set(stored_hashes))
    # Keep only the last 200 hashes to prevent unbounded growth
    if len(all_hashes) > 200:
        all_hashes = all_hashes[-200:]

    cursor = {
        "byte_offset": new_offset,
        "extracted_hashes": all_hashes,
        "last_run": datetime.now().isoformat(),
    }

    if is_session_end:
        # Final save then cleanup
        save_cursor(session_id, cursor)
        cleanup_cursor(session_id)
    else:
        save_cursor(session_id, cursor)

    results["elapsed_seconds"] = round(time.time() - start_time, 2)
    return results


def main():
    """Entry point: reads hook JSON from stdin."""
    try:
        hook_data = {}
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
            if raw.strip():
                hook_data = json.loads(raw)

        session_id = hook_data.get("session_id", f"unknown-{int(time.time())}")
        transcript_path = hook_data.get("transcript_path", "")
        project_path = hook_data.get("cwd") or hook_data.get("project_path", "")
        hook_event = hook_data.get("hook_event_name", "")
        is_session_end = hook_event == "SessionEnd"

        if not transcript_path:
            # No transcript path provided - nothing to extract
            sys.exit(0)

        results = run_extraction(
            session_id=session_id,
            transcript_path=transcript_path,
            project_path=project_path,
            is_session_end=is_session_end,
        )

        # Output summary to stderr (stdout is reserved for hook output)
        print(
            f"[MemoryExtractor] session={session_id} event={hook_event} "
            f"extracted={results['extracted']} stored={results['stored']} "
            f"errors={results['errors']} elapsed={results['elapsed_seconds']}s",
            file=sys.stderr,
        )

    except Exception as e:
        # Fail silently - never block the user's workflow
        print(f"[MemoryExtractor] Error: {e}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
