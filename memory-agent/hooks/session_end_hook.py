#!/usr/bin/env python3
"""
SessionEnd hook for Claude Code.

Called when a Claude Code session ends. Performs final memory extraction
from the conversation transcript and cleans up the cursor file for
this session.

This script:
  1. Reads hook JSON from stdin (session_id, transcript_path, etc.)
  2. Runs final extraction via extract_memories.py (with is_session_end=True)
  3. Optionally invokes the existing session_end.py for full session wrapup
  4. Cleans up cursor state for this session
  5. Exits 0 on success OR failure (never blocks session teardown)

Timing budget: < 5 seconds total.
"""

import sys
import json
import os
import time

# Ensure the hooks directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    start = time.time()

    try:
        # Read hook data from stdin
        hook_data = {}
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
            if raw.strip():
                hook_data = json.loads(raw)

        # Ensure hook_event_name is set
        if "hook_event_name" not in hook_data:
            hook_data["hook_event_name"] = "SessionEnd"

        session_id = hook_data.get("session_id", "")
        transcript_path = hook_data.get("transcript_path", "")
        project_path = hook_data.get("cwd") or hook_data.get("project_path", "")

        # ---------------------------------------------------------------
        # Step 1: Extract memories from transcript (final pass)
        # ---------------------------------------------------------------
        if transcript_path:
            from extract_memories import run_extraction

            results = run_extraction(
                session_id=session_id,
                transcript_path=transcript_path,
                project_path=project_path,
                is_session_end=True,  # This will clean up the cursor after extraction
            )

            elapsed_extract = round(time.time() - start, 2)
            print(
                f"[SessionEnd] Extraction complete: "
                f"extracted={results['extracted']} stored={results['stored']} "
                f"errors={results['errors']} time={elapsed_extract}s",
                file=sys.stderr,
            )
        else:
            print("[SessionEnd] No transcript_path provided, skipping extraction.", file=sys.stderr)
            # Still clean up cursor if session_id is present
            if session_id:
                try:
                    from extract_memories import cleanup_cursor
                    cleanup_cursor(session_id)
                except ImportError:
                    pass

        # ---------------------------------------------------------------
        # Step 1.5: Deregister from cross-session awareness
        # ---------------------------------------------------------------
        if session_id:
            try:
                _deregister_session(session_id, project_path, timeout=2.0)
            except Exception as e:
                print(f"[SessionEnd] Session deregister failed (non-fatal): {e}", file=sys.stderr)

        # ---------------------------------------------------------------
        # Step 2: Trigger the existing session_end.py wrapup logic
        # (summarization, daily log, MEMORY.md sync, flush)
        # Only if we have time left in our budget
        # ---------------------------------------------------------------
        remaining = 5.0 - (time.time() - start)
        if remaining > 1.0 and session_id:
            try:
                _trigger_session_wrapup(session_id, project_path, timeout=remaining - 0.5)
            except Exception as e:
                print(f"[SessionEnd] Session wrapup failed (non-fatal): {e}", file=sys.stderr)

        elapsed_total = round(time.time() - start, 2)
        print(f"[SessionEnd] Complete. Total time: {elapsed_total}s", file=sys.stderr)

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"[SessionEnd] Error (non-fatal): {e} [{elapsed}s]", file=sys.stderr)

    # Always exit 0 - never block session end
    sys.exit(0)


def _deregister_session(session_id: str, project_path: str, timeout: float = 2.0):
    """Deregister this session from cross-session awareness."""
    import urllib.request
    import urllib.error

    memory_agent_url = os.getenv("MEMORY_AGENT_URL", "http://localhost:8102")

    payload = json.dumps({
        "session_id": session_id,
        "project_path": project_path,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{memory_agent_url}/api/sessions/deregister",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=min(timeout, 2.0)) as resp:
            if resp.status == 200:
                print("[SessionEnd] Session deregistered.", file=sys.stderr)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
        print(f"[SessionEnd] Session deregister API call failed: {e}", file=sys.stderr)


def _trigger_session_wrapup(session_id: str, project_path: str, timeout: float = 3.0):
    """
    Trigger the existing session_end.py summarization via the memory agent API.
    This calls key skills: daily_log_append_session, sync_memory_md, pre_compaction_flush.
    Uses a single lightweight API call rather than the full async pipeline.
    """
    import urllib.request
    import urllib.error

    memory_agent_url = os.getenv("MEMORY_AGENT_URL", "http://localhost:8102")
    api_key = os.getenv("MEMORY_API_KEY", "")

    # Call the pre_compaction_flush skill as a lightweight session wrapup
    payload = {
        "jsonrpc": "2.0",
        "method": "tasks/send",
        "params": {
            "message": {"parts": [{"type": "text", "text": ""}]},
            "metadata": {
                "skill_id": "pre_compaction_flush",
                "params": {
                    "project_path": project_path,
                    "session_id": session_id,
                }
            }
        },
        "id": f"session-end-flush-{session_id}"
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Memory-Key"] = api_key

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{memory_agent_url}/a2a",
            data=data,
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=min(timeout, 3.0)) as resp:
            if resp.status == 200:
                print(f"[SessionEnd] Flush triggered successfully.", file=sys.stderr)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
        print(f"[SessionEnd] Flush API call failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
