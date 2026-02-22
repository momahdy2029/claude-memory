#!/usr/bin/env python3
"""
PreCompact hook for Claude Code.

Called before context compaction. Extracts memories from the conversation
transcript so important information is preserved even if compaction
discards conversation turns.

This script:
  1. Reads hook JSON from stdin (session_id, transcript_path, etc.)
  2. Delegates to extract_memories.py for the actual extraction
  3. Exits 0 on success OR failure (never blocks compaction)

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
            hook_data["hook_event_name"] = "PreCompact"

        session_id = hook_data.get("session_id", "")
        transcript_path = hook_data.get("transcript_path", "")

        if not transcript_path:
            # No transcript available, nothing to extract
            print("[PreCompact] No transcript_path provided, skipping extraction.", file=sys.stderr)
            sys.exit(0)

        # Import and run extraction
        from extract_memories import run_extraction

        results = run_extraction(
            session_id=session_id,
            transcript_path=transcript_path,
            project_path=hook_data.get("cwd") or hook_data.get("project_path", ""),
            is_session_end=False,
        )

        elapsed = round(time.time() - start, 2)
        print(
            f"[PreCompact] Extraction complete: "
            f"extracted={results['extracted']} stored={results['stored']} "
            f"errors={results['errors']} total_time={elapsed}s",
            file=sys.stderr,
        )

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"[PreCompact] Error (non-fatal): {e} [{elapsed}s]", file=sys.stderr)

    # Always exit 0 - never block compaction
    sys.exit(0)


if __name__ == "__main__":
    main()
