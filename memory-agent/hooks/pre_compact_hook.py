#!/usr/bin/env python3
"""
PreCompact hook for Claude Code.

Called before context compaction. Extracts memories from the conversation
transcript so important information is preserved even if compaction
discards conversation turns.

This script:
  1. Reads hook JSON from stdin (session_id, transcript_path, etc.)
  2. Delegates to extract_memories.py for the actual extraction
  3. Forces a checkpoint creation before compaction
  4. Calls /api/session/brain-dump to get full session state
  5. Writes structured brain dump to native MEMORY.md
  6. Exits 0 on success OR failure (never blocks compaction)

Timing budget: < 8 seconds total (5s extraction + 3s brain dump).
"""

import sys
import json
import os
import re
import time
import hashlib
from pathlib import Path
from datetime import datetime

# Ensure the hooks directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MEMORY_AGENT_URL = os.getenv("MEMORY_AGENT_URL", "http://localhost:8102")
API_KEY = os.getenv("MEMORY_API_KEY", "")
BRAIN_DUMP_TIMEOUT = 3  # seconds

# Import the canonical slug computation from the services layer
_AGENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _AGENT_DIR)
try:
    from services.native_memory_paths import get_native_memory_md as _canonical_get_memory_md
    _HAS_NATIVE_PATHS = True
except ImportError:
    _HAS_NATIVE_PATHS = False


def _compute_project_slug(project_path: str) -> str:
    """Compute the Claude Code project slug for MEMORY.md path.

    Claude Code stores project memory at:
    ~/.claude/projects/<slug>/memory/MEMORY.md

    Algorithm (reverse-engineered from actual Claude Code directories):
        C:\\foo\\bar  ->  C--foo-bar   (drive colon+sep becomes --)
        /home/foo   ->  home-foo     (leading / stripped)
    """
    p = project_path.replace("\\", "/").rstrip("/")
    # Handle Windows drive letter: C:/ -> C--
    if len(p) >= 2 and p[1] == ":":
        drive = p[0]
        rest = p[2:].lstrip("/")
        slug = f"{drive}--{rest}"
    else:
        slug = p.lstrip("/")
    return slug.replace("/", "-").replace(" ", "-")


def _get_memory_md_path(project_path: str) -> Path:
    """Get the native MEMORY.md path for a project."""
    if _HAS_NATIVE_PATHS:
        return _canonical_get_memory_md(project_path)
    # Fallback if import failed
    slug = _compute_project_slug(project_path)
    return Path.home() / ".claude" / "projects" / slug / "memory" / "MEMORY.md"


def _force_checkpoint(session_id: str, project_path: str):
    """Force a checkpoint creation via A2A. 1s timeout, silent fail."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": f"pre-compact-checkpoint-{int(time.time())}",
        "method": "tasks/send",
        "params": {
            "message": {"parts": [{"type": "text", "text": ""}]},
            "metadata": {
                "skill_id": "checkpoint_create",
                "params": {
                    "session_id": session_id,
                    "summary": "Pre-compaction checkpoint (auto)",
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
        urllib.request.urlopen(req, timeout=1.5)
    except Exception:
        pass


def _get_brain_dump(session_id: str, project_path: str) -> dict:
    """Call /api/session/brain-dump to get full session state."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "session_id": session_id,
        "project_path": project_path,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-Memory-Key"] = API_KEY

    try:
        req = urllib.request.Request(
            f"{MEMORY_AGENT_URL}/api/session/brain-dump",
            data=payload, headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=BRAIN_DUMP_TIMEOUT) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass
    return {}


def _format_brain_dump(dump: dict) -> str:
    """Format brain dump dict into structured markdown for MEMORY.md."""
    lines = []
    timestamp = dump.get("timestamp", datetime.now().isoformat())
    ts_short = timestamp[:16].replace("T", " ")

    lines.append("<!-- SESSION-BRAIN-DUMP START -->")
    lines.append(f"## Last Session ({ts_short})")

    # Goal
    state = dump.get("state")
    if state and isinstance(state, dict):
        goal = state.get("current_goal", "")
        if goal:
            lines.append(f"**Goal:** {goal[:120]}")
        lines.append("")

    # Decisions
    if state and isinstance(state, dict):
        decisions = state.get("decisions_summary", "")
        if decisions:
            lines.append("### Decisions")
            for d in decisions.split("\n"):
                d = d.strip()
                if d:
                    if not d.startswith("- "):
                        d = f"- {d}"
                    lines.append(d)
            lines.append("")

    # Checkpoint info
    cp = dump.get("checkpoint")
    if cp and isinstance(cp, dict):
        summary = cp.get("summary", "")
        if summary:
            lines.append("### Checkpoint")
            lines.append(f"- {summary[:200]}")
            lines.append("")
        key_facts = cp.get("key_facts", [])
        if key_facts:
            lines.append("### Key Facts")
            for f in key_facts[:8]:
                lines.append(f"- {f[:100]}")
            lines.append("")

    # Workflows
    workflows = dump.get("workflows")
    if workflows and isinstance(workflows, list) and len(workflows) > 0:
        lines.append("### Learned Workflows")
        for wf in workflows[:8]:
            name = wf.get("name", "")
            cmds = wf.get("commands", [])
            steps = wf.get("steps", [])
            if cmds:
                cmd_str = " -> ".join(cmds[:4])
                lines.append(f"- **{name}**: `{cmd_str}`")
            elif steps:
                step_str = " -> ".join(steps[:3])
                lines.append(f"- **{name}**: {step_str}")
            else:
                lines.append(f"- **{name}**")
        lines.append("")

    # Key entities
    if state and isinstance(state, dict):
        entities = state.get("entity_registry") or {}
        if entities:
            lines.append("### Key Files")
            for k, v in list(entities.items())[:10]:
                lines.append(f"- {k}: `{v[:60]}`")
            lines.append("")

    # Pending items
    if state and isinstance(state, dict):
        pending = state.get("pending_questions") or []
        if pending:
            lines.append("### Pending")
            for p in pending[:5]:
                lines.append(f"- {p[:80]}")
            lines.append("")

    # Soul brief
    soul = dump.get("soul") or ""
    if soul and isinstance(soul, str) and len(soul) > 10:
        lines.append("### Soul Brief")
        lines.append(soul[:300])
        lines.append("")

    lines.append("<!-- SESSION-BRAIN-DUMP END -->")
    return "\n".join(lines)


def _write_brain_dump_to_memory_md(project_path: str, brain_dump_md: str):
    """Write (or replace) the brain dump section in native MEMORY.md.

    Preserves any existing content outside the brain dump markers.
    Creates the file and directories if they don't exist.
    """
    md_path = _get_memory_md_path(project_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    existing = ""
    if md_path.exists():
        try:
            existing = md_path.read_text(encoding="utf-8")
        except OSError:
            existing = ""

    # Replace existing brain dump section if present
    start_marker = "<!-- SESSION-BRAIN-DUMP START -->"
    end_marker = "<!-- SESSION-BRAIN-DUMP END -->"

    if start_marker in existing and end_marker in existing:
        # Replace between markers (inclusive)
        pattern = re.compile(
            re.escape(start_marker) + r".*?" + re.escape(end_marker),
            re.DOTALL,
        )
        new_content = pattern.sub(brain_dump_md, existing)
    elif existing.strip():
        # Append at top (brain dump should be first thing Claude sees)
        new_content = brain_dump_md + "\n\n" + existing
    else:
        new_content = brain_dump_md

    # Enforce 200-line limit (Claude Code truncates after 200 lines)
    content_lines = new_content.split("\n")
    if len(content_lines) > 195:
        new_content = "\n".join(content_lines[:195])

    try:
        md_path.write_text(new_content, encoding="utf-8")
    except OSError as e:
        print(f"[PreCompact] Failed to write MEMORY.md: {e}", file=sys.stderr)


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
        project_path = hook_data.get("cwd") or hook_data.get("project_path", "")

        if not transcript_path:
            print("[PreCompact] No transcript_path provided, skipping extraction.", file=sys.stderr)
            sys.exit(0)

        # 1. Force a checkpoint before compaction
        if session_id:
            _force_checkpoint(session_id, project_path)

        # 2. Run memory extraction (existing behavior)
        from extract_memories import run_extraction

        results = run_extraction(
            session_id=session_id,
            transcript_path=transcript_path,
            project_path=project_path,
            is_session_end=False,
        )

        elapsed = round(time.time() - start, 2)
        print(
            f"[PreCompact] Extraction complete: "
            f"extracted={results['extracted']} stored={results['stored']} "
            f"errors={results['errors']} total_time={elapsed}s",
            file=sys.stderr,
        )

        # 3. Brain dump: get full session state and write to MEMORY.md
        if session_id and project_path:
            dump = _get_brain_dump(session_id, project_path)
            if dump and dump.get("success"):
                brain_dump_md = _format_brain_dump(dump)
                _write_brain_dump_to_memory_md(project_path, brain_dump_md)
                elapsed2 = round(time.time() - start, 2)
                print(
                    f"[PreCompact] Brain dump written to MEMORY.md [{elapsed2}s]",
                    file=sys.stderr,
                )

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"[PreCompact] Error (non-fatal): {e} [{elapsed}s]", file=sys.stderr)

    # Always exit 0 - never block compaction
    sys.exit(0)


if __name__ == "__main__":
    main()
