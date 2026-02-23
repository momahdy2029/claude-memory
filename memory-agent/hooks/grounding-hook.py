#!/usr/bin/env python3
"""
Grounding Hook for Claude Code - Automatic Context Injection

This script is called by Claude Code's UserPromptSubmit hook.
It fetches the current session context and outputs it to stdout,
which Claude Code automatically injects into Claude's context.

This is the REAL anti-hallucination layer - automatic, not relying on Claude to call tools.

Moltbot-inspired additions:
- Checks flush conditions (events > 50 or time > 30min)
- Loads MEMORY.md content into grounding context
- Loads today's daily log highlights
"""

import os
import sys
import json
import logging
import requests
from pathlib import Path
from typing import Any, Optional

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


def safe_get(data: Any, *keys, default: Any = None) -> Any:
    """
    Safely navigate nested data structures (dicts and lists).

    Args:
        data: The data structure to navigate
        *keys: Keys (str for dict) or indices (int for list) to traverse
        default: Value to return if path doesn't exist

    Returns:
        The value at the path, or default if not found

    Example:
        safe_get(result, "result", "artifacts", 0, "parts", 0, "text")
    """
    for key in keys:
        if data is None:
            return default
        if isinstance(data, dict):
            data = data.get(key, default)
        elif isinstance(data, list) and isinstance(key, int):
            if 0 <= key < len(data):
                data = data[key]
            else:
                return default
        else:
            return default
    return data


def get_project_path():
    """Get current working directory as project path."""
    return os.getcwd()


def load_session_data():
    """Load session data from JSON file."""
    session_file = Path(get_project_path()) / ".claude_session"
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


def save_session_data(data: dict):
    """Save session data to JSON file."""
    session_file = Path(get_project_path()) / ".claude_session"
    try:
        session_file.write_text(json.dumps(data, indent=2))
    except (IOError, OSError) as e:
        logger.warning(f"Failed to save session data: {e}")


def get_session_id():
    """Get or create session ID from environment or file."""
    # Try environment variable first
    session_id = os.getenv("CLAUDE_SESSION_ID")
    if session_id:
        return session_id

    # Try session file in project
    data = load_session_data()
    return data.get("session_id") if data else None


def call_memory_agent(skill_id: str, params: dict) -> Optional[dict]:
    """Call the memory agent API."""
    try:
        response = requests.post(
            f"{MEMORY_AGENT_URL}/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "grounding-hook",
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
        result = response.json()

        # Safely extract the artifact text using safe_get
        artifact_text = safe_get(result, "result", "artifacts", 0, "parts", 0, "text")
        if artifact_text:
            try:
                return json.loads(artifact_text)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse artifact text as JSON for skill '{skill_id}': {e}")
                return None
        return None

    except requests.RequestException as e:
        # Silently fail - don't break Claude Code if memory agent is down
        logger.debug(f"Memory agent request failed for skill '{skill_id}': {e}")
        return None
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to decode memory agent response for skill '{skill_id}': {e}")
        return None


def format_grounding_context(context: dict) -> str:
    """Format the grounding context for injection."""
    if not context or not context.get("success"):
        return ""

    grounding = context.get("grounding", {})

    lines = ["[GROUNDING CONTEXT - VERIFY BEFORE RESPONDING]"]

    # Current goal
    if grounding.get("current_goal"):
        lines.append(f"CURRENT GOAL: {grounding['current_goal']}")

    # Entity registry
    registry = grounding.get("entity_registry", {})
    if registry:
        lines.append("ENTITY REGISTRY (use these exact references):")
        for key, value in list(registry.items())[:5]:
            lines.append(f"  - {key}: {value}")

    # Anchors (verified facts)
    anchors = grounding.get("anchors", [])
    if anchors:
        lines.append("ANCHORS (verified facts - DO NOT CONTRADICT):")
        for anchor in anchors[:5]:
            lines.append(f"  - {anchor}")

    # Recent decisions
    decisions = grounding.get("decisions", [])
    if decisions:
        lines.append("RECENT DECISIONS:")
        for decision in decisions[:3]:
            lines.append(f"  - {decision}")

    # Recent events
    events = grounding.get("recent_events", [])
    if events:
        lines.append("RECENT EVENTS:")
        for event in events[:5]:
            lines.append(f"  - [{event.get('type', '?')}] {event.get('summary', '')}")

    # Contradictions warning
    contradictions = grounding.get("contradictions", [])
    if contradictions:
        lines.append("WARNING - POTENTIAL CONTRADICTIONS DETECTED:")
        for c in contradictions[:3]:
            lines.append(f"  - {c.get('content', '')[:100]}")

    # Pending questions
    pending = grounding.get("pending_questions", [])
    if pending:
        lines.append("PENDING QUESTIONS:")
        for q in pending[:3]:
            lines.append(f"  - {q}")

    lines.append("[/GROUNDING CONTEXT]")
    lines.append("")  # Empty line after

    return "\n".join(lines)


def format_memory_md_context(memory_md: dict) -> str:
    """Format MEMORY.md content for injection."""
    if not memory_md or not memory_md.get("exists"):
        return ""

    summary = memory_md.get("summary", "")
    if not summary:
        return ""

    lines = ["[CORE FACTS from MEMORY.md]"]
    lines.append(summary)
    lines.append("[/CORE FACTS]")
    lines.append("")

    return "\n".join(lines)


def format_daily_highlights(highlights: dict) -> str:
    """Format daily log highlights for injection."""
    if not highlights or not highlights.get("highlights"):
        return ""

    entries = highlights.get("highlights", [])
    if not entries:
        return ""

    lines = ["[TODAY'S HIGHLIGHTS from Daily Log]"]
    for entry in entries[:5]:
        lines.append(f"  - {entry}")
    lines.append("[/TODAY'S HIGHLIGHTS]")
    lines.append("")

    return "\n".join(lines)


def format_curator_context(curator_summary: dict, curator_status: dict) -> str:
    """Format curator context for injection."""
    if not curator_summary and not curator_status:
        return ""

    lines = ["[CURATOR CONTEXT]"]

    # Knowledge graph summary
    if curator_summary:
        context = curator_summary.get("context", "")
        if context:
            lines.append("Relevant Knowledge:")
            for line in context.split("\n")[:10]:
                if line.strip():
                    lines.append(f"  {line}")

        # Graph relationships
        graph_context = curator_summary.get("graph_context")
        if graph_context and graph_context.get("summary"):
            lines.append("")
            lines.append(f"Graph: {graph_context['summary']}")

        # Pending reviews
        pending = curator_summary.get("pending_reviews", {})
        if pending.get("total_pending", 0) > 0:
            lines.append("")
            lines.append("Pending Reviews:")
            if pending.get("duplicate_clusters", 0) > 0:
                lines.append(f"  - {pending['duplicate_clusters']} duplicate clusters")
            if pending.get("suggested_links", 0) > 0:
                lines.append(f"  - {pending['suggested_links']} suggested links")
            if pending.get("orphan_memories", 0) > 0:
                lines.append(f"  - {pending['orphan_memories']} orphan memories")

    # Curator status summary
    if curator_status:
        orphan_count = curator_status.get("orphan_count", 0)
        connection_ratio = curator_status.get("connection_ratio", 0)
        if orphan_count > 10:
            lines.append(f"Warning: {orphan_count} orphan memories need linking")
        if connection_ratio < 0.5:
            lines.append(f"Note: Low graph connectivity ({connection_ratio:.1%})")

    lines.append("[/CURATOR CONTEXT]")
    lines.append("")

    return "\n".join(lines)


def call_rest_api(method: str, path: str, json_body: dict = None, params: dict = None, timeout: int = 3):
    """Call the memory agent REST API (for endpoints that aren't A2A skills)."""
    try:
        url = f"{MEMORY_AGENT_URL}{path}"
        if method == "POST":
            response = requests.post(url, json=json_body, timeout=timeout)
        else:
            response = requests.get(url, params=params, timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException as e:
        logger.debug(f"REST API call failed ({path}): {e}")
    return None


def format_parallel_sessions(heartbeat_result: dict) -> str:
    """Format parallel session info for context injection."""
    if not heartbeat_result or not heartbeat_result.get("success"):
        return ""

    siblings = heartbeat_result.get("active_siblings", [])
    conflicts = heartbeat_result.get("file_conflicts", [])

    if not siblings:
        return ""

    lines = ["[PARALLEL SESSIONS]"]

    # Build a set of conflicting files per session for quick lookup
    conflict_map = {}
    for c in conflicts:
        conflict_map[c["session_id"]] = c.get("conflicting_files", [])

    for sib in siblings:
        label = sib.get("session_label") or sib.get("session_id", "")[:12]
        status = sib.get("status", "active")
        goal = sib.get("current_goal", "")
        files = sib.get("files_modified", [])
        decisions = sib.get("key_decisions", [])

        # Calculate time since last heartbeat
        last_hb = sib.get("last_heartbeat", "")
        time_ago = ""
        if last_hb:
            try:
                from datetime import datetime, timezone
                hb_time = datetime.fromisoformat(last_hb.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc) if hb_time.tzinfo else datetime.now()
                delta = now - hb_time
                minutes = int(delta.total_seconds() / 60)
                time_ago = f" ({minutes}m ago)" if minutes > 0 else " (just now)"
            except (ValueError, TypeError):
                pass

        lines.append(f'Session "{label}" ({status}{time_ago}):')

        if goal:
            lines.append(f"  Working on: {goal}")

        if files:
            shown = files[:5]
            lines.append(f"  Files changed: {', '.join(shown)}")
            if len(files) > 5:
                lines.append(f"    ...and {len(files) - 5} more")

        if decisions:
            for d in decisions[:2]:
                lines.append(f"  Decision: {d}")

        # Conflict warnings
        sib_conflicts = conflict_map.get(sib.get("session_id"), [])
        if sib_conflicts:
            for f in sib_conflicts:
                lines.append(f"  WARNING CONFLICT: You both modified {f}")

    lines.append("[/PARALLEL SESSIONS]")
    lines.append("")

    return "\n".join(lines)


def check_and_trigger_flush(session_id: str, project_path: str):
    """Check if flush is needed and trigger it."""
    # Check flush conditions
    flush_check = call_memory_agent("check_flush_needed", {
        "session_id": session_id
    })

    if flush_check and flush_check.get("flush_needed"):
        reasons = flush_check.get("reasons", [])
        logger.info(f"Flush needed: {', '.join(reasons)}")

        # Trigger flush
        flush_result = call_memory_agent("pre_compaction_flush", {
            "project_path": project_path,
            "session_id": session_id
        })

        if flush_result and flush_result.get("success"):
            logger.info(f"Flush completed: {flush_result.get('file_path')}")


def main():
    """Main entry point for the hook."""
    # Read hook input from stdin (Claude Code sends session_id, cwd, prompt)
    hook_input = {}
    try:
        hook_input = json.load(sys.stdin)
    except Exception:
        pass

    project_path = hook_input.get("cwd") or get_project_path()

    # Get session_id: stdin JSON > env var > .claude_session file
    session_id = hook_input.get("session_id") or get_session_id()

    # Ensure .claude_session file exists for sibling hooks
    if session_id:
        session_data = load_session_data() or {}
        if session_data.get("session_id") != session_id:
            session_data["session_id"] = session_id
            save_session_data(session_data)

    # If no session, try to init one
    if not session_id:
        init_result = call_memory_agent("state_init_session", {
            "project_path": project_path
        })
        if init_result and init_result.get("session_id"):
            session_id = init_result["session_id"]
            # Save session data as JSON
            save_session_data({"session_id": session_id})

    if not session_id:
        # No session, no grounding - exit silently
        sys.exit(0)

    # ============================================================
    # CROSS-SESSION AWARENESS: Heartbeat + parallel session context
    # ============================================================
    parallel_context = ""
    heartbeat_result = call_rest_api("POST", "/api/sessions/heartbeat", {
        "session_id": session_id,
        "project_path": project_path,
    })
    if heartbeat_result:
        parallel_context = format_parallel_sessions(heartbeat_result)

    # ============================================================
    # MOLTBOT-INSPIRED: Check flush conditions
    # ============================================================
    check_and_trigger_flush(session_id, project_path)

    # ============================================================
    # MOLTBOT-INSPIRED: Load MEMORY.md summary
    # ============================================================
    memory_md = call_memory_agent("get_memory_md_summary", {
        "project_path": project_path
    })

    memory_md_context = format_memory_md_context(memory_md) if memory_md else ""

    # ============================================================
    # MOLTBOT-INSPIRED: Load today's daily log highlights
    # ============================================================
    daily_highlights = call_memory_agent("daily_log_highlights", {
        "project_path": project_path
    })

    daily_context = format_daily_highlights(daily_highlights) if daily_highlights else ""

    # ============================================================
    # ORIGINAL: Get grounding context
    # ============================================================
    context = call_memory_agent("context_refresh", {
        "session_id": session_id,
        "include_recent_events": 5,
        "include_state": True,
        "include_checkpoint": True,
        "check_contradictions": True
    })

    grounding_context = format_grounding_context(context) if context else ""

    # ============================================================
    # CURATOR: Get curated context and status
    # ============================================================
    # Get curator summary for current context (lightweight)
    curator_summary = None
    curator_status = None

    # Only fetch curator context if there's user input to contextualize
    user_input = hook_input.get("prompt", "") or hook_input.get("user_prompt", "") or os.getenv("CLAUDE_USER_INPUT", "")
    if user_input and len(user_input) > 10:
        curator_summary = call_memory_agent("curator_get_summary", {
            "query": user_input[:500],  # Limit query length
            "project_path": project_path,
            "max_memories": 5,
            "include_graph": True
        })

    # Always get curator status for warnings
    curator_status = call_memory_agent("curator_get_status", {})

    curator_context = format_curator_context(curator_summary, curator_status)

    # Combine all context
    output_parts = []

    if parallel_context:
        output_parts.append(parallel_context)

    if memory_md_context:
        output_parts.append(memory_md_context)

    if daily_context:
        output_parts.append(daily_context)

    if grounding_context:
        output_parts.append(grounding_context)

    if curator_context:
        output_parts.append(curator_context)

    if output_parts:
        print("\n".join(output_parts))

    sys.exit(0)


if __name__ == "__main__":
    main()
