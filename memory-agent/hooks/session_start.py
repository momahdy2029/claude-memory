#!/usr/bin/env python3
"""Session start hook - auto-loads relevant context.

This hook runs when a Claude Code session starts and:
- Loads project info and preferences
- Retrieves recent decisions and patterns
- Gets unresolved items from previous sessions
- Injects relevant context into the session
- Loads daily logs (Moltbot-inspired)
- Loads MEMORY.md core facts (Moltbot-inspired)

Configure in Claude Code settings:
{
  "hooks": {
    "SessionStart": ["python /path/to/session_start.py"]
  }
}

Output is printed to stdout and injected into Claude's context.
"""
import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

MEMORY_AGENT_URL = os.getenv("MEMORY_AGENT_URL", "http://localhost:8102")
API_KEY = os.getenv("MEMORY_API_KEY", "")
SESSION_ID = os.getenv("CLAUDE_SESSION_ID", "")


async def call_memory_skill(skill_id: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Call a memory agent skill."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-Memory-Key"] = API_KEY

    payload = {
        "jsonrpc": "2.0",
        "method": "skills/call",
        "params": {
            "skill_id": skill_id,
            "params": params
        },
        "id": f"session-start-{datetime.now().isoformat()}"
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{MEMORY_AGENT_URL}/a2a",
                json=payload,
                headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("result", {}).get("result", {})
    except Exception as e:
        pass
    return None


async def call_rest_api(method: str, path: str, json_body: dict = None, params: dict = None) -> Optional[Dict[str, Any]]:
    """Call the memory agent REST API."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            url = f"{MEMORY_AGENT_URL}{path}"
            if method == "POST":
                response = await client.post(url, json=json_body)
            else:
                response = await client.get(url, params=params)
            if response.status_code == 200:
                return response.json()
    except Exception:
        pass
    return None


async def load_session_context(project_path: str) -> str:
    """Load all relevant context for a session start."""
    context_parts = []

    # ============================================================
    # SOUL LAYER: Load soul brief (personality + learning context)
    # ============================================================
    soul_brief = await call_rest_api("GET", "/api/soul/brief", params={
        "project_path": project_path,
    })

    if soul_brief and soul_brief.get("success") and soul_brief.get("brief"):
        context_parts.append(soul_brief["brief"])

    # ============================================================
    # CROSS-SESSION AWARENESS: Register this session + catch-up
    # ============================================================
    session_id = SESSION_ID or os.getenv("CLAUDE_SESSION_ID", "")
    if session_id:
        register_result = await call_rest_api("POST", "/api/sessions/register", {
            "session_id": session_id,
            "project_path": project_path,
        })

        if register_result and register_result.get("active_siblings"):
            siblings = register_result["active_siblings"]
            context_parts.append("\n## Active Parallel Sessions")
            for sib in siblings:
                label = sib.get("session_label") or sib.get("session_id", "")[:12]
                goal = sib.get("current_goal", "unknown")
                files = sib.get("files_modified", [])
                context_parts.append(f"- **{label}**: {goal}")
                if files:
                    context_parts.append(f"  Files: {', '.join(files[:5])}")

        # Get catch-up: what happened while this session was away
        catchup = await call_rest_api("GET", "/api/sessions/catch-up", params={
            "session_id": session_id,
            "project_path": project_path,
        })

        if catchup and catchup.get("sessions"):
            context_parts.append("\n## What Happened While You Were Away")
            for sess in catchup["sessions"]:
                label = sess.get("session_label") or sess.get("session_id", "")[:12]
                events = sess.get("events", [])
                if events:
                    context_parts.append(f"### Session: {label}")
                    for ev in events[:5]:
                        etype = ev.get("event_type", "")
                        summary = ev.get("summary", "")
                        context_parts.append(f"- [{etype}] {summary}")

    # ============================================================
    # MOLTBOT-INSPIRED: Load MEMORY.md first (core facts)
    # ============================================================
    memory_md = await call_memory_skill("read_memory_md", {
        "project_path": project_path
    })

    if memory_md and memory_md.get("exists"):
        context_parts.append("## Core Facts (from MEMORY.md)")
        # Include the summary or first part of content
        content = memory_md.get("content", "")
        # Truncate if too long
        if len(content) > 2000:
            content = content[:2000] + "\n...(truncated)"
        context_parts.append(content)

    # ============================================================
    # MOLTBOT-INSPIRED: Load recent daily logs
    # ============================================================
    daily_logs = await call_memory_skill("daily_log_read", {
        "project_path": project_path,
        "days": 2,
        "max_chars": 3000
    })

    if daily_logs and daily_logs.get("logs"):
        context_parts.append("\n## Recent Activity (from Daily Logs)")
        for log in daily_logs["logs"]:
            log_date = log.get("date", "Unknown")
            log_content = log.get("content", "")
            # Show just the highlights, not full content
            if len(log_content) > 1500:
                log_content = log_content[:1500] + "\n...(truncated)"
            context_parts.append(f"\n### {log_date}")
            context_parts.append(log_content)

    # ============================================================
    # ORIGINAL MEMORY SYSTEM CONTEXT
    # ============================================================

    # 1. Get project info
    project_info = await call_memory_skill("get_project_context", {
        "project_path": project_path,
        "limit": 5
    })

    if project_info and project_info.get("project"):
        proj = project_info["project"]
        context_parts.append(f"\n## Project: {proj.get('name', project_path)}")
        if proj.get("tech_stack"):
            context_parts.append(f"Tech Stack: {', '.join(proj['tech_stack'])}")
        if proj.get("conventions"):
            context_parts.append(f"Conventions: {json.dumps(proj['conventions'], indent=2)}")

    # 2. Get recent decisions
    decisions = await call_memory_skill("semantic_search", {
        "query": "decision architecture approach",
        "project_path": project_path,
        "type": "decision",
        "limit": 5
    })

    if decisions and decisions.get("results"):
        context_parts.append("\n## Recent Decisions")
        for d in decisions["results"][:3]:
            context_parts.append(f"- {d['content'][:150]}")

    # 3. Get recent errors (to avoid repeating)
    errors = await call_memory_skill("semantic_search", {
        "query": "error bug fix problem",
        "project_path": project_path,
        "type": "error",
        "success_only": True,  # Only get solved errors
        "limit": 5
    })

    if errors and errors.get("results"):
        context_parts.append("\n## Past Errors & Solutions")
        for e in errors["results"][:3]:
            context_parts.append(f"- {e['content'][:150]}")

    # 4. Get session handoff (unresolved items)
    handoff = await call_memory_skill("get_session_handoff", {
        "project_path": project_path,
        "include_last_n_sessions": 2
    })

    if handoff:
        if handoff.get("unresolved_questions"):
            context_parts.append("\n## Unresolved from Previous Sessions")
            for q in handoff["unresolved_questions"][:3]:
                context_parts.append(f"- {q}")

        if handoff.get("recent_summaries"):
            context_parts.append("\n## Recent Session Summaries")
            for s in handoff["recent_summaries"][:2]:
                context_parts.append(f"- {s.get('summary', '')[:200]}")

    # 5. Get relevant patterns
    patterns = await call_memory_skill("search_patterns", {
        "query": "common patterns solutions",
        "limit": 3
    })

    if patterns and patterns.get("patterns"):
        context_parts.append("\n## Useful Patterns")
        for p in patterns["patterns"][:2]:
            context_parts.append(f"- **{p['name']}**: {p['solution'][:100]}")

    # 6. Check for anchor conflicts
    conflicts = await call_memory_skill("get_unresolved_conflicts", {
        "project_path": project_path,
        "limit": 3
    })

    if conflicts and conflicts.get("conflicts"):
        context_parts.append("\n## Unresolved Fact Conflicts")
        for c in conflicts["conflicts"]:
            context_parts.append(f"- {c.get('anchor1_summary', '')} vs {c.get('anchor2_summary', '')}")

    if context_parts:
        return "\n".join(context_parts)
    return ""


async def main():
    # Get project path from environment or current directory
    project_path = os.getenv("PROJECT_PATH") or os.getcwd()

    # Try to read from stdin if available
    try:
        if not sys.stdin.isatty():
            data = sys.stdin.read()
            if data:
                hook_data = json.loads(data)
                project_path = hook_data.get("project_path", project_path)
    except:
        pass

    context = await load_session_context(project_path)

    if context:
        # Output context for Claude to see
        print("\n<memory-context>")
        print("# Loaded from Memory System")
        print(context)
        print("</memory-context>\n")
    else:
        print("\n<memory-context>")
        print("# No prior context found for this project")
        print("Starting fresh session.")
        print("</memory-context>\n")


if __name__ == "__main__":
    asyncio.run(main())
