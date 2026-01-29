#!/usr/bin/env python3
"""Session end hook - auto-summarizes and stores session.

This hook runs when a Claude Code session ends and:
- Summarizes the session automatically
- Stores important decisions and learnings
- Updates project insights
- Syncs to CLAUDE.md if needed

Configure in Claude Code settings:
{
  "hooks": {
    "SessionEnd": ["python /path/to/session_end.py"]
  }
}
"""
import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

MEMORY_AGENT_URL = os.getenv("MEMORY_AGENT_URL", "http://localhost:8102")
API_KEY = os.getenv("MEMORY_API_KEY", "")


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
        "id": f"session-end-{datetime.now().isoformat()}"
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{MEMORY_AGENT_URL}/a2a",
                json=payload,
                headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("result", {}).get("result", {})
    except Exception:
        pass
    return None


async def end_session(session_id: str, project_path: str):
    """Handle session end - summarize and store."""
    results = []

    # 1. Auto-summarize the session
    summary = await call_memory_skill("auto_summarize_session", {
        "session_id": session_id,
        "project_path": project_path
    })

    if summary and summary.get("success"):
        results.append(f"Session summarized: {summary.get('summary', '')[:100]}...")

    # 2. Create diary entry
    diary = await call_memory_skill("create_diary_entry", {
        "session_id": session_id,
        "project_path": project_path
    })

    if diary and diary.get("success"):
        results.append(f"Diary entry created: ID {diary.get('memory_id')}")

    # 3. Run insight aggregation
    insights = await call_memory_skill("run_aggregation", {
        "project_path": project_path
    })

    if insights and insights.get("success"):
        new_insights = insights.get("new_insights", 0)
        if new_insights > 0:
            results.append(f"Generated {new_insights} new insights")

    # 4. Check for CLAUDE.md suggestions
    suggestions = await call_memory_skill("suggest_improvements", {
        "project_path": project_path
    })

    if suggestions and suggestions.get("suggestions"):
        results.append(f"CLAUDE.md suggestions: {len(suggestions['suggestions'])} available")

    # 5. Auto-resolve any obvious anchor conflicts
    resolved = await call_memory_skill("auto_resolve_conflicts", {
        "project_path": project_path
    })

    if resolved and resolved.get("resolved_count", 0) > 0:
        results.append(f"Auto-resolved {resolved['resolved_count']} conflicts")

    return results


async def main():
    session_id = os.getenv("SESSION_ID") or f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    project_path = os.getenv("PROJECT_PATH") or os.getcwd()

    # Try to read from stdin
    try:
        if not sys.stdin.isatty():
            data = sys.stdin.read()
            if data:
                hook_data = json.loads(data)
                session_id = hook_data.get("session_id", session_id)
                project_path = hook_data.get("project_path", project_path)
    except:
        pass

    results = await end_session(session_id, project_path)

    if results:
        print("\n[Memory System] Session ended:")
        for r in results:
            print(f"  - {r}")
    else:
        print("\n[Memory System] Session ended (no data captured)")


if __name__ == "__main__":
    asyncio.run(main())
