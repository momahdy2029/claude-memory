#!/usr/bin/env python3
"""Session end hook - auto-summarizes and stores session.

This hook runs when a Claude Code session ends and:
- Summarizes the session automatically
- Stores important decisions and learnings
- Updates project insights
- Syncs to CLAUDE.md if needed
- Appends session summary to daily log (Moltbot-inspired)
- Triggers MEMORY.md sync (Moltbot-inspired)
- Executes pre-compaction flush (Moltbot-inspired)
- Outputs session review summary for user verification

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
from typing import Dict, Any, Optional, List

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


async def call_api(endpoint: str, method: str = "GET", params: Dict = None) -> Optional[Dict[str, Any]]:
    """Call a memory agent REST API endpoint."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-Memory-Key"] = API_KEY

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            url = f"{MEMORY_AGENT_URL}{endpoint}"
            if method == "GET":
                response = await client.get(url, headers=headers, params=params)
            else:
                response = await client.post(url, headers=headers, json=params)

            if response.status_code == 200:
                return response.json()
    except Exception:
        pass
    return None


async def get_session_review_summary(session_id: str) -> Dict[str, Any]:
    """Get a summary of session memories for review."""
    # Get session memories
    memories = await call_api(f"/api/session/{session_id}/memories")
    if not memories or not memories.get("success"):
        return {"success": False, "memory_count": 0}

    # Get suggestions for review
    suggestions = await call_api(f"/api/session/{session_id}/suggestions")

    return {
        "success": True,
        "memory_count": memories.get("memory_count", 0),
        "memories": memories.get("memories", []),
        "summary": memories.get("summary", {}),
        "suggestions": suggestions.get("suggestions", []) if suggestions else [],
        "suggestion_summary": suggestions.get("summary", {}) if suggestions else {}
    }


def format_review_output(review_data: Dict[str, Any], session_id: str) -> List[str]:
    """Format the session review data for output."""
    lines = []

    memory_count = review_data.get("memory_count", 0)
    if memory_count == 0:
        return ["No memories created this session"]

    lines.append(f"Session created {memory_count} memories for review")

    # Type breakdown
    summary = review_data.get("summary", {})
    by_type = summary.get("by_type", {})
    if by_type:
        type_parts = [f"{count} {mtype}" for mtype, count in by_type.items()]
        lines.append(f"  Types: {', '.join(type_parts)}")

    # Suggestion summary
    sugg_summary = review_data.get("suggestion_summary", {})
    if sugg_summary:
        keep = sugg_summary.get("suggested_keep", 0)
        discard = sugg_summary.get("suggested_discard", 0)
        partial = sugg_summary.get("suggested_partial", 0)
        lines.append(f"  Suggestions: {keep} keep, {partial} review, {discard} discard")

    # List high-importance memories
    memories = review_data.get("memories", [])
    high_importance = [m for m in memories if m.get("importance", 5) >= 7]
    if high_importance:
        lines.append("")
        lines.append("  High-importance memories to verify:")
        for m in high_importance[:5]:
            mtype = m.get("type", "chunk")
            content = m.get("content", "")[:60]
            importance = m.get("importance", 5)
            lines.append(f"    [{mtype}] (imp:{importance}) {content}...")

    # Review URL
    lines.append("")
    lines.append(f"  Review at: {MEMORY_AGENT_URL}/dashboard#review/{session_id}")

    return lines


async def end_session(session_id: str, project_path: str):
    """Handle session end - summarize and store."""
    results = []
    session_data = {
        "decisions": [],
        "accomplishments": [],
        "errors_solved": [],
        "notes": []
    }

    # 1. Auto-summarize the session
    summary = await call_memory_skill("auto_summarize_session", {
        "session_id": session_id,
        "project_path": project_path
    })

    if summary and summary.get("success"):
        results.append(f"Session summarized: {summary.get('summary', '')[:100]}...")
        # Extract decisions and accomplishments from summary if available
        if summary.get("key_decisions"):
            session_data["decisions"] = summary["key_decisions"][:5]

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

    # ============================================================
    # MOLTBOT-INSPIRED FEATURES
    # ============================================================

    # 6. Append session summary to daily log
    daily_log = await call_memory_skill("daily_log_append_session", {
        "project_path": project_path,
        "session_id": session_id,
        "decisions": session_data["decisions"],
        "accomplishments": session_data["accomplishments"],
        "errors_solved": session_data["errors_solved"],
        "notes": session_data["notes"]
    })

    if daily_log and daily_log.get("success"):
        results.append(f"Daily log updated: {daily_log.get('file_path', 'unknown')}")

    # 7. Sync MEMORY.md with high-importance items
    memory_md = await call_memory_skill("sync_memory_md", {
        "project_path": project_path,
        "min_importance": 7,
        "min_pattern_success": 3
    })

    if memory_md and memory_md.get("success"):
        counts = memory_md.get("counts", {})
        total_synced = sum(counts.values())
        if total_synced > 0:
            results.append(f"MEMORY.md synced: {total_synced} items")

    # 8. Execute pre-compaction flush
    flush = await call_memory_skill("pre_compaction_flush", {
        "project_path": project_path,
        "session_id": session_id
    })

    if flush and flush.get("success"):
        results.append(f"Memory flush created: {flush.get('file_path', 'unknown')}")

    # ============================================================
    # SESSION REVIEW SUMMARY
    # ============================================================

    # 9. Get session review summary for user verification
    review_data = await get_session_review_summary(session_id)
    if review_data.get("success") and review_data.get("memory_count", 0) > 0:
        review_lines = format_review_output(review_data, session_id)
        results.append("")
        results.append("--- Session Memory Review ---")
        results.extend(review_lines)

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
            if r.startswith("---") or r.startswith("  "):
                print(r)
            else:
                print(f"  - {r}")
    else:
        print("\n[Memory System] Session ended (no data captured)")


if __name__ == "__main__":
    asyncio.run(main())
