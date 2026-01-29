"""Session summarization skill with auto-summarization and session handoff.

Provides:
- Manual session summarization
- Automatic end-of-session summarization
- Session handoff for continuity
- Diary-style detailed session entries
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from services.database import DatabaseService
from services.embeddings import EmbeddingService


async def summarize_session(
    db: DatabaseService,
    embeddings: EmbeddingService,
    session_id: str,
    summary: str,
    key_decisions: Optional[List[str]] = None,
    code_patterns: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    project_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Store a session summary with optional key decisions and code patterns.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        session_id: The session identifier
        summary: Summary of the session
        key_decisions: List of key decisions made during session
        code_patterns: List of important code patterns discovered
        metadata: Additional metadata
        project_path: Project this session worked on

    Returns:
        Dict with stored summary information
    """
    stored_ids = []

    # Store the main session summary
    summary_embedding = await embeddings.generate_embedding(summary)
    summary_meta = {
        **(metadata or {}),
        "summarized_at": datetime.now().isoformat()
    }
    summary_id = await db.store_memory(
        memory_type="session",
        content=summary,
        embedding=summary_embedding,
        metadata=summary_meta,
        session_id=session_id,
        project_path=project_path,
        importance=8  # Session summaries are high importance
    )
    stored_ids.append({"type": "session", "id": summary_id})

    # Store key decisions
    if key_decisions:
        for decision in key_decisions:
            decision_embedding = await embeddings.generate_embedding(decision)
            decision_id = await db.store_memory(
                memory_type="decision",
                content=decision,
                embedding=decision_embedding,
                metadata={"session_summary_id": summary_id},
                session_id=session_id,
                project_path=project_path,
                importance=7  # Decisions are important
            )
            stored_ids.append({"type": "decision", "id": decision_id})

    # Store code patterns
    if code_patterns:
        for pattern in code_patterns:
            pattern_embedding = await embeddings.generate_embedding(pattern)
            pattern_id = await db.store_memory(
                memory_type="code",
                content=pattern,
                embedding=pattern_embedding,
                metadata={"session_summary_id": summary_id},
                session_id=session_id,
                project_path=project_path,
                importance=6  # Code patterns are useful
            )
            stored_ids.append({"type": "code", "id": pattern_id})

    return {
        "success": True,
        "session_id": session_id,
        "project_path": project_path,
        "stored_items": stored_ids,
        "total_items": len(stored_ids),
        "message": f"Session {session_id} summarized with {len(stored_ids)} items"
    }


async def auto_summarize_session(
    db: DatabaseService,
    embeddings: EmbeddingService,
    session_id: str,
    project_path: Optional[str] = None
) -> Dict[str, Any]:
    """Automatically summarize a session based on its timeline events.

    Analyzes the session's timeline to extract:
    - Goals and outcomes
    - Key decisions made
    - Patterns observed
    - Unresolved issues

    Args:
        db: Database service
        embeddings: Embeddings service
        session_id: Session to summarize
        project_path: Project context

    Returns:
        Generated summary with extracted components
    """
    # Get all timeline events for this session
    events = await db.execute_query(
        """
        SELECT event_type, summary, details, outcome, status, is_anchor, created_at
        FROM timeline_events
        WHERE session_id = ?
        ORDER BY sequence_num ASC
        """,
        (session_id,)
    )

    if not events:
        return {
            "success": False,
            "error": "No timeline events found for session",
            "session_id": session_id
        }

    # Get session state for context
    state = await db.execute_query(
        """
        SELECT current_goal, decisions_summary, pending_questions
        FROM session_state
        WHERE session_id = ?
        """,
        (session_id,)
    )
    session_state = state[0] if state else {}

    # Extract components from events
    goals = []
    decisions = []
    observations = []
    errors = []
    unresolved = []
    anchors = []

    for event in events:
        event_type = event.get("event_type", "")
        summary = event.get("summary", "")
        outcome = event.get("outcome", "")
        status = event.get("status", "")

        if event_type == "goal":
            goals.append(summary)
        elif event_type == "decision":
            decisions.append(summary)
        elif event_type == "observation":
            observations.append(summary)
        elif event_type == "error":
            errors.append(summary)
            if status != "resolved":
                unresolved.append(f"Error: {summary}")

        if event.get("is_anchor"):
            anchors.append(summary)

        # Check for pending/incomplete status
        if status in ("pending", "blocked", "failed"):
            unresolved.append(f"{event_type}: {summary}")

    # Check for pending questions in session state
    if session_state.get("pending_questions"):
        try:
            import json
            questions = json.loads(session_state["pending_questions"])
            for q in questions:
                unresolved.append(f"Unanswered: {q}")
        except:
            pass

    # Generate summary text
    summary_parts = []

    # Goals and outcomes
    if goals:
        summary_parts.append(f"Goals: {'; '.join(goals[:3])}")
    elif session_state.get("current_goal"):
        summary_parts.append(f"Goal: {session_state['current_goal']}")

    # Key accomplishments
    completed_count = sum(1 for e in events if e.get("status") == "completed")
    summary_parts.append(f"Completed {completed_count} actions across {len(events)} events.")

    # Decisions
    if decisions:
        summary_parts.append(f"Key decisions: {'; '.join(decisions[:3])}")

    # Issues
    if errors:
        summary_parts.append(f"Encountered {len(errors)} error(s).")

    # Anchors (verified facts)
    if anchors:
        summary_parts.append(f"Verified facts: {len(anchors)}")

    summary_text = " ".join(summary_parts)

    # Store the auto-generated summary
    result = await summarize_session(
        db=db,
        embeddings=embeddings,
        session_id=session_id,
        summary=summary_text,
        key_decisions=decisions[:5] if decisions else None,
        metadata={
            "auto_generated": True,
            "event_count": len(events),
            "unresolved_count": len(unresolved)
        },
        project_path=project_path
    )

    result["auto_summary"] = {
        "goals": goals[:5],
        "decisions": decisions[:5],
        "observations": observations[:5],
        "errors": errors[:5],
        "unresolved": unresolved[:5],
        "anchors": anchors[:5],
        "event_count": len(events)
    }

    return result


async def get_session_handoff(
    db: DatabaseService,
    embeddings: EmbeddingService,
    project_path: Optional[str] = None,
    include_last_n_sessions: int = 3
) -> Dict[str, Any]:
    """Get context handoff from previous sessions for continuity.

    Retrieves summaries and unresolved items from recent sessions
    to provide context for a new session.

    Args:
        db: Database service
        embeddings: Embeddings service
        project_path: Filter to specific project
        include_last_n_sessions: Number of recent sessions to include

    Returns:
        Handoff context with previous session summaries
    """
    # Get recent session summaries
    query = """
        SELECT id, content, session_id, project_path, metadata, importance, created_at
        FROM memories
        WHERE type = 'session'
    """
    params = []

    if project_path:
        query += " AND project_path = ?"
        params.append(project_path)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(include_last_n_sessions)

    summaries = await db.execute_query(query, tuple(params))

    if not summaries:
        return {
            "success": True,
            "has_previous_sessions": False,
            "message": "No previous session summaries found",
            "handoff": None
        }

    # Get unresolved items from recent sessions
    session_ids = [s["session_id"] for s in summaries if s.get("session_id")]

    unresolved_items = []
    if session_ids:
        placeholders = ",".join("?" * len(session_ids))
        unresolved = await db.execute_query(
            f"""
            SELECT summary, event_type, session_id
            FROM timeline_events
            WHERE session_id IN ({placeholders})
            AND status IN ('pending', 'blocked', 'failed')
            ORDER BY created_at DESC
            LIMIT 10
            """,
            tuple(session_ids)
        )
        unresolved_items = [
            {"type": u["event_type"], "summary": u["summary"]}
            for u in (unresolved or [])
        ]

    # Get recent decisions for context
    decisions = await db.execute_query(
        """
        SELECT content, project_path, created_at
        FROM memories
        WHERE type = 'decision'
        AND importance >= 7
        ORDER BY created_at DESC
        LIMIT 5
        """
    )

    # Build handoff
    handoff = {
        "previous_sessions": [
            {
                "session_id": s["session_id"],
                "summary": s["content"][:500],
                "project_path": s.get("project_path"),
                "created_at": s["created_at"]
            }
            for s in summaries
        ],
        "unresolved_items": unresolved_items,
        "recent_decisions": [
            {"content": d["content"][:200], "created_at": d["created_at"]}
            for d in (decisions or [])
        ],
        "context_message": _generate_handoff_message(summaries, unresolved_items)
    }

    return {
        "success": True,
        "has_previous_sessions": True,
        "session_count": len(summaries),
        "unresolved_count": len(unresolved_items),
        "handoff": handoff
    }


def _generate_handoff_message(summaries: List[Dict], unresolved: List[Dict]) -> str:
    """Generate a human-readable handoff message."""
    parts = []

    if summaries:
        last = summaries[0]
        parts.append(f"Last session: {last.get('content', '')[:200]}")

    if unresolved:
        items = [u["summary"][:50] for u in unresolved[:3]]
        parts.append(f"Pending items: {'; '.join(items)}")

    if not parts:
        return "No previous context available."

    return " | ".join(parts)


async def create_diary_entry(
    db: DatabaseService,
    embeddings: EmbeddingService,
    session_id: str,
    project_path: Optional[str] = None,
    user_notes: Optional[str] = None
) -> Dict[str, Any]:
    """Create a detailed diary-style entry for a session.

    Generates a structured narrative of the session including:
    - Timeline of events
    - Key milestones
    - Learnings and insights
    - Recommendations for future sessions

    Args:
        db: Database service
        embeddings: Embeddings service
        session_id: Session to create diary for
        project_path: Project context
        user_notes: Optional user-provided notes to include

    Returns:
        Formatted diary entry
    """
    # Get session timeline
    events = await db.execute_query(
        """
        SELECT event_type, summary, details, outcome, status, is_anchor,
               created_at, confidence
        FROM timeline_events
        WHERE session_id = ?
        ORDER BY sequence_num ASC
        """,
        (session_id,)
    )

    if not events:
        return {
            "success": False,
            "error": "No timeline events found for session"
        }

    # Get session state
    state = await db.execute_query(
        """
        SELECT current_goal, decisions_summary, entity_registry,
               created_at, updated_at
        FROM session_state
        WHERE session_id = ?
        """,
        (session_id,)
    )
    session_state = state[0] if state else {}

    # Build diary entry
    import json

    # Header
    start_time = events[0]["created_at"] if events else "Unknown"
    end_time = events[-1]["created_at"] if events else "Unknown"

    diary_parts = [
        f"# Session Diary: {session_id[:8]}...",
        f"**Date:** {start_time[:10] if start_time else 'Unknown'}",
        f"**Duration:** {start_time} to {end_time}",
        f"**Project:** {project_path or 'Not specified'}",
        "",
        "## Goals",
        session_state.get("current_goal", "No explicit goal recorded."),
        ""
    ]

    # Timeline section
    diary_parts.append("## Session Timeline")
    for i, event in enumerate(events[:20], 1):  # Limit to 20 events
        status_icon = {
            "completed": "[OK]",
            "failed": "[FAIL]",
            "pending": "[...]",
            "blocked": "[BLOCK]"
        }.get(event.get("status", ""), "[-]")

        anchor_mark = " (ANCHOR)" if event.get("is_anchor") else ""
        diary_parts.append(
            f"{i}. {status_icon} **{event['event_type']}**: {event['summary'][:100]}{anchor_mark}"
        )

    # Decisions section
    decisions = [e for e in events if e["event_type"] == "decision"]
    if decisions:
        diary_parts.extend(["", "## Key Decisions"])
        for d in decisions[:5]:
            diary_parts.append(f"- {d['summary']}")

    # Learnings section
    observations = [e for e in events if e["event_type"] == "observation"]
    if observations:
        diary_parts.extend(["", "## Observations & Learnings"])
        for o in observations[:5]:
            diary_parts.append(f"- {o['summary']}")

    # Errors and issues
    errors = [e for e in events if e["event_type"] == "error"]
    if errors:
        diary_parts.extend(["", "## Issues Encountered"])
        for e in errors[:5]:
            resolved = "Resolved" if e.get("status") == "completed" else "Unresolved"
            diary_parts.append(f"- [{resolved}] {e['summary']}")

    # Anchored facts
    anchors = [e for e in events if e.get("is_anchor")]
    if anchors:
        diary_parts.extend(["", "## Verified Facts (Anchors)"])
        for a in anchors[:5]:
            diary_parts.append(f"- {a['summary']}")

    # User notes
    if user_notes:
        diary_parts.extend(["", "## User Notes", user_notes])

    # Statistics
    diary_parts.extend([
        "",
        "## Statistics",
        f"- Total events: {len(events)}",
        f"- Decisions made: {len(decisions)}",
        f"- Errors encountered: {len(errors)}",
        f"- Anchored facts: {len(anchors)}"
    ])

    diary_content = "\n".join(diary_parts)

    # Store diary as a high-importance session memory
    diary_embedding = await embeddings.generate_embedding(diary_content[:2000])
    diary_id = await db.store_memory(
        memory_type="session",
        content=diary_content,
        embedding=diary_embedding,
        metadata={
            "diary_entry": True,
            "event_count": len(events),
            "decision_count": len(decisions),
            "has_user_notes": user_notes is not None
        },
        session_id=session_id,
        project_path=project_path,
        importance=9  # Diary entries are very important
    )

    return {
        "success": True,
        "diary_id": diary_id,
        "session_id": session_id,
        "content": diary_content,
        "stats": {
            "event_count": len(events),
            "decision_count": len(decisions),
            "error_count": len(errors),
            "anchor_count": len(anchors)
        }
    }


async def check_session_inactivity(
    db: DatabaseService,
    session_id: str,
    inactivity_threshold_hours: float = 4.0
) -> Dict[str, Any]:
    """Check if a session has been inactive and should be auto-summarized.

    Args:
        db: Database service
        session_id: Session to check
        inactivity_threshold_hours: Hours of inactivity before triggering

    Returns:
        Whether session should be summarized
    """
    # Get last activity
    result = await db.execute_query(
        """
        SELECT MAX(created_at) as last_event
        FROM timeline_events
        WHERE session_id = ?
        """,
        (session_id,)
    )

    if not result or not result[0].get("last_event"):
        return {"should_summarize": False, "reason": "No events found"}

    last_event = result[0]["last_event"]

    try:
        last_dt = datetime.fromisoformat(last_event.replace('Z', '+00:00'))
        now = datetime.now()
        hours_inactive = (now - last_dt.replace(tzinfo=None)).total_seconds() / 3600

        if hours_inactive >= inactivity_threshold_hours:
            return {
                "should_summarize": True,
                "reason": f"Inactive for {hours_inactive:.1f} hours",
                "last_activity": last_event,
                "hours_inactive": hours_inactive
            }

        return {
            "should_summarize": False,
            "reason": f"Active within threshold ({hours_inactive:.1f}h < {inactivity_threshold_hours}h)",
            "last_activity": last_event,
            "hours_inactive": hours_inactive
        }
    except Exception as e:
        return {"should_summarize": False, "reason": f"Error: {str(e)}"}
