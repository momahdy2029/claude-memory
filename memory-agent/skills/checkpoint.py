"""Checkpoint skills for session snapshots and resumption."""
from typing import Dict, Any, Optional, List
from services.database import DatabaseService
from services.embeddings import EmbeddingService
from services.timeline import TimelineService


async def checkpoint_create(
    db: DatabaseService,
    embeddings: EmbeddingService,
    session_id: str,
    summary: Optional[str] = None,
    key_facts: Optional[List[str]] = None,
    include_state: bool = True
) -> Dict[str, Any]:
    """
    Create a checkpoint snapshot of the current session.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        session_id: The session ID
        summary: Optional custom summary (auto-generated if not provided)
        key_facts: Optional list of key facts to highlight
        include_state: Include current state in checkpoint

    Returns:
        Dict with checkpoint info
    """
    timeline = TimelineService(db, embeddings)

    # Get current state
    state = await db.get_or_create_session_state(session_id)

    # Get recent events
    events_since = state.get("events_since_checkpoint", 0)
    recent_events = await db.get_timeline_events(
        session_id=session_id,
        limit=max(events_since, 25)
    )

    # Build summary if not provided
    if not summary:
        parts = []
        if state.get("current_goal"):
            parts.append(f"Goal: {state['current_goal']}")
        parts.append(f"{len(recent_events)} events since last checkpoint")
        summary = ". ".join(parts) if parts else "Manual checkpoint"

    # Extract key facts if not provided
    if not key_facts:
        key_facts = [
            e["summary"] for e in recent_events
            if e.get("is_anchor") or (e.get("event_type") == "decision" and e.get("confidence", 0) >= 0.8)
        ][:10]

    # Extract decisions
    decisions = [
        e["summary"] for e in recent_events
        if e.get("event_type") == "decision"
    ][:10]

    # Get last event ID
    event_id = recent_events[0]["id"] if recent_events else None

    # Generate embedding for checkpoint summary
    embedding = None
    if embeddings:
        embed_text = summary
        if key_facts:
            embed_text += "\n" + "\n".join(key_facts[:5])
        embedding = await embeddings.generate_embedding(embed_text)

    # Store checkpoint
    checkpoint_id = await db.store_checkpoint(
        session_id=session_id,
        summary=summary,
        event_id=event_id,
        key_facts=key_facts,
        decisions=decisions,
        entities=state.get("entity_registry") if include_state else None,
        current_goal=state.get("current_goal") if include_state else None,
        pending_items=state.get("pending_questions") if include_state else None,
        embedding=embedding,
        event_count=len(recent_events)
    )

    return {
        "success": True,
        "checkpoint_id": checkpoint_id,
        "session_id": session_id,
        "summary": summary,
        "key_facts_count": len(key_facts),
        "decisions_count": len(decisions),
        "events_captured": len(recent_events),
        "message": f"Checkpoint created with ID {checkpoint_id}"
    }


async def checkpoint_load(
    db: DatabaseService,
    session_id: Optional[str] = None,
    checkpoint_id: Optional[int] = None,
    project_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load context from a checkpoint for session resumption.

    Args:
        db: Database service instance
        session_id: Load latest checkpoint for this session
        checkpoint_id: Load specific checkpoint by ID
        project_path: Load latest checkpoint for this project

    Returns:
        Dict with checkpoint context
    """
    checkpoint = None

    if checkpoint_id:
        # Load specific checkpoint
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,))
        row = cursor.fetchone()
        if row:
            import json
            checkpoint = {
                "id": row["id"],
                "session_id": row["session_id"],
                "summary": row["summary"],
                "key_facts": json.loads(row["key_facts"]) if row["key_facts"] else [],
                "decisions": json.loads(row["decisions"]) if row["decisions"] else [],
                "entities": json.loads(row["entities"]) if row["entities"] else {},
                "current_goal": row["current_goal"],
                "pending_items": json.loads(row["pending_items"]) if row["pending_items"] else [],
                "event_count": row["event_count"],
                "created_at": row["created_at"]
            }
    elif session_id:
        checkpoint = await db.get_latest_checkpoint(session_id)
    elif project_path:
        # Get latest session for project, then its checkpoint
        state = await db.get_latest_session_for_project(project_path)
        if state:
            checkpoint = await db.get_latest_checkpoint(state["session_id"])

    if not checkpoint:
        return {
            "success": True,
            "checkpoint": None,
            "message": "No checkpoint found"
        }

    # Build grounding summary
    grounding = []
    if checkpoint.get("current_goal"):
        grounding.append(f"Goal: {checkpoint['current_goal']}")
    if checkpoint.get("key_facts"):
        grounding.append(f"Key facts: {', '.join(checkpoint['key_facts'][:3])}")
    if checkpoint.get("decisions"):
        grounding.append(f"Decisions: {', '.join(checkpoint['decisions'][:3])}")
    if checkpoint.get("pending_items"):
        grounding.append(f"Pending: {', '.join(checkpoint['pending_items'][:3])}")

    return {
        "success": True,
        "checkpoint": checkpoint,
        "grounding_summary": " | ".join(grounding) if grounding else checkpoint.get("summary"),
        "session_id": checkpoint.get("session_id"),
        "message": f"Loaded checkpoint from {checkpoint.get('created_at')}"
    }


async def checkpoint_list(
    db: DatabaseService,
    session_id: str,
    limit: int = 10
) -> Dict[str, Any]:
    """
    List checkpoints for a session.

    Args:
        db: Database service instance
        session_id: The session ID
        limit: Max checkpoints to return

    Returns:
        Dict with checkpoint list
    """
    checkpoints = await db.get_checkpoints_for_session(session_id, limit)

    return {
        "success": True,
        "session_id": session_id,
        "checkpoints": checkpoints,
        "count": len(checkpoints),
        "message": f"Found {len(checkpoints)} checkpoints"
    }
