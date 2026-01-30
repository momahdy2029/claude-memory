"""Timeline skills for session event tracking."""
from typing import Dict, Any, Optional, List
from services.database import DatabaseService
from services.embeddings import EmbeddingService
from services.timeline import TimelineService


async def timeline_log(
    db: DatabaseService,
    embeddings: EmbeddingService,
    session_id: str,
    event_type: str,
    summary: str,
    details: Optional[str] = None,
    project_path: Optional[str] = None,
    parent_event_id: Optional[int] = None,
    root_event_id: Optional[int] = None,
    entities: Optional[Dict[str, List[str]]] = None,
    status: str = "completed",
    outcome: Optional[str] = None,
    confidence: Optional[float] = None,
    is_anchor: bool = False
) -> Dict[str, Any]:
    """
    Log an event to the session timeline.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        session_id: The session ID
        event_type: Type of event:
            - 'user_request': User asks for something
            - 'clarification': User clarifies or corrects
            - 'action': Claude takes an action (file edit, command)
            - 'decision': Explicit choice made
            - 'observation': Something Claude noticed
            - 'error': Error encountered
            - 'checkpoint': Session milestone
        summary: Brief description (<200 chars)
        details: Full context (optional)
        project_path: Project path (optional)
        parent_event_id: ID of parent event (causal chain)
        root_event_id: ID of root user request
        entities: Dict of entity references {"files": [], "functions": [], etc.}
        status: Event status (pending, in_progress, completed, failed)
        outcome: Result or error message
        confidence: Confidence level 0-1
        is_anchor: Whether this is a verified fact

    Returns:
        Dict with event info
    """
    timeline = TimelineService(db, embeddings)

    event_id = await timeline.log_event(
        session_id=session_id,
        event_type=event_type,
        summary=summary,
        details=details,
        project_path=project_path,
        parent_event_id=parent_event_id,
        root_event_id=root_event_id,
        entities=entities,
        status=status,
        outcome=outcome,
        confidence=confidence,
        is_anchor=is_anchor
    )

    return {
        "success": True,
        "event_id": event_id,
        "session_id": session_id,
        "event_type": event_type,
        "message": f"Event logged: {summary[:50]}..."
    }


async def timeline_get(
    db: DatabaseService,
    session_id: str,
    limit: int = 20,
    event_type: Optional[str] = None,
    since_event_id: Optional[int] = None,
    anchors_only: bool = False,
    include_state: bool = True,
    include_checkpoint: bool = True
) -> Dict[str, Any]:
    """
    Retrieve timeline events for a session.

    Args:
        db: Database service instance
        session_id: The session ID
        limit: Max events to return (default 20)
        event_type: Filter by event type
        since_event_id: Only events after this ID
        anchors_only: Only return verified facts
        include_state: Include current session state
        include_checkpoint: Include latest checkpoint

    Returns:
        Dict with events, state, and checkpoint
    """
    result = {
        "success": True,
        "session_id": session_id,
        "events": [],
        "state": None,
        "checkpoint": None
    }

    # Get events
    result["events"] = await db.get_timeline_events(
        session_id=session_id,
        limit=limit,
        event_type=event_type,
        since_event_id=since_event_id,
        anchors_only=anchors_only
    )

    # Get state
    if include_state:
        result["state"] = await db.get_or_create_session_state(session_id)

    # Get checkpoint
    if include_checkpoint:
        result["checkpoint"] = await db.get_latest_checkpoint(session_id)

    result["event_count"] = len(result["events"])

    return result


async def timeline_search(
    db: DatabaseService,
    embeddings: EmbeddingService,
    query: str,
    session_id: Optional[str] = None,
    limit: int = 10,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Semantic search across timeline events.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        query: Search query
        session_id: Limit to specific session (optional)
        limit: Max results (default 10)
        threshold: Minimum similarity (default 0.5)

    Returns:
        Dict with matching events
    """
    timeline = TimelineService(db, embeddings)

    events = await timeline.search_events(
        query=query,
        session_id=session_id,
        limit=limit,
        threshold=threshold
    )

    return {
        "success": True,
        "query": query,
        "session_id": session_id,
        "events": events,
        "count": len(events),
        "message": f"Found {len(events)} matching events"
    }


async def timeline_chain(
    db: DatabaseService,
    session_id: str,
    root_event_id: int,
    include_details: bool = False
) -> Dict[str, Any]:
    """
    Get the full causal chain for a user request.

    Shows the complete timeline:
    user_request → thinking → decision → action → outcome

    Args:
        db: Database service instance
        session_id: The session ID
        root_event_id: The root user_request event ID
        include_details: Whether to include full event details

    Returns:
        Dict with the causal chain as a tree structure
    """
    # Get all events linked to this root
    events = await db.get_timeline_events(
        session_id=session_id,
        limit=100  # Get all related events
    )

    # Filter events linked to this root
    chain_events = [
        e for e in events
        if e.get("root_event_id") == root_event_id or e.get("id") == root_event_id
    ]

    # Sort by sequence number
    chain_events.sort(key=lambda x: x.get("sequence_num", 0))

    # Build tree structure
    def build_tree(parent_id: Optional[int]) -> List[Dict]:
        children = []
        for event in chain_events:
            if event.get("parent_event_id") == parent_id or (parent_id is None and event.get("id") == root_event_id):
                node = {
                    "id": event.get("id"),
                    "type": event.get("event_type"),
                    "summary": event.get("summary", "")[:100],
                    "sequence": event.get("sequence_num"),
                }
                if include_details:
                    node["details"] = event.get("details")
                    node["created_at"] = event.get("created_at")

                # Recursively add children
                node["children"] = build_tree(event.get("id"))
                children.append(node)
        return children

    tree = build_tree(None)

    # Also create a flat timeline view
    flat_timeline = []
    for event in chain_events:
        flat_timeline.append({
            "seq": event.get("sequence_num"),
            "type": event.get("event_type"),
            "summary": event.get("summary", "")[:80]
        })

    return {
        "success": True,
        "session_id": session_id,
        "root_event_id": root_event_id,
        "tree": tree,
        "flat_timeline": flat_timeline,
        "event_count": len(chain_events),
        "message": f"Chain has {len(chain_events)} events"
    }


async def timeline_auto_detect(
    db: DatabaseService,
    embeddings: EmbeddingService,
    session_id: str,
    response_text: str,
    project_path: Optional[str] = None,
    parent_event_id: Optional[int] = None,
    root_event_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Auto-detect and log decisions/observations from a response.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        session_id: The session ID
        response_text: Claude's response text to analyze
        project_path: Project path (optional)
        parent_event_id: Parent event ID for causal chain (optional)
        root_event_id: Root user request event ID (optional)

    Returns:
        Dict with detected and logged events
    """
    timeline = TimelineService(db, embeddings)

    # Detect patterns
    decisions = timeline.detect_decisions(response_text)
    observations = timeline.detect_observations(response_text)
    entities = timeline.extract_entities(response_text)

    # Auto-log events with causal chain linking
    event_ids = await timeline.auto_log_from_response(
        session_id=session_id,
        response_text=response_text,
        project_path=project_path,
        parent_event_id=parent_event_id,
        root_event_id=root_event_id
    )

    return {
        "success": True,
        "session_id": session_id,
        "detected": {
            "decisions": decisions[:5],
            "observations": observations[:5],
            "entities": entities
        },
        "logged_event_ids": event_ids,
        "message": f"Auto-detected {len(decisions)} decisions, {len(observations)} observations"
    }
