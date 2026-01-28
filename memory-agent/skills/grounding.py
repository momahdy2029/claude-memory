"""Grounding skills for anti-hallucination checks."""
import os
from typing import Dict, Any, Optional, List
from services.database import DatabaseService
from services.embeddings import EmbeddingService
from services.timeline import TimelineService

USE_LLM_ANALYSIS = os.getenv("USE_LLM_ANALYSIS", "true").lower() == "true"


async def context_refresh(
    db: DatabaseService,
    embeddings: EmbeddingService,
    session_id: str,
    query: Optional[str] = None,
    include_recent_events: int = 10,
    include_state: bool = True,
    include_checkpoint: bool = True,
    include_relevant_memories: bool = True,
    check_contradictions: bool = True
) -> Dict[str, Any]:
    """
    Pre-response grounding check. Call this before complex responses.

    Provides current context to prevent hallucinations by grounding
    Claude in what has actually happened and been decided.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        session_id: The session ID
        query: What Claude is about to respond about (for relevance)
        include_recent_events: Number of recent events to include
        include_state: Include current session state
        include_checkpoint: Include latest checkpoint
        include_relevant_memories: Search for relevant memories
        check_contradictions: Check for potential contradictions

    Returns:
        Dict with grounding context
    """
    timeline = TimelineService(db, embeddings)

    result = {
        "success": True,
        "session_id": session_id,
        "state": None,  # Full state for staleness checks
        "grounding": {
            "current_goal": None,
            "entity_registry": {},
            "recent_events": [],
            "anchors": [],
            "decisions": [],
            "checkpoint_summary": None,
            "relevant_memories": [],
            "contradictions": []
        }
    }

    # Get current state
    if include_state:
        state = await db.get_or_create_session_state(session_id)
        result["state"] = state  # Include full state for staleness checks
        result["grounding"]["current_goal"] = state.get("current_goal")
        result["grounding"]["entity_registry"] = state.get("entity_registry", {})
        result["grounding"]["pending_questions"] = state.get("pending_questions", [])

    # Get recent events
    if include_recent_events > 0:
        events = await db.get_timeline_events(
            session_id=session_id,
            limit=include_recent_events
        )
        result["grounding"]["recent_events"] = [
            {
                "type": e["event_type"],
                "summary": e["summary"],
                "is_anchor": e.get("is_anchor", False)
            }
            for e in events
        ]

        # Extract anchors (verified facts)
        result["grounding"]["anchors"] = [
            e["summary"] for e in events if e.get("is_anchor")
        ]

        # Extract decisions
        result["grounding"]["decisions"] = [
            e["summary"] for e in events if e.get("event_type") == "decision"
        ]

    # Get latest checkpoint
    if include_checkpoint:
        checkpoint = await db.get_latest_checkpoint(session_id)
        if checkpoint:
            result["grounding"]["checkpoint_summary"] = checkpoint.get("summary")
            # Add checkpoint's key facts to anchors
            if checkpoint.get("key_facts"):
                result["grounding"]["anchors"].extend(checkpoint["key_facts"])

    # Search relevant memories
    if include_relevant_memories and query and embeddings:
        embedding = await embeddings.generate_embedding(query)
        memories = await db.search_similar(
            embedding=embedding,
            limit=5,
            threshold=0.6
        )
        result["grounding"]["relevant_memories"] = [
            {
                "type": m["type"],
                "content": m["content"][:200],
                "similarity": m["similarity"]
            }
            for m in memories
        ]

    # Check for contradictions
    if check_contradictions and query:
        contradictions = await _find_contradictions(
            db, embeddings, query, session_id
        )
        result["grounding"]["contradictions"] = contradictions

    # Generate grounding summary
    result["grounding_summary"] = _generate_grounding_summary(result["grounding"])

    return result


async def check_contradictions(
    db: DatabaseService,
    embeddings: EmbeddingService,
    statement: str,
    session_id: Optional[str] = None,
    scope: str = "session"
) -> Dict[str, Any]:
    """
    Check if a statement contradicts known facts or decisions.

    Uses LLM-based analysis when available for more accurate detection.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        statement: The statement to check
        session_id: Session to check against
        scope: "session" (current only), "project", or "all"

    Returns:
        Dict with contradiction analysis
    """
    # Get anchors for LLM-based checking
    anchors = []
    if session_id:
        events = await db.get_timeline_events(
            session_id=session_id,
            limit=50,
            anchors_only=True
        )
        anchors = [e["summary"] for e in events if e.get("is_anchor")]

    # Try LLM-based analysis first
    llm_result = None
    if USE_LLM_ANALYSIS and anchors:
        try:
            from services.llm_analyzer import LLMAnalyzer
            analyzer = LLMAnalyzer()
            llm_result = await analyzer.check_statement_against_facts(
                statement, anchors
            )
        except:
            pass

    # Fall back to embedding-based search
    contradictions = await _find_contradictions(
        db, embeddings, statement, session_id, scope
    )

    # Merge results
    if llm_result and llm_result.get("has_contradiction"):
        contradictions.insert(0, {
            "type": "llm_analysis",
            "content": llm_result.get("conflicting_fact", "Unknown fact"),
            "reason": llm_result.get("reason", "LLM detected contradiction"),
            "confidence": 0.9  # High confidence for LLM detection
        })

    return {
        "success": True,
        "statement": statement,
        "has_contradictions": len(contradictions) > 0,
        "contradictions": contradictions,
        "confidence": 1.0 - (len(contradictions) * 0.2) if contradictions else 1.0,
        "message": f"Found {len(contradictions)} potential contradictions" if contradictions else "No contradictions found",
        "analysis_method": "llm" if llm_result else "embedding"
    }


async def _find_contradictions(
    db: DatabaseService,
    embeddings: EmbeddingService,
    statement: str,
    session_id: Optional[str] = None,
    scope: str = "session"
) -> List[Dict[str, Any]]:
    """Find potential contradictions to a statement."""
    contradictions = []

    if not embeddings:
        return contradictions

    # Generate embedding for statement
    embedding = await embeddings.generate_embedding(statement)

    # Search timeline events (anchors and decisions)
    if session_id:
        events = await db.search_timeline_events(
            embedding=embedding,
            session_id=session_id if scope == "session" else None,
            limit=10,
            threshold=0.7  # High similarity = potentially contradictory
        )

        for event in events:
            # Check if this might contradict
            if event.get("is_anchor") or event.get("event_type") == "decision":
                # Simple heuristic: high similarity to an anchor/decision
                # might indicate contradiction OR confirmation
                # Flag for human review
                contradictions.append({
                    "type": "timeline_event",
                    "event_type": event.get("event_type"),
                    "content": event.get("summary"),
                    "similarity": event.get("similarity"),
                    "reason": "High similarity to established fact/decision - verify alignment"
                })

    # Search memories for contradictions
    memories = await db.search_similar(
        embedding=embedding,
        limit=5,
        memory_type="decision",
        session_id=session_id if scope == "session" else None,
        threshold=0.7
    )

    for memory in memories:
        contradictions.append({
            "type": "memory",
            "memory_type": memory.get("type"),
            "content": memory.get("content")[:200],
            "similarity": memory.get("similarity"),
            "reason": "Similar decision/fact found in memory - verify consistency"
        })

    # Deduplicate and limit
    seen = set()
    unique_contradictions = []
    for c in contradictions:
        key = c.get("content", "")[:50]
        if key not in seen:
            seen.add(key)
            unique_contradictions.append(c)

    return unique_contradictions[:5]  # Limit to top 5


def _generate_grounding_summary(grounding: Dict[str, Any]) -> str:
    """Generate a concise grounding summary."""
    parts = []

    if grounding.get("current_goal"):
        parts.append(f"Goal: {grounding['current_goal']}")

    if grounding.get("anchors"):
        parts.append(f"Facts: {len(grounding['anchors'])} verified")

    if grounding.get("decisions"):
        parts.append(f"Decisions: {len(grounding['decisions'])} made")

    if grounding.get("entity_registry"):
        entities = list(grounding["entity_registry"].keys())[:3]
        if entities:
            parts.append(f"Entities: {', '.join(entities)}")

    if grounding.get("contradictions"):
        parts.append(f"WARNINGS: {len(grounding['contradictions'])} potential contradictions")

    return " | ".join(parts) if parts else "No context loaded"


async def verify_entity(
    db: DatabaseService,
    session_id: str,
    entity_key: str,
    entity_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Verify an entity reference against the registry.

    Use this when you're about to reference a file, variable, or other entity
    to ensure you have the correct one.

    Args:
        db: Database service instance
        session_id: The session ID
        entity_key: The entity key to verify (e.g., "auth_file")
        entity_type: Optional type filter ("file", "function", etc.)

    Returns:
        Dict with verification result
    """
    state = await db.get_or_create_session_state(session_id)
    registry = state.get("entity_registry", {})

    if entity_key in registry:
        return {
            "success": True,
            "verified": True,
            "entity_key": entity_key,
            "entity_value": registry[entity_key],
            "message": f"Entity '{entity_key}' verified: {registry[entity_key]}"
        }

    # Try to find similar keys
    similar = [k for k in registry.keys() if entity_key.lower() in k.lower() or k.lower() in entity_key.lower()]

    return {
        "success": True,
        "verified": False,
        "entity_key": entity_key,
        "entity_value": None,
        "similar_entities": {k: registry[k] for k in similar[:3]},
        "message": f"Entity '{entity_key}' not found in registry. Similar: {similar[:3]}"
    }


async def mark_anchor(
    db: DatabaseService,
    embeddings: EmbeddingService,
    session_id: str,
    fact: str,
    details: Optional[str] = None,
    project_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Mark a statement as a verified anchor fact.

    Use this to establish facts that should not be contradicted.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        session_id: The session ID
        fact: The verified fact
        details: Additional context
        project_path: Project path

    Returns:
        Dict with anchor info
    """
    timeline = TimelineService(db, embeddings)

    event_id = await timeline.log_event(
        session_id=session_id,
        event_type="anchor",
        summary=fact,
        details=details,
        project_path=project_path,
        confidence=1.0,
        is_anchor=True
    )

    return {
        "success": True,
        "event_id": event_id,
        "fact": fact,
        "message": f"Anchor fact established: {fact[:50]}..."
    }
