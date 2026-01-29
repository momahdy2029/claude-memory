"""Grounding skills for anti-hallucination checks with anchor conflict resolution."""
import os
import json
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
            "contradictions": [],
            "unresolved_conflicts": []
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
        if embedding:
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

    # Check for unresolved anchor conflicts
    conflicts = await get_unresolved_conflicts(db, session_id)
    if conflicts.get("conflicts"):
        result["grounding"]["unresolved_conflicts"] = conflicts["conflicts"]

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
    if not embedding:
        return contradictions

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

    if grounding.get("unresolved_conflicts"):
        parts.append(f"CONFLICTS: {len(grounding['unresolved_conflicts'])} unresolved anchor conflicts")

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
    project_path: Optional[str] = None,
    force: bool = False
) -> Dict[str, Any]:
    """
    Mark a statement as a verified anchor fact with conflict detection.

    Before creating the anchor, checks for potential conflicts with existing
    anchors. If conflicts are found, can optionally proceed with force=True.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        session_id: The session ID
        fact: The verified fact
        details: Additional context
        project_path: Project path
        force: If True, create anchor even if conflicts exist

    Returns:
        Dict with anchor info and any detected conflicts
    """
    timeline = TimelineService(db, embeddings)

    # Check for conflicts with existing anchors
    conflicts = []
    if embeddings:
        fact_embedding = await embeddings.generate_embedding(fact)
        if fact_embedding:
            # Search existing anchors for high similarity
            existing = await db.search_timeline_events(
                embedding=fact_embedding,
                session_id=session_id,
                limit=10,
                threshold=0.75  # High similarity threshold
            )

            for event in existing:
                if event.get("is_anchor") and event.get("similarity", 0) > 0.75:
                    # Check if it's a potential contradiction or update
                    conflict_type = _classify_conflict(fact, event.get("summary", ""))
                    if conflict_type != "identical":
                        conflicts.append({
                            "anchor_id": event.get("id"),
                            "summary": event.get("summary"),
                            "similarity": event.get("similarity"),
                            "conflict_type": conflict_type
                        })

    # If conflicts exist and not forcing, return without creating
    if conflicts and not force:
        return {
            "success": False,
            "has_conflicts": True,
            "conflicts": conflicts,
            "fact": fact,
            "message": f"Found {len(conflicts)} potential conflicts with existing anchors. "
                      f"Use force=True to create anyway, or resolve conflicts first."
        }

    # Create the anchor event
    event_id = await timeline.log_event(
        session_id=session_id,
        event_type="anchor",
        summary=fact,
        details=details,
        project_path=project_path,
        confidence=1.0,
        is_anchor=True
    )

    # If there were conflicts and we're forcing, record them
    if conflicts:
        for conflict in conflicts:
            await _record_conflict(
                db=db,
                session_id=session_id,
                project_path=project_path,
                anchor1_id=conflict["anchor_id"],
                anchor2_id=event_id,
                anchor1_summary=conflict["summary"],
                anchor2_summary=fact,
                conflict_type=conflict["conflict_type"],
                similarity=conflict.get("similarity", 0)
            )

    # Log anchor history
    await _log_anchor_history(
        db=db,
        anchor_id=event_id,
        session_id=session_id,
        project_path=project_path,
        action="created",
        new_summary=fact,
        reason="Manual anchor creation"
    )

    return {
        "success": True,
        "event_id": event_id,
        "fact": fact,
        "conflicts_recorded": len(conflicts) if conflicts else 0,
        "message": f"Anchor fact established: {fact[:50]}..." +
                  (f" (with {len(conflicts)} conflicts recorded)" if conflicts else "")
    }


def _classify_conflict(new_fact: str, existing_fact: str) -> str:
    """Classify the type of conflict between two facts."""
    new_lower = new_fact.lower()
    existing_lower = existing_fact.lower()

    # Check for negation patterns
    negation_words = ["not", "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't", "no longer", "never"]
    new_has_negation = any(word in new_lower for word in negation_words)
    existing_has_negation = any(word in existing_lower for word in negation_words)

    if new_has_negation != existing_has_negation:
        return "contradiction"

    # Check if it's likely an update (same subject, different details)
    # Simple heuristic: first 5 words similar
    new_words = new_lower.split()[:5]
    existing_words = existing_lower.split()[:5]
    common_words = len(set(new_words) & set(existing_words))

    if common_words >= 3:
        return "update"

    if new_lower == existing_lower:
        return "identical"

    return "potential_conflict"


async def _record_conflict(
    db: DatabaseService,
    session_id: str,
    project_path: Optional[str],
    anchor1_id: int,
    anchor2_id: int,
    anchor1_summary: str,
    anchor2_summary: str,
    conflict_type: str,
    similarity: float
):
    """Record an anchor conflict for later resolution."""
    cursor = db.conn.cursor()
    cursor.execute(
        """
        INSERT INTO anchor_conflicts (
            session_id, project_path, anchor1_id, anchor2_id,
            anchor1_summary, anchor2_summary, conflict_type, similarity_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, project_path, anchor1_id, anchor2_id,
         anchor1_summary, anchor2_summary, conflict_type, similarity)
    )
    db.conn.commit()


async def _log_anchor_history(
    db: DatabaseService,
    anchor_id: int,
    session_id: str,
    project_path: Optional[str],
    action: str,
    previous_summary: Optional[str] = None,
    new_summary: Optional[str] = None,
    superseded_by: Optional[int] = None,
    reason: Optional[str] = None,
    confidence: float = 1.0
):
    """Log anchor history for tracking fact evolution."""
    cursor = db.conn.cursor()
    cursor.execute(
        """
        INSERT INTO anchor_history (
            anchor_id, session_id, project_path, action,
            previous_summary, new_summary, superseded_by,
            reason, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (anchor_id, session_id, project_path, action,
         previous_summary, new_summary, superseded_by, reason, confidence)
    )
    db.conn.commit()


async def get_unresolved_conflicts(
    db: DatabaseService,
    session_id: Optional[str] = None,
    project_path: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """Get unresolved anchor conflicts."""
    cursor = db.conn.cursor()

    query = "SELECT * FROM anchor_conflicts WHERE status = 'unresolved'"
    params = []

    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)

    if project_path:
        query += " AND project_path = ?"
        params.append(project_path)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()

    conflicts = [dict(row) for row in rows] if rows else []

    return {
        "success": True,
        "conflicts": conflicts,
        "count": len(conflicts)
    }


async def resolve_conflict(
    db: DatabaseService,
    embeddings: EmbeddingService,
    conflict_id: int,
    resolution: str,
    keep_anchor_id: Optional[int] = None,
    resolved_by: str = "user"
) -> Dict[str, Any]:
    """
    Resolve an anchor conflict.

    Args:
        db: Database service
        embeddings: Embeddings service
        conflict_id: ID of the conflict to resolve
        resolution: One of "keep_first", "keep_second", "keep_both", "supersede"
        keep_anchor_id: For "supersede", which anchor supersedes the other
        resolved_by: Who resolved it (user, auto, etc.)

    Returns:
        Resolution result
    """
    cursor = db.conn.cursor()

    # Get the conflict
    cursor.execute("SELECT * FROM anchor_conflicts WHERE id = ?", (conflict_id,))
    conflict = cursor.fetchone()

    if not conflict:
        return {"success": False, "error": "Conflict not found"}

    conflict = dict(conflict)
    anchor1_id = conflict["anchor1_id"]
    anchor2_id = conflict["anchor2_id"]

    # Handle resolution
    resolved_anchor_id = None

    if resolution == "keep_first":
        # Mark second anchor as superseded
        resolved_anchor_id = anchor1_id
        await _log_anchor_history(
            db=db,
            anchor_id=anchor2_id,
            session_id=conflict.get("session_id"),
            project_path=conflict.get("project_path"),
            action="superseded",
            superseded_by=anchor1_id,
            reason=f"Conflict resolution: kept anchor {anchor1_id}"
        )
        # Optionally mark the timeline event as non-anchor
        cursor.execute(
            "UPDATE timeline_events SET is_anchor = 0 WHERE id = ?",
            (anchor2_id,)
        )

    elif resolution == "keep_second":
        resolved_anchor_id = anchor2_id
        await _log_anchor_history(
            db=db,
            anchor_id=anchor1_id,
            session_id=conflict.get("session_id"),
            project_path=conflict.get("project_path"),
            action="superseded",
            superseded_by=anchor2_id,
            reason=f"Conflict resolution: kept anchor {anchor2_id}"
        )
        cursor.execute(
            "UPDATE timeline_events SET is_anchor = 0 WHERE id = ?",
            (anchor1_id,)
        )

    elif resolution == "keep_both":
        # Both remain as anchors, just mark conflict as acknowledged
        resolved_anchor_id = None
        await _log_anchor_history(
            db=db,
            anchor_id=anchor1_id,
            session_id=conflict.get("session_id"),
            project_path=conflict.get("project_path"),
            action="conflict_acknowledged",
            reason="Both anchors kept despite potential conflict"
        )

    elif resolution == "supersede" and keep_anchor_id:
        resolved_anchor_id = keep_anchor_id
        superseded_id = anchor2_id if keep_anchor_id == anchor1_id else anchor1_id
        await _log_anchor_history(
            db=db,
            anchor_id=superseded_id,
            session_id=conflict.get("session_id"),
            project_path=conflict.get("project_path"),
            action="superseded",
            superseded_by=keep_anchor_id,
            reason=f"Manual supersession by anchor {keep_anchor_id}"
        )
        cursor.execute(
            "UPDATE timeline_events SET is_anchor = 0 WHERE id = ?",
            (superseded_id,)
        )

    # Update conflict status
    cursor.execute(
        """
        UPDATE anchor_conflicts
        SET status = 'resolved',
            resolution = ?,
            resolved_anchor_id = ?,
            resolved_at = datetime('now'),
            resolved_by = ?
        WHERE id = ?
        """,
        (resolution, resolved_anchor_id, resolved_by, conflict_id)
    )

    db.conn.commit()

    return {
        "success": True,
        "conflict_id": conflict_id,
        "resolution": resolution,
        "resolved_anchor_id": resolved_anchor_id,
        "message": f"Conflict resolved: {resolution}"
    }


async def get_anchor_history(
    db: DatabaseService,
    anchor_id: Optional[int] = None,
    session_id: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """Get anchor history for tracking fact evolution."""
    cursor = db.conn.cursor()

    query = "SELECT * FROM anchor_history WHERE 1=1"
    params = []

    if anchor_id:
        query += " AND anchor_id = ?"
        params.append(anchor_id)

    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()

    history = [dict(row) for row in rows] if rows else []

    return {
        "success": True,
        "history": history,
        "count": len(history)
    }


async def auto_resolve_conflicts(
    db: DatabaseService,
    embeddings: EmbeddingService,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Attempt automatic resolution of simple conflicts.

    Auto-resolves:
    - Identical duplicates (keep newer)
    - Clear updates (same subject, newer timestamp wins)

    Args:
        db: Database service
        embeddings: Embeddings service
        session_id: Filter to specific session

    Returns:
        Resolution results
    """
    conflicts = await get_unresolved_conflicts(db, session_id)
    resolved = 0
    skipped = 0

    for conflict in conflicts.get("conflicts", []):
        conflict_type = conflict.get("conflict_type", "")

        # Auto-resolve identical duplicates
        if conflict_type == "identical":
            await resolve_conflict(
                db=db,
                embeddings=embeddings,
                conflict_id=conflict["id"],
                resolution="keep_second",  # Keep newer
                resolved_by="auto"
            )
            resolved += 1

        # Auto-resolve clear updates
        elif conflict_type == "update":
            await resolve_conflict(
                db=db,
                embeddings=embeddings,
                conflict_id=conflict["id"],
                resolution="keep_second",  # Keep newer (update)
                resolved_by="auto"
            )
            resolved += 1

        else:
            # Skip conflicts that need human review
            skipped += 1

    return {
        "success": True,
        "resolved": resolved,
        "skipped": skipped,
        "message": f"Auto-resolved {resolved} conflicts, {skipped} need manual review"
    }
