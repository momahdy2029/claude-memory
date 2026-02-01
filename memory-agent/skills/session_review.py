"""Session review skill for end-of-session memory verification.

Allows users to review memories created during a session and mark them as:
- keep: Verified as useful, increases confidence
- discard: Not useful, decreases confidence significantly
- partial: Partially useful, sets confidence to middle value
"""
from typing import Dict, Any, Optional, List
from services.database import DatabaseService
from services.embeddings import EmbeddingService


async def get_session_memories(
    db: DatabaseService,
    session_id: str,
    include_patterns: bool = False,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get all memories created in a specific session for review.

    Args:
        db: Database service instance
        session_id: Session identifier to filter memories
        include_patterns: Whether to include patterns created during session
        limit: Maximum number of memories to return

    Returns:
        Dict with memories list and metadata
    """
    if not session_id:
        return {
            "success": False,
            "error": "session_id is required"
        }

    # Get memories for this session
    memories_query = """
        SELECT
            id, type, content, project_path, project_name,
            outcome, success, outcome_status, confidence,
            importance, tags, created_at
        FROM memories
        WHERE session_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """

    memories = await db.execute_query(memories_query, [session_id, limit])

    result = {
        "success": True,
        "session_id": session_id,
        "memories": memories or [],
        "memory_count": len(memories) if memories else 0
    }

    # Optionally include patterns
    if include_patterns:
        patterns_query = """
            SELECT
                id, name, problem_type, solution,
                success_count, failure_count, created_at
            FROM patterns
            WHERE created_at >= (
                SELECT MIN(created_at) FROM memories WHERE session_id = ?
            )
            AND created_at <= (
                SELECT MAX(created_at) FROM memories WHERE session_id = ?
            )
            ORDER BY created_at DESC
            LIMIT ?
        """
        patterns = await db.execute_query(patterns_query, [session_id, session_id, limit])
        result["patterns"] = patterns or []
        result["pattern_count"] = len(patterns) if patterns else 0

    # Generate summary
    type_counts = {}
    for m in (memories or []):
        mtype = m.get("type", "unknown")
        type_counts[mtype] = type_counts.get(mtype, 0) + 1

    result["summary"] = {
        "by_type": type_counts,
        "total_memories": result["memory_count"],
        "avg_importance": (
            sum(m.get("importance", 5) for m in (memories or [])) / len(memories)
            if memories else 0
        ),
        "avg_confidence": (
            sum(m.get("confidence", 0.5) for m in (memories or [])) / len(memories)
            if memories else 0
        )
    }

    return result


async def review_session_memories(
    db: DatabaseService,
    session_id: str,
    reviews: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Process user review decisions for session memories.

    Args:
        db: Database service instance
        session_id: Session identifier
        reviews: List of review decisions, each containing:
            - memory_id: ID of the memory
            - decision: 'keep', 'discard', or 'partial'
            - feedback: Optional user feedback

    Returns:
        Dict with processing results
    """
    if not session_id:
        return {
            "success": False,
            "error": "session_id is required"
        }

    if not reviews:
        return {
            "success": False,
            "error": "reviews list is required"
        }

    results = {
        "success": True,
        "session_id": session_id,
        "processed": 0,
        "kept": 0,
        "discarded": 0,
        "partial": 0,
        "errors": []
    }

    # Confidence mappings for each decision
    confidence_map = {
        "keep": 0.9,      # High confidence - verified useful
        "partial": 0.5,   # Medium confidence - partially useful
        "discard": 0.1    # Low confidence - not useful
    }

    outcome_status_map = {
        "keep": "success",
        "partial": "partial",
        "discard": "failed"
    }

    for review in reviews:
        memory_id = review.get("memory_id")
        decision = review.get("decision", "keep").lower()
        feedback = review.get("feedback")

        if not memory_id:
            results["errors"].append({
                "error": "memory_id is required",
                "review": review
            })
            continue

        if decision not in confidence_map:
            results["errors"].append({
                "memory_id": memory_id,
                "error": f"Invalid decision: {decision}. Must be 'keep', 'discard', or 'partial'"
            })
            continue

        try:
            # Update confidence
            new_confidence = confidence_map[decision]
            await db.update_memory_confidence(memory_id, new_confidence)

            # Update outcome status
            new_outcome_status = outcome_status_map[decision]
            await db.update_memory_outcome(
                memory_id=memory_id,
                outcome_status=new_outcome_status
            )

            # Store feedback if provided
            if feedback:
                await db.execute_query(
                    "UPDATE memories SET user_feedback = ? WHERE id = ?",
                    [feedback, memory_id]
                )

            results["processed"] += 1

            if decision == "keep":
                results["kept"] += 1
            elif decision == "discard":
                results["discarded"] += 1
            elif decision == "partial":
                results["partial"] += 1

        except Exception as e:
            results["errors"].append({
                "memory_id": memory_id,
                "error": str(e)
            })

    # Calculate success rate
    total = results["kept"] + results["discarded"] + results["partial"]
    if total > 0:
        results["keep_rate"] = round(results["kept"] / total * 100, 1)
    else:
        results["keep_rate"] = 0

    return results


async def suggest_session_reviews(
    db: DatabaseService,
    embeddings: EmbeddingService,
    session_id: str
) -> Dict[str, Any]:
    """
    Analyze session memories and suggest which to keep/discard.

    Suggestions are based on:
    - Memory type (decisions and patterns more likely to keep)
    - Importance score
    - Outcome status if already set
    - Similarity to existing successful memories

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        session_id: Session identifier

    Returns:
        Dict with suggested reviews
    """
    if not session_id:
        return {
            "success": False,
            "error": "session_id is required"
        }

    # Get session memories
    memories_result = await get_session_memories(db, session_id)
    if not memories_result.get("success"):
        return memories_result

    memories = memories_result.get("memories", [])
    suggestions = []

    # Types that are more likely to be valuable
    high_value_types = {"decision", "error", "pattern", "preference"}

    for memory in memories:
        memory_id = memory.get("id")
        memory_type = memory.get("type", "chunk")
        importance = memory.get("importance", 5)
        outcome_status = memory.get("outcome_status", "pending")
        confidence = memory.get("confidence", 0.5)
        content = memory.get("content", "")

        suggestion = {
            "memory_id": memory_id,
            "type": memory_type,
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "current_confidence": confidence,
            "importance": importance
        }

        # Determine suggestion based on heuristics
        if outcome_status == "success":
            suggestion["suggested_decision"] = "keep"
            suggestion["reason"] = "Already marked as successful"
        elif outcome_status == "failed":
            suggestion["suggested_decision"] = "discard"
            suggestion["reason"] = "Already marked as failed"
        elif outcome_status == "partial":
            suggestion["suggested_decision"] = "partial"
            suggestion["reason"] = "Already marked as partial success"
        elif memory_type in high_value_types:
            if importance >= 7:
                suggestion["suggested_decision"] = "keep"
                suggestion["reason"] = f"High importance {memory_type}"
            else:
                suggestion["suggested_decision"] = "partial"
                suggestion["reason"] = f"Review this {memory_type}"
        elif importance >= 8:
            suggestion["suggested_decision"] = "keep"
            suggestion["reason"] = "High importance score"
        elif importance <= 3:
            suggestion["suggested_decision"] = "discard"
            suggestion["reason"] = "Low importance score"
        else:
            suggestion["suggested_decision"] = "partial"
            suggestion["reason"] = "Review recommended"

        suggestions.append(suggestion)

    return {
        "success": True,
        "session_id": session_id,
        "suggestions": suggestions,
        "summary": {
            "total": len(suggestions),
            "suggested_keep": sum(1 for s in suggestions if s["suggested_decision"] == "keep"),
            "suggested_discard": sum(1 for s in suggestions if s["suggested_decision"] == "discard"),
            "suggested_partial": sum(1 for s in suggestions if s["suggested_decision"] == "partial")
        }
    }


async def get_recent_sessions(
    db: DatabaseService,
    project_path: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get recent sessions with memory counts for review selection.

    Args:
        db: Database service instance
        project_path: Optional filter by project
        limit: Maximum number of sessions to return

    Returns:
        Dict with session list
    """
    query = """
        SELECT
            session_id,
            COUNT(*) as memory_count,
            MIN(created_at) as started_at,
            MAX(created_at) as ended_at,
            project_path,
            project_name,
            SUM(CASE WHEN outcome_status = 'success' THEN 1 ELSE 0 END) as success_count,
            SUM(CASE WHEN outcome_status = 'failed' THEN 1 ELSE 0 END) as failed_count,
            SUM(CASE WHEN outcome_status = 'pending' THEN 1 ELSE 0 END) as pending_count,
            AVG(confidence) as avg_confidence
        FROM memories
        WHERE session_id IS NOT NULL
    """
    params = []

    if project_path:
        query += " AND project_path = ?"
        params.append(project_path)

    query += """
        GROUP BY session_id
        ORDER BY MAX(created_at) DESC
        LIMIT ?
    """
    params.append(limit)

    sessions = await db.execute_query(query, params)

    return {
        "success": True,
        "sessions": sessions or [],
        "count": len(sessions) if sessions else 0
    }


async def bulk_review_by_type(
    db: DatabaseService,
    session_id: str,
    type_decisions: Dict[str, str]
) -> Dict[str, Any]:
    """
    Apply review decisions to all memories of specific types in a session.

    Useful for quickly processing sessions, e.g.:
    - Keep all 'decision' and 'error' memories
    - Discard all 'chunk' memories

    Args:
        db: Database service instance
        session_id: Session identifier
        type_decisions: Dict mapping memory types to decisions
            e.g., {"decision": "keep", "chunk": "discard", "error": "keep"}

    Returns:
        Dict with processing results
    """
    if not session_id:
        return {
            "success": False,
            "error": "session_id is required"
        }

    if not type_decisions:
        return {
            "success": False,
            "error": "type_decisions is required"
        }

    # Get session memories
    memories_result = await get_session_memories(db, session_id)
    if not memories_result.get("success"):
        return memories_result

    memories = memories_result.get("memories", [])

    # Build reviews based on type decisions
    reviews = []
    for memory in memories:
        memory_type = memory.get("type", "chunk")
        if memory_type in type_decisions:
            reviews.append({
                "memory_id": memory.get("id"),
                "decision": type_decisions[memory_type]
            })

    # Process reviews
    return await review_session_memories(db, session_id, reviews)
