"""Self-correcting confidence tracker for memory reliability.

Tracks solution outcomes and adjusts confidence automatically:
- When a solution works: increase confidence by 0.15 (max 1.0)
- When a solution fails: decrease confidence by 0.2 (min 0.0)
- After 3 consecutive failures: mark as unreliable

This creates a learning loop where frequently successful solutions
gain trust while failed solutions are demoted automatically.
"""
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


# Confidence adjustment constants
CONFIDENCE_INCREASE = 0.15  # Boost when solution works
CONFIDENCE_DECREASE = 0.20  # Penalty when solution fails
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
UNRELIABLE_CONFIDENCE = 0.1  # Confidence floor for unreliable memories
MAX_CONSECUTIVE_FAILURES = 3  # Mark unreliable after this many failures


async def report_solution_outcome(
    db,
    memory_id: int,
    worked: bool,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Report whether a solution from memory worked or failed.

    This is the core feedback mechanism for self-correcting confidence.

    Args:
        db: Database service instance
        memory_id: ID of the memory to update
        worked: True if solution worked, False if it failed
        context: Optional context about the usage (environment, problem, etc.)

    Returns:
        Dict with updated confidence, failure_count, and reliability status

    Behavior:
        - worked=True: confidence += 0.15, failure_count reset to 0
        - worked=False: confidence -= 0.20, failure_count += 1
        - After 3 consecutive failures: mark as unreliable (outcome_status='failed')
    """
    cursor = db.conn.cursor()

    # Get current memory state
    cursor.execute("""
        SELECT id, confidence, outcome_status, failure_count,
               times_worked, times_failed, metadata
        FROM memories WHERE id = ?
    """, [memory_id])

    row = cursor.fetchone()
    if not row:
        return {
            "success": False,
            "error": f"Memory with ID {memory_id} not found",
            "error_code": "MEMORY_NOT_FOUND"
        }

    # Extract current values (handle missing columns gracefully)
    current_confidence = row["confidence"] if row["confidence"] is not None else 0.5
    current_failure_count = row["failure_count"] if "failure_count" in row.keys() and row["failure_count"] is not None else 0
    times_worked = row["times_worked"] if "times_worked" in row.keys() and row["times_worked"] is not None else 0
    times_failed = row["times_failed"] if "times_failed" in row.keys() and row["times_failed"] is not None else 0
    current_outcome_status = row["outcome_status"] if "outcome_status" in row.keys() else "pending"

    # Load metadata
    try:
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
    except (json.JSONDecodeError, TypeError):
        metadata = {}

    # Calculate new values
    if worked:
        # Solution worked - boost confidence and reset failure streak
        new_confidence = min(MAX_CONFIDENCE, current_confidence + CONFIDENCE_INCREASE)
        new_failure_count = 0  # Reset consecutive failures
        times_worked += 1

        # Update outcome status to success if not already set or was pending
        if current_outcome_status in ('pending', 'partial', None):
            new_outcome_status = 'success'
        else:
            new_outcome_status = current_outcome_status

        action = "boosted"
        message = f"Solution worked! Confidence increased from {current_confidence:.3f} to {new_confidence:.3f}"
    else:
        # Solution failed - decrease confidence and increment failure streak
        new_confidence = max(MIN_CONFIDENCE, current_confidence - CONFIDENCE_DECREASE)
        new_failure_count = current_failure_count + 1
        times_failed += 1

        # Check if memory should be marked as unreliable
        if new_failure_count >= MAX_CONSECUTIVE_FAILURES:
            new_confidence = UNRELIABLE_CONFIDENCE
            new_outcome_status = 'failed'
            action = "marked_unreliable"
            message = f"Memory marked as unreliable after {new_failure_count} consecutive failures"
            logger.warning(f"Memory {memory_id} marked unreliable: {new_failure_count} consecutive failures")
        else:
            new_outcome_status = current_outcome_status if current_outcome_status != 'success' else 'partial'
            action = "penalized"
            message = f"Solution failed. Confidence decreased from {current_confidence:.3f} to {new_confidence:.3f}"

    # Record outcome in metadata history
    outcome_history = metadata.get("outcome_history", [])
    outcome_history.append({
        "timestamp": datetime.now().isoformat(),
        "worked": worked,
        "confidence_before": current_confidence,
        "confidence_after": new_confidence,
        "context": context
    })
    # Keep last 20 outcomes to avoid unbounded growth
    metadata["outcome_history"] = outcome_history[-20:]
    metadata["last_outcome"] = {
        "worked": worked,
        "timestamp": datetime.now().isoformat()
    }

    # Update the memory
    cursor.execute("""
        UPDATE memories SET
            confidence = ?,
            failure_count = ?,
            times_worked = ?,
            times_failed = ?,
            outcome_status = ?,
            metadata = ?,
            updated_at = datetime('now')
        WHERE id = ?
    """, [
        new_confidence,
        new_failure_count,
        times_worked,
        times_failed,
        new_outcome_status,
        json.dumps(metadata),
        memory_id
    ])
    db.conn.commit()

    # Calculate reliability classification
    reliability = _classify_reliability(new_confidence, new_failure_count, times_worked, times_failed)

    return {
        "success": True,
        "memory_id": memory_id,
        "action": action,
        "message": message,
        "old_confidence": current_confidence,
        "new_confidence": new_confidence,
        "failure_count": new_failure_count,
        "times_worked": times_worked,
        "times_failed": times_failed,
        "outcome_status": new_outcome_status,
        "reliability": reliability,
        "is_unreliable": new_failure_count >= MAX_CONSECUTIVE_FAILURES
    }


async def get_reliability_stats(
    db,
    memory_id: int
) -> Dict[str, Any]:
    """Get detailed reliability statistics for a memory.

    Returns comprehensive reliability information including:
    - Current confidence score
    - Usage statistics (times worked/failed)
    - Failure streak count
    - Reliability classification
    - Outcome history

    Args:
        db: Database service instance
        memory_id: ID of the memory to analyze

    Returns:
        Dict with full reliability stats and history
    """
    cursor = db.conn.cursor()

    cursor.execute("""
        SELECT id, content, type, confidence, outcome_status,
               failure_count, times_worked, times_failed,
               created_at, updated_at, metadata
        FROM memories WHERE id = ?
    """, [memory_id])

    row = cursor.fetchone()
    if not row:
        return {
            "success": False,
            "error": f"Memory with ID {memory_id} not found",
            "error_code": "MEMORY_NOT_FOUND"
        }

    # Extract values (handle missing columns gracefully)
    confidence = row["confidence"] if row["confidence"] is not None else 0.5
    failure_count = row["failure_count"] if "failure_count" in row.keys() and row["failure_count"] is not None else 0
    times_worked = row["times_worked"] if "times_worked" in row.keys() and row["times_worked"] is not None else 0
    times_failed = row["times_failed"] if "times_failed" in row.keys() and row["times_failed"] is not None else 0
    outcome_status = row["outcome_status"] if "outcome_status" in row.keys() else "pending"

    # Load metadata for outcome history
    try:
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
    except (json.JSONDecodeError, TypeError):
        metadata = {}

    outcome_history = metadata.get("outcome_history", [])
    last_outcome = metadata.get("last_outcome")

    # Calculate reliability classification
    reliability = _classify_reliability(confidence, failure_count, times_worked, times_failed)

    # Calculate success rate
    total_uses = times_worked + times_failed
    success_rate = (times_worked / total_uses) if total_uses > 0 else None

    return {
        "success": True,
        "memory_id": memory_id,
        "content_preview": row["content"][:200] if row["content"] else None,
        "type": row["type"],
        "confidence": confidence,
        "times_worked": times_worked,
        "times_failed": times_failed,
        "total_uses": total_uses,
        "success_rate": round(success_rate, 3) if success_rate is not None else None,
        "failure_count": failure_count,
        "consecutive_failures": failure_count,  # Same as failure_count for clarity
        "outcome_status": outcome_status,
        "reliability": reliability,
        "is_unreliable": failure_count >= MAX_CONSECUTIVE_FAILURES,
        "last_outcome": last_outcome,
        "outcome_history": outcome_history,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "interpretation": _interpret_reliability(reliability, confidence, failure_count)
    }


async def get_unreliable_memories(
    db,
    project_path: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """Get all memories marked as unreliable.

    Args:
        db: Database service instance
        project_path: Optional filter by project
        limit: Maximum number of results

    Returns:
        Dict with list of unreliable memories
    """
    from services.database import normalize_path

    cursor = db.conn.cursor()

    query = """
        SELECT id, content, type, confidence, outcome_status,
               failure_count, times_worked, times_failed, project_path,
               created_at, updated_at
        FROM memories
        WHERE (failure_count >= ? OR outcome_status = 'failed')
    """
    params = [MAX_CONSECUTIVE_FAILURES]

    if project_path:
        project_path = normalize_path(project_path)
        query += " AND project_path = ?"
        params.append(project_path)

    query += " ORDER BY failure_count DESC, updated_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()

    memories = []
    for row in rows:
        confidence = row["confidence"] if row["confidence"] is not None else 0.5
        failure_count = row["failure_count"] if "failure_count" in row.keys() and row["failure_count"] is not None else 0
        times_worked = row["times_worked"] if "times_worked" in row.keys() and row["times_worked"] is not None else 0
        times_failed = row["times_failed"] if "times_failed" in row.keys() and row["times_failed"] is not None else 0

        memories.append({
            "id": row["id"],
            "content_preview": row["content"][:200] if row["content"] else None,
            "type": row["type"],
            "confidence": confidence,
            "failure_count": failure_count,
            "times_worked": times_worked,
            "times_failed": times_failed,
            "outcome_status": row["outcome_status"],
            "project_path": row["project_path"],
            "reliability": _classify_reliability(confidence, failure_count, times_worked, times_failed),
            "updated_at": row["updated_at"]
        })

    return {
        "success": True,
        "unreliable_memories": memories,
        "count": len(memories),
        "project_path": project_path
    }


async def reset_memory_reliability(
    db,
    memory_id: int,
    new_confidence: float = 0.5
) -> Dict[str, Any]:
    """Reset a memory's reliability stats (admin function).

    Useful when a memory has been fixed or updated and should be
    given a fresh chance.

    Args:
        db: Database service instance
        memory_id: ID of the memory to reset
        new_confidence: Starting confidence (default 0.5)

    Returns:
        Dict with reset status
    """
    cursor = db.conn.cursor()

    # Verify memory exists
    cursor.execute("SELECT id FROM memories WHERE id = ?", [memory_id])
    if not cursor.fetchone():
        return {
            "success": False,
            "error": f"Memory with ID {memory_id} not found",
            "error_code": "MEMORY_NOT_FOUND"
        }

    # Clamp confidence to valid range
    new_confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, new_confidence))

    cursor.execute("""
        UPDATE memories SET
            confidence = ?,
            failure_count = 0,
            times_worked = 0,
            times_failed = 0,
            outcome_status = 'pending',
            updated_at = datetime('now')
        WHERE id = ?
    """, [new_confidence, memory_id])
    db.conn.commit()

    return {
        "success": True,
        "memory_id": memory_id,
        "message": "Reliability stats reset",
        "new_confidence": new_confidence,
        "failure_count": 0,
        "outcome_status": "pending"
    }


def _classify_reliability(
    confidence: float,
    failure_count: int,
    times_worked: int,
    times_failed: int
) -> str:
    """Classify reliability based on confidence and usage stats.

    Returns:
        One of: 'proven', 'high', 'moderate', 'low', 'unreliable', 'untested'
    """
    total_uses = times_worked + times_failed

    if total_uses == 0:
        return "untested"

    if failure_count >= MAX_CONSECUTIVE_FAILURES:
        return "unreliable"

    if confidence >= 0.85 and times_worked >= 3:
        return "proven"
    elif confidence >= 0.7:
        return "high"
    elif confidence >= 0.5:
        return "moderate"
    elif confidence >= 0.3:
        return "low"
    else:
        return "unreliable"


def _interpret_reliability(
    reliability: str,
    confidence: float,
    failure_count: int
) -> str:
    """Human-readable interpretation of reliability status."""
    interpretations = {
        "proven": "This solution has been repeatedly verified and is highly reliable.",
        "high": "This solution has a good track record and can be trusted.",
        "moderate": "This solution may work, but consider verifying before relying on it.",
        "low": "This solution has mixed results. Use with caution.",
        "unreliable": f"This solution has failed {failure_count} times consecutively. Consider alternatives.",
        "untested": "This solution has not been tested yet. Report outcome after use."
    }
    return interpretations.get(reliability, "Unknown reliability status")


# Export functions that integrate with MCP
async def memory_worked(
    db,
    memory_id: int,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """MCP-friendly wrapper: Report that a memory solution worked."""
    return await report_solution_outcome(db, memory_id, worked=True, context=context)


async def memory_failed(
    db,
    memory_id: int,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """MCP-friendly wrapper: Report that a memory solution failed."""
    return await report_solution_outcome(db, memory_id, worked=False, context=context)
