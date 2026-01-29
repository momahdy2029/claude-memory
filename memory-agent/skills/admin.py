"""Admin skills for memory system management.

Provides:
- Embedding model switching
- Memory reindexing
- System statistics
- Background reindexing with progress tracking
"""
import asyncio
import time
from typing import Dict, Any, Optional, List
from services.embeddings import get_embedding_service, MODEL_CONFIGS


# Global reindex state for progress tracking
_reindex_state: Dict[str, Any] = {
    "running": False,
    "progress": 0,
    "total": 0,
    "current_model": None,
    "started_at": None,
    "errors": [],
    "completed_at": None
}


async def get_embedding_status(
    db,
    embeddings
) -> Dict[str, Any]:
    """Get current embedding service status.

    Returns:
        Status including model, health, and available models
    """
    status = embeddings.get_status()
    health = await embeddings.check_health()

    # Get memory stats by model
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT embedding_model, COUNT(*) as count
        FROM memories
        GROUP BY embedding_model
    """)
    model_counts = {row[0] or 'nomic-embed-text': row[1] for row in cursor.fetchall()}

    return {
        "success": True,
        "status": status,
        "health": health,
        "available_models": embeddings.get_available_models(),
        "memories_by_model": model_counts
    }


async def switch_embedding_model(
    db,
    embeddings,
    model: str,
    reindex_existing: bool = False
) -> Dict[str, Any]:
    """Switch the default embedding model.

    Args:
        db: Database service
        embeddings: Embedding service
        model: New model name to use
        reindex_existing: If True, queue background reindex of existing memories

    Returns:
        Switch result with optional reindex status
    """
    # Validate model
    if model not in MODEL_CONFIGS and model != "default":
        available = [k for k in MODEL_CONFIGS.keys() if "alias_for" not in MODEL_CONFIGS.get(k, {})]
        return {
            "success": False,
            "error": f"Unknown model '{model}'. Available: {available}"
        }

    old_model = embeddings.get_current_model()
    embeddings.set_model(model)
    new_model = embeddings.get_current_model()

    # Check if model is available in Ollama
    health = await embeddings.check_health(force=True)

    result = {
        "success": True,
        "old_model": old_model,
        "new_model": new_model,
        "new_dimension": embeddings.get_dimension(),
        "model_available": health.get("model_loaded", False),
        "message": f"Switched from {old_model} to {new_model}"
    }

    if not health.get("model_loaded", False):
        result["warning"] = f"Model '{new_model}' not found in Ollama. Run: ollama pull {new_model}"

    if reindex_existing:
        # Start background reindex
        asyncio.create_task(_background_reindex(db, embeddings, new_model))
        result["reindex_started"] = True
        result["message"] += ". Background reindexing started."

    return result


async def reindex_memories(
    db,
    embeddings,
    model: Optional[str] = None,
    project_path: Optional[str] = None,
    batch_size: int = 10,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Reindex memories with current or specified embedding model.

    Args:
        db: Database service
        embeddings: Embedding service
        model: Model to use (None = current model)
        project_path: Filter to specific project
        batch_size: Number of memories per batch
        dry_run: If True, only count what would be reindexed

    Returns:
        Reindex results
    """
    global _reindex_state

    if _reindex_state["running"]:
        return {
            "success": False,
            "error": "Reindex already in progress",
            "progress": _reindex_state
        }

    use_model = model or embeddings.get_current_model()

    # Count memories to reindex
    cursor = db.conn.cursor()
    query = "SELECT COUNT(*) FROM memories WHERE embedding_model IS NULL OR embedding_model != ?"
    params = [use_model]

    if project_path:
        query += " AND project_path = ?"
        params.append(project_path)

    cursor.execute(query, params)
    total = cursor.fetchone()[0]

    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "would_reindex": total,
            "target_model": use_model,
            "project_filter": project_path
        }

    if total == 0:
        return {
            "success": True,
            "message": "No memories need reindexing",
            "total": 0
        }

    # Start background reindex
    asyncio.create_task(_background_reindex(db, embeddings, use_model, project_path, batch_size))

    return {
        "success": True,
        "message": f"Background reindexing started for {total} memories",
        "total": total,
        "target_model": use_model,
        "project_filter": project_path
    }


async def get_reindex_progress(
    db,
    embeddings
) -> Dict[str, Any]:
    """Get current reindex progress.

    Returns:
        Progress status including completion percentage
    """
    global _reindex_state

    if not _reindex_state["running"] and _reindex_state["completed_at"] is None:
        return {
            "success": True,
            "status": "idle",
            "message": "No reindex in progress"
        }

    progress_pct = 0
    if _reindex_state["total"] > 0:
        progress_pct = round(_reindex_state["progress"] / _reindex_state["total"] * 100, 1)

    status = "running" if _reindex_state["running"] else "completed"

    result = {
        "success": True,
        "status": status,
        "progress": _reindex_state["progress"],
        "total": _reindex_state["total"],
        "progress_percent": progress_pct,
        "model": _reindex_state["current_model"],
        "started_at": _reindex_state["started_at"],
        "errors_count": len(_reindex_state["errors"])
    }

    if _reindex_state["completed_at"]:
        result["completed_at"] = _reindex_state["completed_at"]
        result["duration_seconds"] = _reindex_state["completed_at"] - _reindex_state["started_at"]

    if _reindex_state["errors"]:
        result["recent_errors"] = _reindex_state["errors"][-5:]  # Last 5 errors

    return result


async def cancel_reindex(
    db,
    embeddings
) -> Dict[str, Any]:
    """Cancel a running reindex operation.

    Returns:
        Cancellation result
    """
    global _reindex_state

    if not _reindex_state["running"]:
        return {
            "success": False,
            "error": "No reindex in progress"
        }

    _reindex_state["running"] = False
    _reindex_state["completed_at"] = time.time()

    return {
        "success": True,
        "message": f"Reindex cancelled at {_reindex_state['progress']}/{_reindex_state['total']}",
        "progress": _reindex_state["progress"],
        "total": _reindex_state["total"]
    }


async def get_model_info(
    db,
    embeddings,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Get detailed information about an embedding model.

    Args:
        db: Database service
        embeddings: Embedding service
        model: Model name (None = current model)

    Returns:
        Model details including dimension and availability
    """
    use_model = model or embeddings.get_current_model()

    config = MODEL_CONFIGS.get(use_model, {})
    if "alias_for" in config:
        use_model = config["alias_for"]
        config = MODEL_CONFIGS.get(use_model, {})

    # Check availability in Ollama
    ollama_models = await embeddings.get_ollama_models()
    is_available = any(use_model in m for m in ollama_models)

    # Count memories using this model
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM memories WHERE embedding_model = ?",
        [use_model]
    )
    memory_count = cursor.fetchone()[0]

    return {
        "success": True,
        "model": use_model,
        "dimension": config.get("dimension", 768),
        "description": config.get("description", "Unknown model"),
        "is_current": use_model == embeddings.get_current_model(),
        "available_in_ollama": is_available,
        "memory_count": memory_count,
        "pull_command": f"ollama pull {use_model}" if not is_available else None
    }


async def _background_reindex(
    db,
    embeddings,
    model: str,
    project_path: Optional[str] = None,
    batch_size: int = 10
):
    """Background task for reindexing memories.

    Updates global _reindex_state for progress tracking.
    """
    global _reindex_state

    _reindex_state = {
        "running": True,
        "progress": 0,
        "total": 0,
        "current_model": model,
        "started_at": time.time(),
        "errors": [],
        "completed_at": None
    }

    try:
        cursor = db.conn.cursor()

        # Get memories to reindex
        query = """
            SELECT id, content FROM memories
            WHERE embedding_model IS NULL OR embedding_model != ?
        """
        params = [model]

        if project_path:
            query += " AND project_path = ?"
            params.append(project_path)

        cursor.execute(query, params)
        memories = cursor.fetchall()
        _reindex_state["total"] = len(memories)

        # Process in batches
        for i in range(0, len(memories), batch_size):
            if not _reindex_state["running"]:
                break  # Cancelled

            batch = memories[i:i + batch_size]

            for memory_id, content in batch:
                if not _reindex_state["running"]:
                    break

                try:
                    # Generate new embedding
                    embedding = await embeddings.generate_embedding(content, model=model)

                    if embedding:
                        # Update memory with new embedding
                        import json
                        cursor.execute("""
                            UPDATE memories
                            SET embedding = ?, embedding_model = ?
                            WHERE id = ?
                        """, [json.dumps(embedding), model, memory_id])
                        db.conn.commit()
                    else:
                        _reindex_state["errors"].append({
                            "memory_id": memory_id,
                            "error": "Failed to generate embedding"
                        })

                except Exception as e:
                    _reindex_state["errors"].append({
                        "memory_id": memory_id,
                        "error": str(e)
                    })

                _reindex_state["progress"] += 1

            # Small delay between batches to avoid overwhelming Ollama
            await asyncio.sleep(0.1)

        _reindex_state["running"] = False
        _reindex_state["completed_at"] = time.time()

    except Exception as e:
        _reindex_state["running"] = False
        _reindex_state["completed_at"] = time.time()
        _reindex_state["errors"].append({
            "error": f"Reindex failed: {str(e)}"
        })


async def get_system_stats(
    db,
    embeddings
) -> Dict[str, Any]:
    """Get comprehensive system statistics.

    Returns:
        System-wide statistics including memory counts, models, and health
    """
    cursor = db.conn.cursor()

    # Memory counts
    cursor.execute("SELECT COUNT(*) FROM memories")
    total_memories = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM patterns")
    total_patterns = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM projects")
    total_projects = cursor.fetchone()[0]

    # Memories by type
    cursor.execute("""
        SELECT type, COUNT(*) FROM memories
        GROUP BY type
    """)
    memories_by_type = {row[0]: row[1] for row in cursor.fetchall()}

    # Memories by model
    cursor.execute("""
        SELECT embedding_model, COUNT(*) FROM memories
        GROUP BY embedding_model
    """)
    memories_by_model = {row[0] or 'nomic-embed-text': row[1] for row in cursor.fetchall()}

    # Recent activity
    cursor.execute("""
        SELECT COUNT(*) FROM memories
        WHERE created_at > datetime('now', '-24 hours')
    """)
    memories_24h = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM memories
        WHERE created_at > datetime('now', '-7 days')
    """)
    memories_7d = cursor.fetchone()[0]

    # Embedding health
    health = await embeddings.check_health()

    return {
        "success": True,
        "totals": {
            "memories": total_memories,
            "patterns": total_patterns,
            "projects": total_projects
        },
        "memories_by_type": memories_by_type,
        "memories_by_model": memories_by_model,
        "recent_activity": {
            "last_24h": memories_24h,
            "last_7d": memories_7d
        },
        "embedding_service": {
            "model": embeddings.get_current_model(),
            "dimension": embeddings.get_dimension(),
            "healthy": health.get("healthy", False),
            "degraded": embeddings.is_degraded()
        },
        "reindex_status": {
            "running": _reindex_state["running"],
            "progress": _reindex_state["progress"],
            "total": _reindex_state["total"]
        }
    }
