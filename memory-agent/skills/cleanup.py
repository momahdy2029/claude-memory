"""Skills for memory cleanup and maintenance.

Provides:
- Manual cleanup triggers
- Dry-run preview
- Archive management
- Configuration management
"""
from typing import Dict, Any, Optional, List
from services.cleanup import get_cleanup_service


async def memory_cleanup(
    db,
    embeddings,
    project_path: Optional[str] = None,
    dry_run: bool = True
) -> Dict[str, Any]:
    """Run memory cleanup with optional preview mode.

    Cleans up:
    - Low-relevance memories (below threshold)
    - Expired memories (older than retention period)
    - Duplicate memories (merged by similarity)

    Args:
        db: Database service
        embeddings: Embeddings service
        project_path: Filter to specific project (None = all)
        dry_run: If True, only preview what would be cleaned

    Returns:
        Cleanup results with counts and details
    """
    cleanup = get_cleanup_service(db, embeddings)
    result = await cleanup.run_cleanup(
        project_path=project_path,
        dry_run=dry_run
    )

    # Add helpful message
    if dry_run:
        result["message"] = (
            f"DRY RUN: Would archive {result['total_archived']} memories, "
            f"delete {result['total_deleted']}, merge {result['total_merged']} duplicates. "
            f"Run with dry_run=False to execute."
        )
    else:
        result["message"] = (
            f"Cleanup complete: Archived {result['total_archived']}, "
            f"deleted {result['total_deleted']}, merged {result['total_merged']} duplicates."
        )

    return result


async def get_archived_memories(
    db,
    embeddings,
    project_path: Optional[str] = None,
    reason: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """Get archived memories that can be restored.

    Args:
        db: Database service
        embeddings: Embeddings service (not used but kept for consistency)
        project_path: Filter by project
        reason: Filter by archive reason (low_relevance, expired, duplicate)
        limit: Maximum results

    Returns:
        List of archived memories
    """
    cleanup = get_cleanup_service(db, embeddings)
    archives = await cleanup.get_archived_memories(
        project_path=project_path,
        reason=reason,
        limit=limit
    )

    return {
        "success": True,
        "archives": archives,
        "count": len(archives),
        "filters": {
            "project_path": project_path,
            "reason": reason
        }
    }


async def restore_memory(
    db,
    embeddings,
    archive_id: int
) -> Dict[str, Any]:
    """Restore an archived memory back to active storage.

    Args:
        db: Database service
        embeddings: Embeddings service
        archive_id: ID of the archived memory to restore

    Returns:
        Restoration result with new memory ID
    """
    cleanup = get_cleanup_service(db, embeddings)
    return await cleanup.restore_memory(archive_id)


async def get_cleanup_config(
    db,
    embeddings,
    project_path: Optional[str] = None
) -> Dict[str, Any]:
    """Get cleanup configuration for a project.

    Args:
        db: Database service
        embeddings: Embeddings service
        project_path: Project to get config for (None = defaults)

    Returns:
        Cleanup configuration settings
    """
    cleanup = get_cleanup_service(db, embeddings)
    config = await cleanup.get_config(project_path)

    return {
        "success": True,
        "project_path": project_path,
        "config": config
    }


async def set_cleanup_config(
    db,
    embeddings,
    project_path: Optional[str] = None,
    retention_days: Optional[int] = None,
    min_relevance_score: Optional[float] = None,
    keep_high_importance: Optional[bool] = None,
    importance_threshold: Optional[int] = None,
    dedup_enabled: Optional[bool] = None,
    dedup_threshold: Optional[float] = None,
    archive_before_delete: Optional[bool] = None,
    auto_cleanup_enabled: Optional[bool] = None
) -> Dict[str, Any]:
    """Update cleanup configuration for a project.

    Args:
        db: Database service
        embeddings: Embeddings service
        project_path: Project to configure
        retention_days: Days to keep memories before cleanup
        min_relevance_score: Minimum relevance score to keep
        keep_high_importance: Whether to protect high-importance memories
        importance_threshold: What counts as "high importance"
        dedup_enabled: Whether to deduplicate
        dedup_threshold: Similarity threshold for duplicates
        archive_before_delete: Whether to archive before deleting
        auto_cleanup_enabled: Whether to run automatic cleanup

    Returns:
        Updated configuration
    """
    cleanup = get_cleanup_service(db, embeddings)

    # Get current config as base
    current = await cleanup.get_config(project_path)

    # Update with provided values
    if retention_days is not None:
        current["retention_days"] = retention_days
    if min_relevance_score is not None:
        current["min_relevance_score"] = min_relevance_score
    if keep_high_importance is not None:
        current["keep_high_importance"] = keep_high_importance
    if importance_threshold is not None:
        current["importance_threshold"] = importance_threshold
    if dedup_enabled is not None:
        current["dedup_enabled"] = dedup_enabled
    if dedup_threshold is not None:
        current["dedup_threshold"] = dedup_threshold
    if archive_before_delete is not None:
        current["archive_before_delete"] = archive_before_delete
    if auto_cleanup_enabled is not None:
        current["auto_cleanup_enabled"] = auto_cleanup_enabled

    await cleanup.save_config(project_path, current)

    return {
        "success": True,
        "project_path": project_path,
        "config": current,
        "message": "Configuration updated"
    }


async def get_cleanup_stats(
    db,
    embeddings
) -> Dict[str, Any]:
    """Get overall cleanup statistics.

    Args:
        db: Database service
        embeddings: Embeddings service

    Returns:
        Cleanup statistics including recent activity
    """
    cleanup = get_cleanup_service(db, embeddings)
    stats = await cleanup.get_cleanup_stats()

    return {
        "success": True,
        "stats": stats
    }


async def purge_expired_archives(
    db,
    embeddings
) -> Dict[str, Any]:
    """Permanently delete archived memories past their expiration.

    This action is irreversible. Only call when you're sure
    you want to permanently remove old archives.

    Args:
        db: Database service
        embeddings: Embeddings service

    Returns:
        Purge results
    """
    cleanup = get_cleanup_service(db, embeddings)
    return await cleanup.purge_expired_archives()
