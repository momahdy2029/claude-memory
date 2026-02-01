"""Curator MCP Skills - Graph exploration and maintenance capabilities."""
import logging
from typing import Dict, Any, Optional, List

from services.database import DatabaseService
from services.embeddings import EmbeddingService
from services.curator import get_curator

logger = logging.getLogger(__name__)


async def curator_explore(
    db: DatabaseService,
    embeddings: EmbeddingService,
    start_node_id: int,
    max_depth: int = 3,
    mode: str = "bfs",
    relationship_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Explore the memory graph from a starting node.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        start_node_id: ID of the memory to start from
        max_depth: Maximum traversal depth (default 3)
        mode: 'bfs' (breadth-first) or 'dfs' (depth-first)
        relationship_filter: Only follow these relationship types

    Returns:
        Dict with explored nodes, edges, clusters, and insights
    """
    curator = get_curator(db, embeddings)
    return await curator.explore_graph(
        start_node_id=start_node_id,
        max_depth=max_depth,
        mode=mode,
        relationship_filter=relationship_filter
    )


async def curator_find_duplicates(
    db: DatabaseService,
    embeddings: EmbeddingService,
    project_path: Optional[str] = None,
    similarity_threshold: float = 0.92,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Find semantically similar (duplicate) memories.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        project_path: Optional project filter
        similarity_threshold: Minimum similarity to consider duplicates
        limit: Maximum number of duplicate pairs to return

    Returns:
        Dict with duplicate clusters and merge suggestions
    """
    curator = get_curator(db, embeddings)
    return await curator.find_duplicates(
        project_path=project_path,
        similarity_threshold=similarity_threshold,
        limit=limit
    )


async def curator_suggest_links(
    db: DatabaseService,
    embeddings: EmbeddingService,
    memory_id: Optional[int] = None,
    project_path: Optional[str] = None,
    similarity_threshold: float = 0.7,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Suggest missing relationships between memories.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        memory_id: Optional specific memory to find links for
        project_path: Optional project filter
        similarity_threshold: Minimum similarity for suggestions
        limit: Maximum suggestions to return

    Returns:
        Dict with suggested relationships
    """
    curator = get_curator(db, embeddings)
    return await curator.suggest_relationships(
        memory_id=memory_id,
        project_path=project_path,
        similarity_threshold=similarity_threshold,
        limit=limit
    )


async def curator_merge(
    db: DatabaseService,
    embeddings: EmbeddingService,
    keep_id: int,
    remove_ids: List[int],
    merge_content: bool = False
) -> Dict[str, Any]:
    """
    Merge duplicate memories into one.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        keep_id: Memory ID to keep
        remove_ids: Memory IDs to merge into keep_id
        merge_content: If True, append removed content to kept memory

    Returns:
        Dict with merge result
    """
    curator = get_curator(db, embeddings)
    return await curator.merge_memories(
        keep_id=keep_id,
        remove_ids=remove_ids,
        merge_content=merge_content
    )


async def curator_get_summary(
    db: DatabaseService,
    embeddings: EmbeddingService,
    query: str,
    project_path: Optional[str] = None,
    max_memories: int = 10,
    include_graph: bool = True
) -> Dict[str, Any]:
    """
    Generate curated context summary for a query.

    This is what gets injected into the main Claude's context.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        query: The topic/query to generate context for
        project_path: Optional project filter
        max_memories: Maximum memories to include
        include_graph: Include relationship graph context

    Returns:
        Dict with curated context summary
    """
    curator = get_curator(db, embeddings)
    return await curator.generate_summary(
        query=query,
        project_path=project_path,
        max_memories=max_memories,
        include_graph=include_graph
    )


async def curator_run_maintenance(
    db: DatabaseService,
    embeddings: EmbeddingService,
    project_path: Optional[str] = None,
    tasks: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run curator maintenance tasks.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        project_path: Optional project filter
        tasks: Specific tasks to run, or None for all
               Options: dedup, orphans, links, decay, quality

    Returns:
        Dict with maintenance report
    """
    curator = get_curator(db, embeddings)
    return await curator.run_maintenance(
        project_path=project_path,
        tasks=tasks
    )


async def curator_get_report(
    db: DatabaseService,
    embeddings: EmbeddingService,
    project_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get the latest curator report.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        project_path: Optional project filter

    Returns:
        Dict with latest report or None
    """
    curator = get_curator(db, embeddings)
    report = await curator.get_latest_report(project_path=project_path)
    if report:
        return report
    return {"message": "No curator reports found"}


async def curator_get_status(
    db: DatabaseService,
    embeddings: EmbeddingService
) -> Dict[str, Any]:
    """
    Get current curator agent status.

    Args:
        db: Database service instance
        embeddings: Embedding service instance

    Returns:
        Dict with curator status
    """
    curator = get_curator(db, embeddings)
    return await curator.get_status()


async def curator_score_quality(
    db: DatabaseService,
    embeddings: EmbeddingService,
    memory_id: Optional[int] = None,
    project_path: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Calculate quality scores for memories.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        memory_id: Optional specific memory to score
        project_path: Optional project filter
        limit: Maximum memories to score

    Returns:
        Dict with quality scores and insights
    """
    curator = get_curator(db, embeddings)
    return await curator.score_quality(
        memory_id=memory_id,
        project_path=project_path,
        limit=limit
    )


async def curator_find_orphans(
    db: DatabaseService,
    embeddings: EmbeddingService,
    project_path: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Find memories with no relationships (orphans).

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        project_path: Optional project filter
        limit: Maximum orphans to return

    Returns:
        Dict with orphan memories
    """
    curator = get_curator(db, embeddings)
    orphans = await curator.find_orphan_memories(
        project_path=project_path,
        limit=limit
    )
    return {
        "orphans": orphans,
        "total_found": len(orphans),
        "recommendation": "Consider linking these memories or archiving if no longer relevant"
    }


async def curator_get_config(
    db: DatabaseService,
    embeddings: EmbeddingService,
    project_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get curator configuration.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        project_path: Optional project path

    Returns:
        Dict with curator configuration
    """
    curator = get_curator(db, embeddings)
    return await curator.get_config(project_path=project_path)


async def curator_update_config(
    db: DatabaseService,
    embeddings: EmbeddingService,
    project_path: str,
    auto_dedup_enabled: Optional[bool] = None,
    auto_link_enabled: Optional[bool] = None,
    dedup_threshold: Optional[float] = None,
    maintenance_interval_hours: Optional[int] = None,
    curator_active: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Update curator configuration for a project.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        project_path: Project path to update config for
        auto_dedup_enabled: Enable auto-deduplication
        auto_link_enabled: Enable auto-linking
        dedup_threshold: Similarity threshold for duplicates
        maintenance_interval_hours: Hours between maintenance runs
        curator_active: Enable/disable curator

    Returns:
        Dict with updated configuration
    """
    curator = get_curator(db, embeddings)

    config_updates = {}
    if auto_dedup_enabled is not None:
        config_updates["auto_dedup_enabled"] = auto_dedup_enabled
    if auto_link_enabled is not None:
        config_updates["auto_link_enabled"] = auto_link_enabled
    if dedup_threshold is not None:
        config_updates["dedup_threshold"] = dedup_threshold
    if maintenance_interval_hours is not None:
        config_updates["maintenance_interval_hours"] = maintenance_interval_hours
    if curator_active is not None:
        config_updates["curator_active"] = curator_active

    return await curator.update_config(project_path=project_path, **config_updates)
