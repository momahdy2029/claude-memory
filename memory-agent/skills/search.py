"""Semantic search skill with context filtering."""
from typing import Dict, Any, Optional, List
from services.database import DatabaseService
from services.embeddings import EmbeddingService


async def semantic_search(
    db: DatabaseService,
    embeddings: EmbeddingService,
    query: str,
    limit: int = 10,
    memory_type: Optional[str] = None,
    session_id: Optional[str] = None,
    project_path: Optional[str] = None,
    agent_type: Optional[str] = None,
    success_only: bool = False,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Search memories using semantic similarity with context filters.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        query: Search query text
        limit: Maximum number of results
        memory_type: Filter by type (session, decision, code, chunk, error)
        session_id: Filter by session ID
        project_path: Filter by project
        agent_type: Filter by agent that created the memory
        success_only: Only return memories marked as successful
        threshold: Minimum similarity threshold (0-1)

    Returns:
        Dict with search results ranked by similarity * importance
    """
    # Generate embedding for the query
    query_embedding = await embeddings.generate_embedding(query)

    # Search for similar memories
    results = await db.search_similar(
        embedding=query_embedding,
        limit=limit,
        memory_type=memory_type,
        session_id=session_id,
        project_path=project_path,
        agent_type=agent_type,
        success_only=success_only,
        threshold=threshold
    )

    return {
        "success": True,
        "query": query,
        "results": results,
        "count": len(results),
        "filters": {
            "type": memory_type,
            "project": project_path,
            "agent": agent_type,
            "success_only": success_only
        },
        "threshold": threshold
    }


async def search_patterns(
    db: DatabaseService,
    embeddings: EmbeddingService,
    query: str,
    limit: int = 5,
    problem_type: Optional[str] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Search for reusable solution patterns.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        query: Problem description or search query
        limit: Maximum number of results
        problem_type: Filter by problem type
        threshold: Minimum similarity threshold

    Returns:
        Dict with patterns ranked by similarity * success_rate
    """
    query_embedding = await embeddings.generate_embedding(query)

    results = await db.search_patterns(
        embedding=query_embedding,
        limit=limit,
        problem_type=problem_type,
        threshold=threshold
    )

    return {
        "success": True,
        "query": query,
        "patterns": results,
        "count": len(results),
        "problem_type": problem_type
    }


async def get_project_context(
    db: DatabaseService,
    embeddings: EmbeddingService,
    project_path: str,
    query: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get all relevant context for a project.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        project_path: Path to the project
        query: Optional query to filter relevant memories
        limit: Max memories to return

    Returns:
        Dict with project info and relevant memories
    """
    # Get project info
    project = await db.get_project(project_path)

    # Get recent decisions for this project
    decisions = await db.get_memories_by_type(
        memory_type="decision",
        project_path=project_path,
        limit=limit
    )

    # Get patterns used in this project
    patterns = await db.get_memories_by_type(
        memory_type="code",
        project_path=project_path,
        limit=limit
    )

    # If query provided, also do semantic search
    relevant = []
    if query:
        query_embedding = await embeddings.generate_embedding(query)
        relevant = await db.search_similar(
            embedding=query_embedding,
            project_path=project_path,
            limit=limit,
            threshold=0.4
        )

    return {
        "success": True,
        "project": project,
        "decisions": decisions,
        "code_patterns": patterns,
        "relevant_to_query": relevant if query else None
    }
