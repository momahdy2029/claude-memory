"""Retrieve memory skill with project filtering."""
from typing import Dict, Any, Optional, List
from services.database import DatabaseService


async def retrieve_memory(
    db: DatabaseService,
    memory_id: Optional[int] = None,
    memory_type: Optional[str] = None,
    session_id: Optional[str] = None,
    project_path: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Retrieve memories by ID or filter criteria.

    Args:
        db: Database service instance
        memory_id: Specific memory ID to retrieve
        memory_type: Filter by memory type
        session_id: Filter by session ID
        project_path: Filter by project path
        limit: Maximum number of memories to return

    Returns:
        Dict with retrieved memories
    """
    if memory_id:
        memory = await db.get_memory(memory_id)
        if memory:
            return {
                "success": True,
                "memories": [memory],
                "count": 1
            }
        return {
            "success": False,
            "message": f"Memory with ID {memory_id} not found",
            "memories": [],
            "count": 0
        }

    if memory_type:
        memories = await db.get_memories_by_type(
            memory_type=memory_type,
            limit=limit,
            session_id=session_id,
            project_path=project_path
        )
        return {
            "success": True,
            "memories": memories,
            "count": len(memories),
            "filters": {
                "type": memory_type,
                "project": project_path,
                "session": session_id
            }
        }

    # Return stats if no specific criteria
    stats = await db.get_stats()
    return {
        "success": True,
        "stats": stats,
        "message": "Provide memory_id or memory_type to retrieve specific memories"
    }
