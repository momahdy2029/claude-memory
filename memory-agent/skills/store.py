"""Store memory skill with rich context support."""
from typing import Dict, Any, Optional, List
from services.database import DatabaseService
from services.embeddings import EmbeddingService


async def store_memory(
    db: DatabaseService,
    embeddings: EmbeddingService,
    content: str,
    memory_type: str = "chunk",
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    # Project context
    project_path: Optional[str] = None,
    project_name: Optional[str] = None,
    project_type: Optional[str] = None,
    tech_stack: Optional[List[str]] = None,
    # Session context
    chat_id: Optional[str] = None,
    # Agent context
    agent_type: Optional[str] = None,
    skill_used: Optional[str] = None,
    tools_used: Optional[List[str]] = None,
    # Outcome
    outcome: Optional[str] = None,
    success: Optional[bool] = None,
    # Classification
    tags: Optional[List[str]] = None,
    importance: int = 5
) -> Dict[str, Any]:
    """
    Store a memory with semantic embedding and rich context.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        content: The text content to store
        memory_type: Type of memory:
            - 'session': Session summaries
            - 'decision': Architectural/design decisions
            - 'code': Code patterns and snippets
            - 'chunk': General conversation chunks
            - 'error': Error patterns and solutions
            - 'preference': User preferences
        metadata: Optional additional metadata
        session_id: Session identifier
        project_path: Full path to the project
        project_name: Human-readable project name
        project_type: Type (wordpress, react, python, etc.)
        tech_stack: List of technologies used
        chat_id: Specific chat/conversation ID
        agent_type: Agent that processed this (Explore, Plan, etc.)
        skill_used: Skill that was invoked
        tools_used: List of tools that were called
        outcome: Description of what happened
        success: Whether the operation succeeded
        tags: Classification tags
        importance: 1-10 scale of importance (default 5)

    Returns:
        Dict with stored memory ID and status
    """
    # Generate embedding for the content
    embedding = await embeddings.generate_embedding(content)

    # Store in database with full context
    memory_id = await db.store_memory(
        memory_type=memory_type,
        content=content,
        embedding=embedding,
        metadata=metadata,
        session_id=session_id,
        project_path=project_path,
        project_name=project_name,
        project_type=project_type,
        tech_stack=tech_stack,
        chat_id=chat_id,
        agent_type=agent_type,
        skill_used=skill_used,
        tools_used=tools_used,
        outcome=outcome,
        success=success,
        tags=tags,
        importance=importance
    )

    return {
        "success": True,
        "memory_id": memory_id,
        "type": memory_type,
        "importance": importance,
        "project": project_path,
        "message": f"Memory stored successfully with ID {memory_id}"
    }


async def store_project(
    db: DatabaseService,
    path: str,
    name: Optional[str] = None,
    project_type: Optional[str] = None,
    tech_stack: Optional[List[str]] = None,
    conventions: Optional[Dict[str, Any]] = None,
    preferences: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Store or update project-level information.

    Args:
        db: Database service instance
        path: Full path to the project
        name: Human-readable project name
        project_type: Type (wordpress, react, python, etc.)
        tech_stack: List of technologies
        conventions: Coding conventions (naming, structure, etc.)
        preferences: User preferences for this project

    Returns:
        Dict with project info
    """
    project_id = await db.store_project(
        path=path,
        name=name,
        project_type=project_type,
        tech_stack=tech_stack,
        conventions=conventions,
        preferences=preferences
    )

    return {
        "success": True,
        "project_id": project_id,
        "path": path,
        "message": f"Project info stored/updated for {path}"
    }


async def store_pattern(
    db: DatabaseService,
    embeddings: EmbeddingService,
    name: str,
    solution: str,
    problem_type: Optional[str] = None,
    tech_context: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Store a reusable solution pattern.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        name: Pattern name
        solution: The solution/approach
        problem_type: Category (bug_fix, feature, refactor, config, etc.)
        tech_context: Technologies this applies to
        metadata: Additional info

    Returns:
        Dict with pattern info
    """
    embedding = await embeddings.generate_embedding(f"{name}: {solution}")

    pattern_id = await db.store_pattern(
        name=name,
        solution=solution,
        embedding=embedding,
        problem_type=problem_type,
        tech_context=tech_context,
        metadata=metadata
    )

    return {
        "success": True,
        "pattern_id": pattern_id,
        "name": name,
        "message": f"Pattern '{name}' stored successfully"
    }
