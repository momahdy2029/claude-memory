"""Store memory skill with rich context support."""
import logging
from typing import Dict, Any, Optional, List
from services.database import DatabaseService
from services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


# ============================================================
# INTERNAL HELPER - Auto-infer relationships
# ============================================================

async def _auto_infer_relationships(
    db: DatabaseService,
    embeddings: EmbeddingService,
    memory_id: int,
    content: str,
    memory_type: str,
    outcome: str,
    session_id: str,
    project_path: str = None
) -> List[str]:
    """
    Automatically infer and create relationships based on content analysis.
    Called internally after storing a memory.

    This is NOT a skill - it's an internal helper function.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        memory_id: ID of the newly stored memory
        content: The content of the memory
        memory_type: Type of memory (decision, code, error, etc.)
        outcome: Outcome status (success, partial, failed, pending)
        session_id: Current session ID
        project_path: Optional project path filter

    Returns:
        List of relationship descriptions that were created
    """
    relationships_created = []
    content_lower = content.lower()

    # 1. Fix Detection: If this is a successful decision/code after a recent error
    if outcome == 'success' and memory_type in ['decision', 'code']:
        if session_id:
            try:
                recent_errors = await db.get_memories_by_type(
                    memory_type='error',
                    session_id=session_id,
                    limit=3
                )
                for error in recent_errors:
                    if error['id'] != memory_id:
                        result = await db.create_relationship(
                            memory_id, error['id'], 'fixes', strength=0.9
                        )
                        if result.get('success'):
                            relationships_created.append(f"fixes error #{error['id']}")
            except Exception as e:
                logger.debug(f"Fix detection failed: {e}")

    # 2. Causal Keyword Detection
    causal_keywords = ['because', 'due to', 'caused by', 'result of', 'since']
    if any(kw in content_lower for kw in causal_keywords):
        if embeddings:
            try:
                embedding = await embeddings.generate_embedding(content)
                similar = await db.search_similar(
                    embedding, limit=3, threshold=0.7, project_path=project_path
                )
                for mem in similar:
                    if mem['id'] != memory_id:
                        result = await db.create_relationship(
                            memory_id, mem['id'], 'caused_by', strength=0.7
                        )
                        if result.get('success'):
                            relationships_created.append(f"caused_by #{mem['id']}")
            except Exception as e:
                logger.debug(f"Causal detection failed: {e}")

    # 3. Support Detection
    support_keywords = ['supports', 'evidence for', 'proves', 'confirms', 'validates']
    if any(kw in content_lower for kw in support_keywords):
        if embeddings:
            try:
                embedding = await embeddings.generate_embedding(content)
                similar = await db.search_similar(
                    embedding, limit=2, threshold=0.75, project_path=project_path
                )
                for mem in similar:
                    if mem['id'] != memory_id:
                        result = await db.create_relationship(
                            memory_id, mem['id'], 'supports', strength=0.8
                        )
                        if result.get('success'):
                            relationships_created.append(f"supports #{mem['id']}")
            except Exception as e:
                logger.debug(f"Support detection failed: {e}")

    # 4. Contradiction Detection
    contradiction_keywords = ['but actually', 'wrong', 'incorrect', 'not true', 'instead', 'actually']
    if any(kw in content_lower for kw in contradiction_keywords):
        if embeddings:
            try:
                embedding = await embeddings.generate_embedding(content)
                similar = await db.search_similar(
                    embedding, limit=2, threshold=0.8, project_path=project_path
                )
                for mem in similar:
                    if mem['id'] != memory_id:
                        result = await db.create_relationship(
                            memory_id, mem['id'], 'contradicts', strength=0.85
                        )
                        if result.get('success'):
                            relationships_created.append(f"contradicts #{mem['id']}")
            except Exception as e:
                logger.debug(f"Contradiction detection failed: {e}")

    # 5. Temporal Proximity: Link to recent memories in same session
    if session_id:
        try:
            # Get recent memories from same session (any type)
            recent = await db.get_memories_by_type(
                memory_type=memory_type,  # Same type for relevance
                session_id=session_id,
                limit=3
            )
            for mem in recent:
                if mem['id'] != memory_id:
                    result = await db.create_relationship(
                        memory_id, mem['id'], 'related', strength=0.5
                    )
                    if result.get('success'):
                        relationships_created.append(f"related to #{mem['id']}")
        except Exception as e:
            logger.debug(f"Temporal proximity detection failed: {e}")

    # 6. High Semantic Similarity: Strong related link
    if embeddings:
        try:
            embedding = await embeddings.generate_embedding(content)
            very_similar = await db.search_similar(
                embedding, limit=2, threshold=0.85, project_path=project_path
            )
            for mem in very_similar:
                if mem['id'] != memory_id:
                    strength = mem.get('score', 0.85)
                    result = await db.create_relationship(
                        memory_id, mem['id'], 'related', strength=strength
                    )
                    if result.get('success'):
                        relationships_created.append(f"highly related to #{mem['id']}")
        except Exception as e:
            logger.debug(f"Semantic similarity detection failed: {e}")

    return relationships_created


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
    # Outcome (legacy)
    outcome: Optional[str] = None,
    success: Optional[bool] = None,
    # Classification
    tags: Optional[List[str]] = None,
    importance: int = 5,
    confidence: float = 0.5,
    # Outcome spectrum
    outcome_status: str = 'pending',
    fixed: Optional[List[str]] = None,
    did_not_fix: Optional[List[str]] = None,
    caused: Optional[List[str]] = None
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
        outcome: Description of what happened (legacy field)
        success: Whether the operation succeeded (legacy field)
        tags: Classification tags
        importance: 1-10 scale of importance (default 5)
        confidence: Reliability score 0.0 (unreliable) to 1.0 (proven), default 0.5
        outcome_status: Status of the solution:
            - 'pending': Not yet verified (default)
            - 'success': Fully worked
            - 'partial': Partially worked
            - 'failed': Did not work
            - 'superseded': Replaced by another solution
        fixed: List of what this solution fixed
        did_not_fix: List of what remains unfixed
        caused: List of side effects this solution caused

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
        importance=importance,
        confidence=confidence,
        outcome_status=outcome_status,
        fixed=fixed,
        did_not_fix=did_not_fix,
        caused=caused
    )

    # Auto-infer relationships (silent, internal)
    relationships_created = []
    try:
        relationships_created = await _auto_infer_relationships(
            db=db,
            embeddings=embeddings,
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            outcome=outcome_status,
            session_id=session_id,
            project_path=project_path
        )
        if relationships_created:
            logger.info(f"Auto-created {len(relationships_created)} relationships for memory #{memory_id}")
    except Exception as e:
        logger.warning(f"Failed to auto-infer relationships: {e}")
        # Don't fail the store operation if relationship inference fails

    return {
        "success": True,
        "memory_id": memory_id,
        "type": memory_type,
        "importance": importance,
        "confidence": confidence,
        "outcome_status": outcome_status,
        "project": project_path,
        "relationships_created": relationships_created,
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
