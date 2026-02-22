"""Store memory skill with rich context support."""
import logging
from typing import Dict, Any, Optional, List
from services.database import DatabaseService
from services.embeddings import EmbeddingService, EmbeddingError

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

    # Generate embedding once for all similarity-based detections
    cached_embedding = None
    if embeddings:
        try:
            cached_embedding = await embeddings.generate_embedding(content)
        except Exception as e:
            logger.debug(f"Embedding generation for relationship inference failed: {e}")

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
        if cached_embedding is not None:
            try:
                similar = await db.search_similar(
                    cached_embedding, limit=3, threshold=0.7, project_path=project_path
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
        if cached_embedding is not None:
            try:
                similar = await db.search_similar(
                    cached_embedding, limit=2, threshold=0.75, project_path=project_path
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
        if cached_embedding is not None:
            try:
                similar = await db.search_similar(
                    cached_embedding, limit=2, threshold=0.8, project_path=project_path
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
            recent = await db.get_memories_by_type(
                memory_type=memory_type,
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
    if cached_embedding is not None:
        try:
            very_similar = await db.search_similar(
                cached_embedding, limit=2, threshold=0.85, project_path=project_path
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
    # Consolidate legacy outcome/success into outcome_status
    # If caller uses legacy params, derive outcome_status from them
    if outcome_status == 'pending':
        if success is True:
            outcome_status = 'success'
        elif success is False:
            outcome_status = 'failed'
        elif outcome and isinstance(outcome, str):
            # Map common outcome text values
            outcome_lower = outcome.lower().strip()
            if outcome_lower in ('success', 'worked', 'fixed', 'resolved'):
                outcome_status = 'success'
            elif outcome_lower in ('failed', 'broken', 'error'):
                outcome_status = 'failed'
            elif outcome_lower in ('partial', 'partially'):
                outcome_status = 'partial'

    # Generate embedding for the content with status tracking
    embed_result = await embeddings.generate_embedding_with_status(content)
    embedding = embed_result.embedding

    if not embed_result.ok:
        logger.warning(
            f"Embedding failed ({embed_result.error.value}): {embed_result.error_message}. "
            f"Memory will be stored without embedding (not semantically searchable)."
        )

    # === Dedup check: find near-duplicates before storing ===
    link_to_id = None  # Set if we find a near-duplicate to link after storing
    if embedding:
        try:
            duplicates = await db.find_similar_for_dedup(
                embedding=embedding,
                project_path=project_path,
                threshold=0.92,
                limit=3
            )
            if duplicates:
                best_match = duplicates[0]
                if best_match['similarity'] >= 0.95:
                    # Very high similarity - merge into existing memory
                    # Keeps longer content, higher importance/confidence
                    updated_id = await db.merge_memory(
                        existing_id=best_match['id'],
                        new_content=content,
                        new_importance=importance,
                        new_confidence=confidence
                    )
                    logger.info(
                        f"Dedup: merged with memory #{best_match['id']} "
                        f"(similarity: {best_match['similarity']:.3f})"
                    )
                    return {
                        "success": True,
                        "memory_id": updated_id,
                        "action": "merged",
                        "merged_with": best_match['id'],
                        "similarity": best_match['similarity'],
                        "type": memory_type,
                        "importance": importance,
                        "confidence": confidence,
                        "outcome_status": outcome_status,
                        "project": project_path,
                        "relationships_created": [],
                        "message": (
                            f"Merged with existing memory #{best_match['id']} "
                            f"(similarity: {best_match['similarity']:.2f})"
                        )
                    }
                elif best_match['similarity'] >= 0.92:
                    # High similarity but not identical - store new but link as related
                    # We'll create the relationship after storing below
                    link_to_id = best_match['id']
                    link_similarity = best_match['similarity']
                    logger.info(
                        f"Dedup: will link to memory #{link_to_id} "
                        f"(similarity: {link_similarity:.3f})"
                    )
        except Exception as e:
            # Dedup is best-effort; never block a store operation
            logger.warning(f"Dedup check failed (non-fatal): {e}")
            link_to_id = None
    # === End dedup check ===

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

    # If dedup found a near-duplicate (0.92-0.95 range), link as related
    dedup_linked = False
    if link_to_id is not None:
        try:
            result = await db.create_relationship(
                memory_id, link_to_id, 'related', strength=0.9
            )
            if result.get('success'):
                relationships_created.append(f"near-duplicate of #{link_to_id}")
                dedup_linked = True
                logger.info(f"Dedup: linked memory #{memory_id} to near-duplicate #{link_to_id}")
        except Exception as e:
            logger.warning(f"Failed to create dedup relationship: {e}")

    response = {
        "success": True,
        "memory_id": memory_id,
        "type": memory_type,
        "importance": importance,
        "confidence": confidence,
        "outcome_status": outcome_status,
        "project": project_path,
        "relationships_created": relationships_created,
        "has_embedding": embedding is not None,
        "message": f"Memory stored successfully with ID {memory_id}"
    }
    if not embed_result.ok:
        response["embedding_error"] = embed_result.error.value
        response["embedding_error_detail"] = embed_result.error_message
    if dedup_linked:
        response["action"] = "stored_and_linked"
        response["linked_to"] = link_to_id

    return response


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
