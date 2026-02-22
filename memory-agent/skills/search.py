"""Semantic search skill with context filtering and fallback support."""
import logging
from typing import Dict, Any, Optional, List
from services.database import DatabaseService
from services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


async def _enrich_with_graph_context(db: DatabaseService, results: list, max_related_items: int = 2) -> list:
    """
    Enrich search results with relationship context using batch queries.

    Uses batch DB queries (IN clauses) instead of per-result sequential queries,
    providing 5-10x speedup on the enrichment phase for typical 10-result sets.

    Args:
        db: Database service instance
        results: List of search results to enrich
        max_related_items: Max related items per relationship type (caps response size)

    Returns:
        List of enriched results with graph context added
    """
    import asyncio

    if not results:
        return results

    # Collect all memory IDs that need enrichment
    all_ids = [r.get('id') for r in results if r.get('id')]
    if not all_ids:
        return results

    # Categorize IDs by type for targeted batch queries
    error_ids = [r['id'] for r in results if r.get('id') and r.get('type') == 'error']
    decision_ids = [r['id'] for r in results if r.get('id') and r.get('type') == 'decision']
    causal_ids = [r['id'] for r in results if r.get('id') and r.get('type') in ('error', 'decision', 'code')]

    # Run all batch queries in parallel
    batch_tasks = {}

    # Batch: fixes for errors (incoming 'fixes' relationships)
    if error_ids:
        batch_tasks['fixes'] = db.get_related_memories_batch(error_ids, 'fixes', direction='incoming')

    # Batch: supports for decisions (incoming 'supports' relationships)
    if decision_ids:
        batch_tasks['supports'] = db.get_related_memories_batch(decision_ids, 'supports', direction='incoming')
        batch_tasks['caused'] = db.get_related_memories_batch(decision_ids, 'caused_by', direction='outgoing')

    # Batch: contradictions for all results
    batch_tasks['contradictions'] = db.find_contradictions_batch(all_ids)

    # Causal chains still need per-ID traversal (recursive), but run in parallel
    causal_futures = {}
    for mid in causal_ids:
        causal_futures[mid] = db.get_causal_chain(mid, max_depth=3)

    # Gather all batch results
    batch_results = {}
    try:
        if batch_tasks:
            keys = list(batch_tasks.keys())
            values = await asyncio.gather(*batch_tasks.values(), return_exceptions=True)
            for k, v in zip(keys, values):
                batch_results[k] = v if not isinstance(v, Exception) else {}
    except Exception as e:
        logger.warning(f"Batch graph enrichment failed: {e}")
        batch_results = {}

    # Gather causal chain results in parallel
    causal_results = {}
    if causal_futures:
        try:
            keys = list(causal_futures.keys())
            values = await asyncio.gather(*causal_futures.values(), return_exceptions=True)
            for k, v in zip(keys, values):
                causal_results[k] = v if not isinstance(v, Exception) else None
        except Exception as e:
            logger.debug(f"Causal chain batch failed: {e}")

    # Assemble enriched results
    enriched = []
    fixes_map = batch_results.get('fixes', {})
    supports_map = batch_results.get('supports', {})
    caused_map = batch_results.get('caused', {})
    contradictions_map = batch_results.get('contradictions', {})

    for result in results:
        memory_id = result.get('id')
        if not memory_id:
            enriched.append(result)
            continue

        enriched_result = dict(result)
        memory_type = result.get('type', '')

        # Errors: known fixes
        if memory_type == 'error' and memory_id in fixes_map:
            fixes = fixes_map[memory_id]
            if fixes:
                enriched_result['known_fixes'] = [
                    {'id': f['id'], 'content': f['content'][:200], 'outcome': f.get('outcome')}
                    for f in fixes[:max_related_items]
                ]

        # Decisions: rationale and consequences
        if memory_type == 'decision':
            supports = supports_map.get(memory_id, [])
            if supports:
                enriched_result['rationale'] = [
                    {'id': s['id'], 'content': s['content'][:200]}
                    for s in supports[:max_related_items]
                ]
            caused = caused_map.get(memory_id, [])
            if caused:
                enriched_result['consequences'] = [
                    {'id': c['id'], 'content': c['content'][:200]}
                    for c in caused[:max_related_items]
                ]

        # All types: contradictions
        contradictions = contradictions_map.get(memory_id, [])
        if contradictions:
            enriched_result['contradictions'] = [
                {'id': c['id'], 'content': c['content'][:200]}
                for c in contradictions[:max_related_items]
            ]

        # Causal chains
        if memory_type in ('error', 'decision', 'code') and memory_id in causal_results:
            chain = causal_results[memory_id]
            if chain and (chain.get('causes') or chain.get('fixes') or chain.get('root_causes')):
                enriched_result['causal_chain'] = chain

        enriched.append(enriched_result)

    return enriched


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
    threshold: float = 0.5,
    # Outcome spectrum filters
    include_failed: bool = False,
    include_superseded: bool = False,
    include_unreliable: bool = False,
    outcome_status: Optional[str] = None,
    # Context-aware search
    current_context: Optional[Dict[str, Any]] = None,
    auto_detect_context: bool = True,
    # Graph enrichment
    include_graph: bool = True,
    # Adaptive ranking
    temperature: Optional[float] = None
) -> Dict[str, Any]:
    """
    Search memories using semantic similarity with context filters.

    Includes automatic fallback to keyword search when Ollama is unavailable.

    Outcome-aware search behavior:
    - 'success' memories rank highest (1.5x boost)
    - 'partial' memories shown with warning (1.0x - no penalty)
    - 'failed' memories excluded by default (use include_failed=True to show)
    - 'superseded' memories excluded and replaced with their superseding memory
    - 'pending' memories shown normally (1.0x)
    - Unreliable memories (failure_count >= 3) excluded by default (use include_unreliable=True)

    Context-aware search:
    - If current_context provided, memories that worked in similar contexts get +0.2 boost
    - Memories that failed in similar contexts get -0.2 penalty
    - If auto_detect_context=True and project_path provided, context is auto-detected

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        query: Search query text
        limit: Maximum number of results
        memory_type: Filter by type (session, decision, code, chunk, error)
        session_id: Filter by session ID
        project_path: Filter by project
        agent_type: Filter by agent that created the memory
        success_only: Only return memories marked as successful (legacy)
        threshold: Minimum similarity threshold (0-1)
        include_failed: Include memories with outcome_status='failed' (default False)
        include_superseded: Include memories with outcome_status='superseded' (default False)
        include_unreliable: Include memories with failure_count >= 3 (default False)
        outcome_status: Filter by specific outcome status
        current_context: Context dict with project_type, tech_stack, file_patterns
        auto_detect_context: If True and project_path provided, auto-detect context
        include_graph: Enrich results with graph context (fixes, rationale, contradictions)

    Returns:
        Dict with search results ranked by: (similarity * 0.7) + (confidence * 0.3) + context_adjustment
        Each result includes outcome_status, outcome_warning, outcome_boost, context_adjustment,
        and context_recommendation fields.
    """
    # Auto-detect context from project_path if enabled
    detected_context = None
    if auto_detect_context and project_path and not current_context:
        try:
            from skills.context import detect_project_context
            detected_context = detect_project_context(project_path)
        except Exception:
            pass

    # Use provided context or detected context
    search_context = current_context or detected_context

    # Generate embedding for the query (may return None if Ollama unavailable)
    query_embedding = await embeddings.generate_embedding(query)

    # Determine search method based on embedding availability
    search_method = "semantic"
    results = []

    if query_embedding is not None:
        # Use semantic search with embeddings
        results = await db.search_similar(
            embedding=query_embedding,
            limit=limit,
            memory_type=memory_type,
            session_id=session_id,
            project_path=project_path,
            agent_type=agent_type,
            success_only=success_only,
            threshold=threshold,
            include_failed=include_failed,
            include_superseded=include_superseded,
            include_unreliable=include_unreliable,
            outcome_status=outcome_status,
            current_context=search_context,
            query_text=query,
            temperature=temperature
        )
    else:
        # Fallback to keyword search
        search_method = "keyword"
        results = await db.keyword_search(
            query=query,
            limit=limit,
            memory_type=memory_type,
            session_id=session_id,
            project_path=project_path,
            agent_type=agent_type,
            success_only=success_only,
            include_failed=include_failed,
            include_superseded=include_superseded,
            include_unreliable=include_unreliable,
            outcome_status=outcome_status
        )

    # Enrich with graph context if requested
    if include_graph:
        try:
            results = await _enrich_with_graph_context(db, results)
        except Exception as e:
            logger.warning(f"Failed to enrich with graph context: {e}")
            # Continue with unenriched results

    # Promote accessed memories if their tier should be higher (fire-and-forget)
    if results:
        try:
            from services.tier_manager import TierManager
            tier_mgr = TierManager(db)
            for r in results:
                mem_id = r.get('id')
                if mem_id:
                    await tier_mgr.promote_on_access(mem_id)
        except Exception as e:
            logger.debug(f"Tier promotion on access failed (non-fatal): {e}")

    return {
        "success": True,
        "query": query,
        "results": results,
        "count": len(results),
        "search_method": search_method,
        "degraded_mode": embeddings.is_degraded(),
        "filters": {
            "type": memory_type,
            "project": project_path,
            "agent": agent_type,
            "success_only": success_only,
            "include_failed": include_failed,
            "include_superseded": include_superseded,
            "include_unreliable": include_unreliable,
            "outcome_status": outcome_status,
            "include_graph": include_graph
        },
        "context_aware": search_context is not None,
        "detected_context": detected_context,
        "threshold": threshold if search_method == "semantic" else None,
        "temperature": temperature
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

    Includes fallback to keyword search when Ollama is unavailable.

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

    search_method = "semantic"
    results = []

    if query_embedding is not None:
        results = await db.search_patterns(
            embedding=query_embedding,
            limit=limit,
            problem_type=problem_type,
            threshold=threshold
        )
    else:
        # Fallback: keyword search on patterns table
        search_method = "keyword"
        results = await db.keyword_search_patterns(
            query=query,
            limit=limit,
            problem_type=problem_type
        )

    return {
        "success": True,
        "query": query,
        "patterns": results,
        "count": len(results),
        "search_method": search_method,
        "degraded_mode": embeddings.is_degraded(),
        "problem_type": problem_type
    }


async def get_project_context(
    db: DatabaseService,
    embeddings: EmbeddingService,
    project_path: str,
    query: Optional[str] = None,
    limit: int = 5
) -> Dict[str, Any]:
    """
    Get all relevant context for a project.

    Includes fallback to keyword search when Ollama is unavailable.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        project_path: Path to the project
        query: Optional query to filter relevant memories
        limit: Max memories to return (default reduced to 5 for response size)

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

    # If query provided, search for relevant memories
    relevant = []
    search_method = None
    if query:
        query_embedding = await embeddings.generate_embedding(query)
        if query_embedding is not None:
            search_method = "semantic"
            relevant = await db.search_similar(
                embedding=query_embedding,
                project_path=project_path,
                limit=limit,
                threshold=0.4
            )
        else:
            # Fallback to keyword search
            search_method = "keyword"
            relevant = await db.keyword_search(
                query=query,
                project_path=project_path,
                limit=limit
            )

    # Enrich all result lists with graph context (relationships, contradictions, causal chains)
    try:
        if decisions:
            decisions = await _enrich_with_graph_context(db, decisions)
        if patterns:
            patterns = await _enrich_with_graph_context(db, patterns)
        if relevant:
            relevant = await _enrich_with_graph_context(db, relevant)
    except Exception as e:
        logger.warning(f"Failed to enrich project context with graph: {e}")

    return {
        "success": True,
        "project": project,
        "decisions": decisions,
        "code_patterns": patterns,
        "relevant_to_query": relevant if query else None,
        "search_method": search_method,
        "degraded_mode": embeddings.is_degraded()
    }
