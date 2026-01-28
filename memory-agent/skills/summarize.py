"""Session summarization skill with project context."""
from typing import Dict, Any, Optional, List
from datetime import datetime
from services.database import DatabaseService
from services.embeddings import EmbeddingService


async def summarize_session(
    db: DatabaseService,
    embeddings: EmbeddingService,
    session_id: str,
    summary: str,
    key_decisions: Optional[List[str]] = None,
    code_patterns: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    project_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Store a session summary with optional key decisions and code patterns.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        session_id: The session identifier
        summary: Summary of the session
        key_decisions: List of key decisions made during session
        code_patterns: List of important code patterns discovered
        metadata: Additional metadata
        project_path: Project this session worked on

    Returns:
        Dict with stored summary information
    """
    stored_ids = []

    # Store the main session summary
    summary_embedding = await embeddings.generate_embedding(summary)
    summary_meta = {
        **(metadata or {}),
        "summarized_at": datetime.now().isoformat()
    }
    summary_id = await db.store_memory(
        memory_type="session",
        content=summary,
        embedding=summary_embedding,
        metadata=summary_meta,
        session_id=session_id,
        project_path=project_path,
        importance=8  # Session summaries are high importance
    )
    stored_ids.append({"type": "session", "id": summary_id})

    # Store key decisions
    if key_decisions:
        for decision in key_decisions:
            decision_embedding = await embeddings.generate_embedding(decision)
            decision_id = await db.store_memory(
                memory_type="decision",
                content=decision,
                embedding=decision_embedding,
                metadata={"session_summary_id": summary_id},
                session_id=session_id,
                project_path=project_path,
                importance=7  # Decisions are important
            )
            stored_ids.append({"type": "decision", "id": decision_id})

    # Store code patterns
    if code_patterns:
        for pattern in code_patterns:
            pattern_embedding = await embeddings.generate_embedding(pattern)
            pattern_id = await db.store_memory(
                memory_type="code",
                content=pattern,
                embedding=pattern_embedding,
                metadata={"session_summary_id": summary_id},
                session_id=session_id,
                project_path=project_path,
                importance=6  # Code patterns are useful
            )
            stored_ids.append({"type": "code", "id": pattern_id})

    return {
        "success": True,
        "session_id": session_id,
        "project_path": project_path,
        "stored_items": stored_ids,
        "total_items": len(stored_ids),
        "message": f"Session {session_id} summarized with {len(stored_ids)} items"
    }
