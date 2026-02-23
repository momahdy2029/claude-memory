"""MCP stdio server for Claude Memory.

Thin adapter that exposes the memory system's core skills as MCP tools,
allowing any MCP-compatible client (OpenClaw, Claude Code, etc.) to
store, search, and manage memories over stdio JSON-RPC.

Usage:
    python mcp_server.py              # stdio mode (default)
    mcp dev mcp_server.py             # interactive inspector

Shares the same SQLite database as the HTTP server (main.py) via WAL mode.
"""

# ── CRITICAL: Suppress stdout noise before ANY library imports ──────────
# stdout is reserved exclusively for MCP JSON-RPC protocol messages.
# Any stray print/progress-bar on stdout will corrupt the protocol.

import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import logging

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("mcp-claude-memory")

# Add memory-agent/ to sys.path so local imports (services.*, skills.*, config) resolve
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

# ── Imports ─────────────────────────────────────────────────────────────

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# MCP SDK - support both v1 (FastMCP) and v2 (MCPServer) import paths
try:
    from mcp.server.fastmcp import FastMCP, Context
except ImportError:
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP, Context
    except ImportError:
        raise ImportError(
            "MCP SDK not found. Install with: pip install 'mcp>=1.0.0'"
        )

from services.database import DatabaseService
from services.embeddings import EmbeddingService
from config import config

# Direct skill imports - no HTTP, no FastAPI dependency
from skills.store import store_memory, store_project, store_pattern
from skills.search import semantic_search, search_patterns, get_project_context
from skills.timeline import timeline_log


# ── Lifespan: DB + Embeddings initialization ───────────────────────────

@dataclass
class AppContext:
    db: DatabaseService
    embeddings: EmbeddingService


@asynccontextmanager
async def app_lifespan(server: Any) -> AsyncIterator[AppContext]:
    """Initialize database and embedding services on startup, clean up on shutdown."""
    logger.info("Initializing Claude Memory MCP server...")

    db = DatabaseService()
    await db.connect()
    await db.initialize_schema()
    logger.info(f"Database connected: {db.db_path}")

    logger.info(
        f"Loading embedding model: {config.EMBEDDING_MODEL} "
        f"(provider: {config.EMBEDDING_PROVIDER}) - this may take a moment..."
    )
    embeddings = EmbeddingService(
        provider_type=config.EMBEDDING_PROVIDER,
        model=config.EMBEDDING_MODEL,
    )
    logger.info(f"Embeddings ready (dim={embeddings.get_dimension()})")

    try:
        yield AppContext(db=db, embeddings=embeddings)
    finally:
        if db.conn:
            db.conn.close()
        logger.info("Claude Memory MCP server shut down.")


# ── MCP Server ──────────────────────────────────────────────────────────

mcp_server = FastMCP("claude-memory", lifespan=app_lifespan)


def _get_app(ctx: Context) -> AppContext:
    """Extract AppContext from the MCP request context."""
    return ctx.request_context.lifespan_context


# ── Tools ───────────────────────────────────────────────────────────────

@mcp_server.tool()
async def memory_store(
    ctx: Context,
    content: str,
    memory_type: str = "chunk",
    tags: Optional[List[str]] = None,
    importance: int = 5,
    outcome: Optional[str] = None,
    success: Optional[bool] = None,
    project_path: Optional[str] = None,
    project_name: Optional[str] = None,
    project_type: Optional[str] = None,
    tech_stack: Optional[List[str]] = None,
    agent_type: Optional[str] = None,
) -> str:
    """Store a memory with semantic embedding.

    Args:
        content: Content to remember
        memory_type: Type: session, decision, code, chunk, error, preference
        tags: Classification tags
        importance: 1-10 importance scale (default 5)
        outcome: What happened
        success: Did it work?
        project_path: Project path
        project_name: Project name
        project_type: Project type (wordpress, react, etc.)
        tech_stack: Technologies used
        agent_type: Agent used (Explore, Plan, etc.)
    """
    app = _get_app(ctx)
    result = await store_memory(
        db=app.db,
        embeddings=app.embeddings,
        content=content,
        memory_type=memory_type,
        tags=tags,
        importance=importance,
        outcome=outcome,
        success=success,
        project_path=project_path,
        project_name=project_name,
        project_type=project_type,
        tech_stack=tech_stack,
        agent_type=agent_type,
    )

    # Auto-create a timeline event for every stored memory
    try:
        event_type_map = {
            "decision": "decision",
            "error": "error",
            "code": "action",
            "session": "checkpoint",
            "preference": "observation",
            "chunk": "observation",
        }
        await timeline_log(
            db=app.db,
            embeddings=app.embeddings,
            session_id=str(uuid.uuid4()),
            event_type=event_type_map.get(memory_type, "observation"),
            summary=content[:200],
            details=content if len(content) > 200 else None,
            project_path=project_path,
        )
    except Exception as e:
        logger.debug(f"Timeline piggyback failed (non-fatal): {e}")

    return json.dumps(result, default=str)


@mcp_server.tool()
async def memory_search(
    ctx: Context,
    query: str,
    limit: int = 10,
    memory_type: Optional[str] = None,
    project_path: Optional[str] = None,
    success_only: bool = False,
    threshold: float = 0.5,
) -> str:
    """Search memories using natural language. Returns similar content ranked by relevance.

    Args:
        query: Search query
        limit: Max results (default 10)
        memory_type: Filter: session, decision, code, chunk, error, preference
        project_path: Filter by project
        success_only: Only return successful memories
        threshold: Minimum similarity 0-1 (default 0.5)
    """
    app = _get_app(ctx)
    result = await semantic_search(
        db=app.db,
        embeddings=app.embeddings,
        query=query,
        limit=limit,
        memory_type=memory_type,
        project_path=project_path,
        success_only=success_only,
        threshold=threshold,
    )
    return json.dumps(result, default=str)


@mcp_server.tool()
async def memory_search_patterns(
    ctx: Context,
    query: str,
    limit: int = 5,
    problem_type: Optional[str] = None,
    threshold: float = 0.5,
) -> str:
    """Search for reusable solution patterns, ranked by similarity and success rate.

    Args:
        query: Problem description
        limit: Max results (default 5)
        problem_type: Filter: bug_fix, feature, refactor, config, performance
        threshold: Minimum similarity 0-1 (default 0.5)
    """
    app = _get_app(ctx)
    result = await search_patterns(
        db=app.db,
        embeddings=app.embeddings,
        query=query,
        limit=limit,
        problem_type=problem_type,
        threshold=threshold,
    )
    return json.dumps(result, default=str)


@mcp_server.tool()
async def memory_store_pattern(
    ctx: Context,
    name: str,
    solution: str,
    problem_type: Optional[str] = None,
    tech_context: Optional[List[str]] = None,
) -> str:
    """Store a reusable solution pattern for future reference.

    Args:
        name: Pattern name
        solution: The solution
        problem_type: Type: bug_fix, feature, refactor, config, performance
        tech_context: Technologies this applies to
    """
    app = _get_app(ctx)
    result = await store_pattern(
        db=app.db,
        embeddings=app.embeddings,
        name=name,
        solution=solution,
        problem_type=problem_type,
        tech_context=tech_context,
    )
    return json.dumps(result, default=str)


@mcp_server.tool()
async def memory_store_project(
    ctx: Context,
    path: str,
    name: Optional[str] = None,
    project_type: Optional[str] = None,
    tech_stack: Optional[List[str]] = None,
    conventions: Optional[Dict[str, Any]] = None,
    preferences: Optional[Dict[str, Any]] = None,
) -> str:
    """Store project info (tech stack, conventions, preferences).

    Args:
        path: Project path
        name: Project name
        project_type: Project type
        tech_stack: Technologies used
        conventions: Coding conventions dict
        preferences: User preferences dict
    """
    app = _get_app(ctx)
    result = await store_project(
        db=app.db,
        path=path,
        name=name,
        project_type=project_type,
        tech_stack=tech_stack,
        conventions=conventions,
        preferences=preferences,
    )
    return json.dumps(result, default=str)


@mcp_server.tool()
async def memory_get_project(
    ctx: Context,
    project_path: str,
) -> str:
    """Get stored info about a project.

    Args:
        project_path: Project path to look up
    """
    app = _get_app(ctx)
    result = await app.db.get_project(project_path)
    if result is None:
        return json.dumps({"found": False, "project_path": project_path})
    return json.dumps(result, default=str)


@mcp_server.tool()
async def memory_context(
    ctx: Context,
    project_path: Optional[str] = None,
    query: Optional[str] = None,
    include_decisions: bool = True,
    include_errors: bool = True,
    include_patterns: bool = True,
) -> str:
    """Load relevant memories for the current session. Call at session start to get
    project info, recent decisions, patterns, and relevant past errors/solutions.

    Args:
        project_path: Project path (optional, filters results)
        query: Optional semantic query to find relevant memories
        include_decisions: Include recent decisions (default true)
        include_errors: Include recent errors (default true)
        include_patterns: Include solution patterns (default true)
    """
    app = _get_app(ctx)
    result: Dict[str, Any] = {}

    # Project context (includes decisions, code patterns, and query-relevant memories)
    if project_path:
        project_ctx = await get_project_context(
            db=app.db,
            embeddings=app.embeddings,
            project_path=project_path,
            query=query,
        )
        result["project"] = project_ctx

    # Solution patterns (when a query is provided)
    if include_patterns and query:
        patterns = await search_patterns(
            db=app.db,
            embeddings=app.embeddings,
            query=query,
            limit=5,
        )
        result["patterns"] = patterns

    # Recent errors (when requested and query provided)
    if include_errors and query:
        errors = await semantic_search(
            db=app.db,
            embeddings=app.embeddings,
            query=query,
            limit=5,
            memory_type="error",
            project_path=project_path,
        )
        result["recent_errors"] = errors

    # Stats
    try:
        stats = await app.db.get_stats()
        result["stats"] = stats
    except Exception as e:
        result["stats_error"] = str(e)

    result["success"] = True
    return json.dumps(result, default=str)


@mcp_server.tool()
async def memory_timeline_log(
    ctx: Context,
    summary: str,
    event_type: str = "observation",
    details: Optional[str] = None,
    project_path: Optional[str] = None,
    session_id: Optional[str] = None,
    status: str = "completed",
    outcome: Optional[str] = None,
    is_anchor: bool = False,
) -> str:
    """Log an event to the session timeline.

    Use this to record significant events: decisions made, errors encountered,
    actions taken, or observations during a session.

    Args:
        summary: Brief description of the event (<200 chars)
        event_type: Type: user_request, clarification, action, decision, observation, error, checkpoint
        details: Full context (optional, for longer descriptions)
        project_path: Project path
        session_id: Session identifier (auto-generated if omitted)
        status: Event status: completed, in_progress, failed, reverted
        outcome: Result description
        is_anchor: Mark as verified/anchor fact
    """
    app = _get_app(ctx)
    result = await timeline_log(
        db=app.db,
        embeddings=app.embeddings,
        session_id=session_id or str(uuid.uuid4()),
        event_type=event_type,
        summary=summary,
        details=details,
        project_path=project_path,
        status=status,
        outcome=outcome,
        is_anchor=is_anchor,
    )
    return json.dumps(result, default=str)


@mcp_server.tool()
async def memory_stats(ctx: Context) -> str:
    """Get memory statistics including total memories, database size, and breakdown by type."""
    app = _get_app(ctx)
    result = await app.db.get_stats()
    return json.dumps(result, default=str)


@mcp_server.tool()
async def memory_dashboard(ctx: Context) -> str:
    """Open the Claude Memory real-time dashboard in the browser."""
    import webbrowser

    url = config.get_dashboard_url()
    try:
        webbrowser.open(url)
        return json.dumps({
            "success": True,
            "url": url,
            "message": f"Dashboard opened at {url}",
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e),
            "message": f"Could not auto-open browser. Visit {url} manually.",
        })


@mcp_server.tool()
async def memory_sync_native(
    ctx: Context,
    project_path: Optional[str] = None,
    direction: str = "both",
) -> str:
    """Sync between MCP DB and Claude's native MEMORY.md.

    Bridges the MCP vector memory DB with Claude Code's built-in auto memory
    (~/.claude/projects/<slug>/memory/MEMORY.md).

    Args:
        project_path: Project path to sync. If omitted, syncs current project.
        direction: 'to_native' (MCP->MEMORY.md), 'from_native' (MEMORY.md->MCP), or 'both'
    """
    from services.native_memory_sync import (
        sync_mcp_to_native,
        sync_native_to_mcp,
        sync_bidirectional,
    )

    app = _get_app(ctx)

    if not project_path:
        return json.dumps({
            "success": False,
            "error": "project_path is required for native memory sync",
        })

    try:
        if direction == "to_native":
            result = await sync_mcp_to_native(app.db, project_path)
        elif direction == "from_native":
            result = await sync_native_to_mcp(app.db, app.embeddings, project_path)
        elif direction == "both":
            result = await sync_bidirectional(app.db, app.embeddings, project_path)
        else:
            return json.dumps({
                "success": False,
                "error": f"Invalid direction '{direction}'. Use 'to_native', 'from_native', or 'both'.",
            })

        return json.dumps(result, default=str)
    except Exception as e:
        logger.error(f"memory_sync_native failed: {e}")
        return json.dumps({"success": False, "error": str(e)})


# ── Cross-Session Awareness Tools ────────────────────────────────────────

@mcp_server.tool()
async def memory_active_sessions(
    ctx: Context,
    project_path: str,
    exclude_session_id: Optional[str] = None,
) -> str:
    """List active parallel Claude Code sessions for a project.

    Use this to see what other sessions are currently working on,
    what files they've modified, and detect potential conflicts.

    Args:
        project_path: Project path to check
        exclude_session_id: Optional session ID to exclude from results
    """
    app = _get_app(ctx)
    sessions = await app.db.get_active_sessions(project_path, exclude_session_id)
    return json.dumps({
        "success": True,
        "sessions": sessions,
        "count": len(sessions),
    }, default=str)


@mcp_server.tool()
async def memory_session_catchup(
    ctx: Context,
    project_path: str,
    session_id: Optional[str] = None,
    since: Optional[str] = None,
) -> str:
    """Get a catch-up summary of what other sessions did.

    Returns recent cross-session activity grouped by session,
    including file changes, decisions, and goals.

    Args:
        project_path: Project path to check
        session_id: Current session ID (to exclude own events)
        since: ISO timestamp to get events after (optional)
    """
    app = _get_app(ctx)
    from services.session_awareness import get_session_awareness
    awareness = get_session_awareness(app.db)
    result = await awareness.get_catchup(
        session_id=session_id or "",
        project_path=project_path,
        since=since,
    )
    return json.dumps(result, default=str)


# ── Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp_server.run(transport="stdio")
