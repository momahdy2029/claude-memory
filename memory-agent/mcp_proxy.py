"""Slim MCP proxy for Claude Memory.

Thin adapter that exposes 3 unified tools over stdio JSON-RPC,
forwarding all work to the HTTP backend (main.py on port 8102).

NO embedding model loaded. NO database connection. Just HTTP calls.

Tools:
    memory_ask   - Unified search (replaces memory_search, memory_search_patterns,
                   memory_context, memory_get_project, memory_active_sessions,
                   memory_session_catchup)
    memory_store - Unified store (replaces memory_store, memory_store_pattern,
                   memory_store_project)
    memory_status - Quick stats + project info (replaces memory_stats, memory_dashboard)

Usage:
    python mcp_proxy.py              # stdio mode (default)
"""

# ── Suppress stdout noise before ANY library imports ─────────────────
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import logging

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("mcp-proxy")

# ── Imports ──────────────────────────────────────────────────────────

import json
import asyncio
from typing import Optional, List, Dict, Any

import httpx

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "MCP SDK not found. Install with: pip install 'mcp>=1.0.0'"
    )

# ── Config ───────────────────────────────────────────────────────────

BACKEND_URL = os.environ.get("MEMORY_AGENT_URL", "http://localhost:8102")
TIMEOUT = 5.0  # seconds per HTTP call

# ── HTTP helpers ─────────────────────────────────────────────────────


async def _rest_get(path: str, params: dict = None) -> Optional[dict]:
    """GET request to the backend REST API."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(f"{BACKEND_URL}{path}", params=params)
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.debug(f"REST GET {path} failed: {e}")
    return None


async def _rest_post(path: str, body: dict = None) -> Optional[dict]:
    """POST request to the backend REST API."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(f"{BACKEND_URL}{path}", json=body or {})
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.debug(f"REST POST {path} failed: {e}")
    return None


async def _a2a_skill(skill_id: str, params: dict) -> Optional[dict]:
    """Call a backend skill via the A2A JSON-RPC protocol."""
    payload = {
        "jsonrpc": "2.0",
        "id": f"proxy-{skill_id}",
        "method": "tasks/send",
        "params": {
            "message": {"parts": [{"type": "text", "text": ""}]},
            "metadata": {"skill_id": skill_id, "params": params},
        },
    }
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(f"{BACKEND_URL}/a2a", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                # Extract artifact text from A2A response
                try:
                    text = data["result"]["artifacts"][0]["parts"][0]["text"]
                    return json.loads(text)
                except (KeyError, IndexError, json.JSONDecodeError):
                    return data.get("result")
    except Exception as e:
        logger.debug(f"A2A skill {skill_id} failed: {e}")
    return None


# ── MCP Server ───────────────────────────────────────────────────────

mcp_server = FastMCP("claude-memory")


@mcp_server.tool()
async def memory_ask(
    query: str,
    project_path: Optional[str] = None,
    type_hint: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Search memories using natural language. Returns similar content ranked by relevance.

    Unified search that combines semantic search and pattern matching.
    Use type_hint to focus: "pattern" for solutions, "error" for past bugs,
    "session" for session context, "decision" for architectural choices.

    Args:
        query: Search query (natural language)
        project_path: Filter by project path
        type_hint: Focus search: pattern, error, session, decision, project, context
        limit: Max results (default 10)
    """
    tasks = []
    results: Dict[str, Any] = {"query": query}

    # -- Semantic search (always) --
    search_params = {"query": query, "limit": limit}
    if project_path:
        search_params["project_path"] = project_path
    if type_hint in ("error", "decision", "session", "code", "preference"):
        search_params["memory_type"] = type_hint

    tasks.append(("search", _rest_get("/api/search", search_params)))

    # -- Pattern search (always, lightweight) --
    pattern_params: dict = {"query": query, "limit": 5}
    if type_hint == "pattern":
        pattern_params["limit"] = limit
    tasks.append(("patterns", _a2a_skill("search_patterns", pattern_params)))

    # -- Project context (if type_hint is project/context, fetch project metadata) --
    if type_hint in ("project", "context") and project_path:
        tasks.append(("project", _a2a_skill("get_project_context", {
            "project_path": project_path,
            "limit": 5,
        })))

    # -- Session context (if type_hint is session) --
    if type_hint == "session" and project_path:
        tasks.append(("sessions", _rest_get("/api/sessions/active", {
            "project_path": project_path,
        })))

    # Run all in parallel
    gathered = await asyncio.gather(
        *[t[1] for t in tasks], return_exceptions=True
    )

    for (label, _), result in zip(tasks, gathered):
        if isinstance(result, Exception) or result is None:
            continue
        if label == "search" and result.get("results"):
            results["memories"] = result["results"][:limit]
        elif label == "patterns" and result.get("patterns"):
            results["patterns"] = result["patterns"][:5]
        elif label == "project" and result.get("project"):
            results["project_context"] = result["project"]
        elif label == "sessions":
            results["active_sessions"] = result.get("sessions", [])

    results["success"] = bool(results.get("memories") or results.get("patterns"))
    return json.dumps(results, default=str)


@mcp_server.tool()
async def memory_store(
    content: str,
    memory_type: str = "chunk",
    importance: int = 5,
    tags: Optional[List[str]] = None,
    project_path: Optional[str] = None,
    outcome: Optional[str] = None,
    success: Optional[bool] = None,
    # Pattern-specific fields
    pattern_name: Optional[str] = None,
    problem_type: Optional[str] = None,
    tech_context: Optional[List[str]] = None,
    # Project-specific fields
    project_name: Optional[str] = None,
    project_type: Optional[str] = None,
    tech_stack: Optional[List[str]] = None,
    conventions: Optional[Dict[str, Any]] = None,
    preferences: Optional[Dict[str, Any]] = None,
) -> str:
    """Store a memory, pattern, or project info. Routes automatically by type.

    For memories: set content, memory_type, importance, tags.
    For patterns: set content as solution, pattern_name, problem_type.
    For projects: set project_path, project_name, tech_stack, conventions.

    Args:
        content: Content to remember (or solution for patterns)
        memory_type: Type: session, decision, code, chunk, error, preference
        importance: 1-10 importance scale (default 5)
        tags: Classification tags
        project_path: Project path
        outcome: What happened
        success: Did it work?
        pattern_name: Pattern name (triggers pattern storage)
        problem_type: Pattern type: bug_fix, feature, refactor, config, performance
        tech_context: Technologies for pattern
        project_name: Project name (triggers project storage)
        project_type: Project type (wordpress, react, etc.)
        tech_stack: Technologies used
        conventions: Coding conventions dict
        preferences: User preferences dict
    """
    # Route to pattern storage
    if pattern_name:
        result = await _a2a_skill("store_pattern", {
            "name": pattern_name,
            "solution": content,
            "problem_type": problem_type,
            "tech_context": tech_context,
        })
        return json.dumps(result or {"error": "Memory agent unavailable"}, default=str)

    # Route to project storage
    if project_type or tech_stack or conventions or preferences:
        result = await _a2a_skill("store_project", {
            "path": project_path or "",
            "name": project_name,
            "project_type": project_type,
            "tech_stack": tech_stack,
            "conventions": conventions,
            "preferences": preferences,
        })
        return json.dumps(result or {"error": "Memory agent unavailable"}, default=str)

    # Default: store memory
    result = await _a2a_skill("store_memory", {
        "content": content,
        "memory_type": memory_type,
        "importance": importance,
        "tags": tags,
        "project_path": project_path,
        "outcome": outcome,
        "success": success,
    })
    return json.dumps(result or {"error": "Memory agent unavailable"}, default=str)


@mcp_server.tool()
async def memory_status(
    project_path: Optional[str] = None,
) -> str:
    """Get memory system status: stats, project info, and health check.

    Args:
        project_path: Optional project path for project-specific info
    """
    tasks = [("stats", _rest_get("/api/stats"))]

    if project_path:
        tasks.append(("project", _a2a_skill("get_project_context", {
            "project_path": project_path,
            "limit": 3,
        })))

    gathered = await asyncio.gather(
        *[t[1] for t in tasks], return_exceptions=True
    )

    result: Dict[str, Any] = {"success": True}

    for (label, _), data in zip(tasks, gathered):
        if isinstance(data, Exception) or data is None:
            continue
        if label == "stats":
            result["stats"] = data
        elif label == "project":
            result["project"] = data

    if not result.get("stats"):
        result["success"] = False
        result["error"] = "Memory agent unavailable - is main.py running on port 8102?"

    return json.dumps(result, default=str)


# ── Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"Starting slim MCP proxy -> {BACKEND_URL}")
    mcp_server.run(transport="stdio")
