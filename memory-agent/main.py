"""
Claude Memory Agent - A2A Server with FastAPI.

Provides semantic memory storage and retrieval for Claude Code sessions.
Implements Google A2A protocol for agent-to-agent communication.
Enhanced with rich context support for cross-project memory management.
"""
import os
import json
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from agent_card import AGENT_CARD
from services.database import DatabaseService, normalize_path
from services.embeddings import EmbeddingService

# Original memory skills
from skills.store import store_memory, store_project, store_pattern
from skills.retrieve import retrieve_memory
from skills.search import semantic_search, search_patterns, get_project_context
from skills.summarize import summarize_session

# Timeline skills (Anti-Hallucination Layer)
from skills.timeline import timeline_log, timeline_get, timeline_search, timeline_auto_detect
from skills.state import state_get, state_update, state_init_session
from skills.checkpoint import checkpoint_create, checkpoint_load, checkpoint_list
from skills.grounding import context_refresh, check_contradictions, verify_entity, mark_anchor

# CLAUDE.md management skills
from skills.claude_md import (
    claude_md_read, claude_md_add_section, claude_md_update_section,
    claude_md_add_instruction, claude_md_list_sections, claude_md_suggest_from_session
)

# Verification skills (Best-of-N, Quote Extraction)
from skills.verification import best_of_n_verify, extract_quotes, require_grounding

# Agent registry for dashboard
from services.agent_registry import (
    AVAILABLE_AGENTS, AVAILABLE_MCPS, AVAILABLE_HOOKS,
    AGENT_CATEGORIES, get_agents_by_category, get_agent_by_id
)

load_dotenv()

# Initialize services
db = DatabaseService()
embeddings = EmbeddingService()

# Task storage
tasks: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    await db.connect()
    await db.initialize_schema()
    print(f"Memory Agent v2.0 started on port {os.getenv('PORT', 8102)}")
    yield
    await db.disconnect()


app = FastAPI(
    title="Claude Memory Agent",
    description="Persistent semantic memory for Claude Code sessions with cross-project support",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============= Pydantic Models =============

class A2AMessage(BaseModel):
    role: str
    parts: list


class A2ARequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Any
    method: str
    params: Optional[Dict[str, Any]] = None


# ============= A2A Endpoints =============

@app.get("/.well-known/agent.json")
async def get_agent_card():
    return JSONResponse(content=AGENT_CARD)


@app.post("/a2a")
async def a2a_endpoint(request: A2ARequest):
    try:
        if request.method == "tasks/send":
            return await handle_task_send(request)
        elif request.method == "tasks/get":
            return await handle_task_get(request)
        elif request.method == "tasks/cancel":
            return await handle_task_cancel(request)
        else:
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": request.id,
                "error": {"code": -32601, "message": f"Method not found: {request.method}"}
            })
    except Exception as e:
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": request.id,
            "error": {"code": -32000, "message": str(e)}
        })


async def handle_task_send(request: A2ARequest) -> JSONResponse:
    params = request.params or {}
    task_id = params.get("id") or str(uuid.uuid4())
    message = params.get("message", {})
    session_id = params.get("sessionId")
    metadata = params.get("metadata", {})

    parts = message.get("parts", [])
    text_content = ""
    skill_id = metadata.get("skill_id", "semantic_search")
    skill_params = metadata.get("params", {})

    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            text_content = part.get("text", "")
        elif isinstance(part, str):
            text_content = part

    try:
        result = await execute_skill(
            skill_id=skill_id,
            query=text_content,
            params=skill_params,
            session_id=session_id
        )

        tasks[task_id] = {
            "id": task_id,
            "status": "completed",
            "result": result,
            "created_at": datetime.now().isoformat()
        }

        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": request.id,
            "result": {
                "id": task_id,
                "status": {"state": "completed"},
                "artifacts": [{"parts": [{"type": "text", "text": json.dumps(result, indent=2)}]}]
            }
        })

    except Exception as e:
        tasks[task_id] = {
            "id": task_id,
            "status": "failed",
            "error": str(e),
            "created_at": datetime.now().isoformat()
        }
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": request.id,
            "error": {"code": -32000, "message": str(e)}
        })


async def handle_task_get(request: A2ARequest) -> JSONResponse:
    params = request.params or {}
    task_id = params.get("id")

    if not task_id or task_id not in tasks:
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": request.id,
            "error": {"code": -32602, "message": f"Task not found: {task_id}"}
        })

    task = tasks[task_id]
    return JSONResponse(content={
        "jsonrpc": "2.0",
        "id": request.id,
        "result": {
            "id": task_id,
            "status": {"state": task["status"]},
            "artifacts": [{"parts": [{"type": "text", "text": json.dumps(task.get("result", {}), indent=2)}]}] if task.get("result") else []
        }
    })


async def handle_task_cancel(request: A2ARequest) -> JSONResponse:
    params = request.params or {}
    task_id = params.get("id")
    if task_id and task_id in tasks:
        tasks[task_id]["status"] = "cancelled"
    return JSONResponse(content={
        "jsonrpc": "2.0",
        "id": request.id,
        "result": {"id": task_id, "status": {"state": "cancelled"}}
    })


async def execute_skill(
    skill_id: str,
    query: str,
    params: Dict[str, Any],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Execute the specified skill with enhanced context support."""

    if skill_id == "store_memory":
        return await store_memory(
            db=db,
            embeddings=embeddings,
            content=params.get("content", query),
            memory_type=params.get("type", "chunk"),
            metadata=params.get("metadata"),
            session_id=session_id or params.get("session_id"),
            # Project context
            project_path=params.get("project_path"),
            project_name=params.get("project_name"),
            project_type=params.get("project_type"),
            tech_stack=params.get("tech_stack"),
            # Agent context
            agent_type=params.get("agent_type"),
            skill_used=params.get("skill_used"),
            tools_used=params.get("tools_used"),
            # Outcome
            outcome=params.get("outcome"),
            success=params.get("success"),
            # Classification
            tags=params.get("tags"),
            importance=params.get("importance", 5)
        )

    elif skill_id == "store_project":
        return await store_project(
            db=db,
            path=params.get("path"),
            name=params.get("name"),
            project_type=params.get("project_type"),
            tech_stack=params.get("tech_stack"),
            conventions=params.get("conventions"),
            preferences=params.get("preferences")
        )

    elif skill_id == "store_pattern":
        return await store_pattern(
            db=db,
            embeddings=embeddings,
            name=params.get("name"),
            solution=params.get("solution"),
            problem_type=params.get("problem_type"),
            tech_context=params.get("tech_context"),
            metadata=params.get("metadata")
        )

    elif skill_id == "retrieve_memory":
        return await retrieve_memory(
            db=db,
            memory_id=params.get("memory_id"),
            memory_type=params.get("type"),
            session_id=session_id or params.get("session_id"),
            project_path=params.get("project_path"),
            limit=params.get("limit", 10)
        )

    elif skill_id == "semantic_search":
        return await semantic_search(
            db=db,
            embeddings=embeddings,
            query=params.get("query", query),
            limit=params.get("limit", 10),
            memory_type=params.get("type"),
            session_id=session_id or params.get("session_id"),
            project_path=params.get("project_path"),
            agent_type=params.get("agent_type"),
            success_only=params.get("success_only", False),
            threshold=params.get("threshold", 0.5)
        )

    elif skill_id == "search_patterns":
        return await search_patterns(
            db=db,
            embeddings=embeddings,
            query=params.get("query", query),
            limit=params.get("limit", 5),
            problem_type=params.get("problem_type"),
            threshold=params.get("threshold", 0.5)
        )

    elif skill_id == "get_project_context":
        return await get_project_context(
            db=db,
            embeddings=embeddings,
            project_path=params.get("project_path"),
            query=params.get("query"),
            limit=params.get("limit", 10)
        )

    elif skill_id == "summarize_session":
        return await summarize_session(
            db=db,
            embeddings=embeddings,
            session_id=session_id or params.get("session_id", str(uuid.uuid4())),
            summary=params.get("summary", query),
            key_decisions=params.get("key_decisions"),
            code_patterns=params.get("code_patterns"),
            metadata=params.get("metadata"),
            project_path=params.get("project_path")
        )

    elif skill_id == "get_stats":
        return await db.get_stats()

    # ============================================================
    # TIMELINE SKILLS
    # ============================================================

    elif skill_id == "timeline_log":
        return await timeline_log(
            db=db,
            embeddings=embeddings,
            session_id=params.get("session_id") or session_id or str(uuid.uuid4()),
            event_type=params.get("event_type", "observation"),
            summary=params.get("summary", query),
            details=params.get("details"),
            project_path=params.get("project_path"),
            parent_event_id=params.get("parent_event_id"),
            root_event_id=params.get("root_event_id"),
            entities=params.get("entities"),
            status=params.get("status", "completed"),
            outcome=params.get("outcome"),
            confidence=params.get("confidence"),
            is_anchor=params.get("is_anchor", False)
        )

    elif skill_id == "timeline_get":
        return await timeline_get(
            db=db,
            session_id=params.get("session_id") or session_id,
            limit=params.get("limit", 20),
            event_type=params.get("event_type"),
            since_event_id=params.get("since_event_id"),
            anchors_only=params.get("anchors_only", False),
            include_state=params.get("include_state", True),
            include_checkpoint=params.get("include_checkpoint", True)
        )

    elif skill_id == "timeline_search":
        return await timeline_search(
            db=db,
            embeddings=embeddings,
            query=params.get("query", query),
            session_id=params.get("session_id") or session_id,
            limit=params.get("limit", 10),
            threshold=params.get("threshold", 0.5)
        )

    elif skill_id == "timeline_auto_detect":
        return await timeline_auto_detect(
            db=db,
            embeddings=embeddings,
            session_id=params.get("session_id") or session_id or str(uuid.uuid4()),
            response_text=params.get("response_text", query),
            project_path=params.get("project_path"),
            parent_event_id=params.get("parent_event_id")
        )

    # ============================================================
    # STATE SKILLS
    # ============================================================

    elif skill_id == "state_get":
        return await state_get(
            db=db,
            session_id=params.get("session_id") or session_id,
            project_path=params.get("project_path")
        )

    elif skill_id == "state_update":
        return await state_update(
            db=db,
            session_id=params.get("session_id") or session_id,
            current_goal=params.get("current_goal"),
            pending_questions=params.get("pending_questions"),
            add_question=params.get("add_question"),
            remove_question=params.get("remove_question"),
            register_entity=params.get("register_entity"),
            entity_registry=params.get("entity_registry"),
            add_decision=params.get("add_decision"),
            decisions_summary=params.get("decisions_summary")
        )

    elif skill_id == "state_init_session":
        return await state_init_session(
            db=db,
            embeddings=embeddings,
            project_path=params.get("project_path")
        )

    # ============================================================
    # CHECKPOINT SKILLS
    # ============================================================

    elif skill_id == "checkpoint_create":
        return await checkpoint_create(
            db=db,
            embeddings=embeddings,
            session_id=params.get("session_id") or session_id,
            summary=params.get("summary"),
            key_facts=params.get("key_facts"),
            include_state=params.get("include_state", True)
        )

    elif skill_id == "checkpoint_load":
        return await checkpoint_load(
            db=db,
            session_id=params.get("session_id") or session_id,
            checkpoint_id=params.get("checkpoint_id"),
            project_path=params.get("project_path")
        )

    elif skill_id == "checkpoint_list":
        return await checkpoint_list(
            db=db,
            session_id=params.get("session_id") or session_id,
            limit=params.get("limit", 10)
        )

    # ============================================================
    # GROUNDING SKILLS (Anti-Hallucination)
    # ============================================================

    elif skill_id == "context_refresh":
        return await context_refresh(
            db=db,
            embeddings=embeddings,
            session_id=params.get("session_id") or session_id,
            query=params.get("query", query) if query else None,
            include_recent_events=params.get("include_recent_events", 10),
            include_state=params.get("include_state", True),
            include_checkpoint=params.get("include_checkpoint", True),
            include_relevant_memories=params.get("include_relevant_memories", True),
            check_contradictions=params.get("check_contradictions", True)
        )

    elif skill_id == "check_contradictions":
        return await check_contradictions(
            db=db,
            embeddings=embeddings,
            statement=params.get("statement", query),
            session_id=params.get("session_id") or session_id,
            scope=params.get("scope", "session")
        )

    elif skill_id == "verify_entity":
        return await verify_entity(
            db=db,
            session_id=params.get("session_id") or session_id,
            entity_key=params.get("entity_key"),
            entity_type=params.get("entity_type")
        )

    elif skill_id == "mark_anchor":
        return await mark_anchor(
            db=db,
            embeddings=embeddings,
            session_id=params.get("session_id") or session_id,
            fact=params.get("fact", query),
            details=params.get("details"),
            project_path=params.get("project_path")
        )

    # ============================================================
    # CLAUDE.MD MANAGEMENT SKILLS
    # ============================================================

    elif skill_id == "claude_md_read":
        return await claude_md_read(
            section=params.get("section")
        )

    elif skill_id == "claude_md_add_section":
        return await claude_md_add_section(
            section_name=params.get("section_name"),
            content=params.get("content", query),
            position=params.get("position", "end")
        )

    elif skill_id == "claude_md_update_section":
        return await claude_md_update_section(
            section_name=params.get("section_name"),
            content=params.get("content", query),
            mode=params.get("mode", "replace")
        )

    elif skill_id == "claude_md_add_instruction":
        return await claude_md_add_instruction(
            section_name=params.get("section_name"),
            instruction=params.get("instruction", query),
            bullet_style=params.get("bullet_style", "-")
        )

    elif skill_id == "claude_md_list_sections":
        return await claude_md_list_sections()

    elif skill_id == "claude_md_suggest":
        return await claude_md_suggest_from_session(
            db=db,
            session_id=params.get("session_id") or session_id,
            min_importance=params.get("min_importance", 7)
        )

    # ============================================================
    # VERIFICATION SKILLS (Best-of-N, Quote Extraction)
    # ============================================================

    elif skill_id == "best_of_n_verify":
        return await best_of_n_verify(
            query=params.get("query", query),
            n=params.get("n", 3),
            context=params.get("context"),
            threshold=params.get("threshold", 0.7)
        )

    elif skill_id == "extract_quotes":
        return await extract_quotes(
            document=params.get("document", ""),
            query=params.get("query", query),
            max_quotes=params.get("max_quotes", 5),
            min_length=params.get("min_length", 20)
        )

    elif skill_id == "require_grounding":
        return await require_grounding(
            db=db,
            session_id=params.get("session_id") or session_id,
            statement=params.get("statement", query),
            source_type=params.get("source_type", "any")
        )

    else:
        raise ValueError(f"Unknown skill: {skill_id}")


# ============= REST API Endpoints =============

@app.get("/api/stats")
async def api_get_stats():
    stats = await db.get_stats()
    # Add timeline stats
    try:
        timeline_stats = await db.execute_query(
            "SELECT COUNT(*) as count FROM timeline_events"
        )
        stats["total_timeline_events"] = timeline_stats[0]["count"] if timeline_stats else 0
    except:
        stats["total_timeline_events"] = 0
    return stats


@app.get("/dashboard")
async def serve_dashboard():
    """Serve the monitoring dashboard."""
    from fastapi.responses import FileResponse
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    return FileResponse(dashboard_path, media_type="text/html")


@app.get("/api/projects")
async def get_all_projects():
    """Get all projects that have sessions."""
    try:
        # Get unique projects from session_state
        projects = await db.execute_query("""
            SELECT DISTINCT
                project_path,
                MAX(updated_at) as last_activity,
                COUNT(*) as session_count
            FROM session_state
            WHERE project_path IS NOT NULL
            GROUP BY project_path
            ORDER BY last_activity DESC
        """)

        # Also get from memories table
        memory_projects = await db.execute_query("""
            SELECT DISTINCT
                project_path,
                MAX(created_at) as last_activity,
                COUNT(*) as memory_count
            FROM memories
            WHERE project_path IS NOT NULL
            GROUP BY project_path
            ORDER BY last_activity DESC
        """)

        # Merge and deduplicate - normalize paths to prevent duplicates
        all_projects = {}
        for p in (projects or []):
            path = p.get('project_path')
            if path:
                # Normalize path to ensure consistent keys
                normalized = normalize_path(path)
                if normalized in all_projects:
                    # Merge session counts
                    all_projects[normalized]['session_count'] += p.get('session_count', 0)
                else:
                    all_projects[normalized] = {
                        'project_path': normalized,
                        'last_activity': p.get('last_activity'),
                        'session_count': p.get('session_count', 0),
                        'memory_count': 0
                    }

        for p in (memory_projects or []):
            path = p.get('project_path')
            if path:
                # Normalize path to ensure consistent keys
                normalized = normalize_path(path)
                if normalized in all_projects:
                    all_projects[normalized]['memory_count'] = p.get('memory_count', 0)
                else:
                    all_projects[normalized] = {
                        'project_path': normalized,
                        'last_activity': p.get('last_activity'),
                        'session_count': 0,
                        'memory_count': p.get('memory_count', 0)
                    }

        return {
            'success': True,
            'projects': list(all_projects.values()),
            'count': len(all_projects)
        }
    except Exception as e:
        return {'success': False, 'error': str(e), 'projects': []}


@app.get("/api/sessions/{project_path:path}")
async def get_project_sessions(project_path: str):
    """Get all sessions for a project."""
    # Normalize path to prevent duplicates from different separators
    project_path = normalize_path(project_path)
    try:
        sessions = await db.execute_query("""
            SELECT
                session_id,
                current_goal,
                updated_at,
                created_at
            FROM session_state
            WHERE project_path = ?
            ORDER BY updated_at DESC
        """, (project_path,))

        return {
            'success': True,
            'sessions': sessions or [],
            'count': len(sessions or [])
        }
    except Exception as e:
        return {'success': False, 'error': str(e), 'sessions': []}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0", "timestamp": datetime.now().isoformat()}


# ============= Agent Configuration API =============

@app.get("/api/agents")
async def get_all_agents():
    """Get all available agents with categories."""
    return {
        "success": True,
        "agents": AVAILABLE_AGENTS,
        "categories": AGENT_CATEGORIES,
        "by_category": get_agents_by_category(),
        "total": len(AVAILABLE_AGENTS)
    }


@app.get("/api/mcps")
async def get_all_mcps():
    """Get all available MCP servers."""
    return {
        "success": True,
        "mcps": AVAILABLE_MCPS,
        "total": len(AVAILABLE_MCPS)
    }


@app.get("/api/hooks")
async def get_all_hooks():
    """Get all available hooks."""
    return {
        "success": True,
        "hooks": AVAILABLE_HOOKS,
        "total": len(AVAILABLE_HOOKS)
    }


@app.get("/api/project/{project_path:path}/config")
async def get_project_config(project_path: str):
    """Get full configuration for a project."""
    # Normalize path to prevent duplicates from different separators
    project_path = normalize_path(project_path)
    try:
        # Get agent configs
        agent_configs = await db.execute_query(
            "SELECT * FROM project_agent_config WHERE project_path = ?",
            (project_path,)
        )

        # Get MCP configs
        mcp_configs = await db.execute_query(
            "SELECT * FROM project_mcp_config WHERE project_path = ?",
            (project_path,)
        )

        # Get hook configs
        hook_configs = await db.execute_query(
            "SELECT * FROM project_hook_config WHERE project_path = ?",
            (project_path,)
        )

        # Get project preferences
        prefs = await db.execute_query(
            "SELECT * FROM project_preferences WHERE project_path = ?",
            (project_path,)
        )

        # Build agent status map (enabled/disabled)
        agent_status = {}
        for config in (agent_configs or []):
            agent_status[config['agent_id']] = {
                'enabled': bool(config['enabled']),
                'priority': config['priority'],
                'settings': json.loads(config['settings']) if config['settings'] else {}
            }

        # Fill in defaults for unconfigured agents
        for agent in AVAILABLE_AGENTS:
            if agent['id'] not in agent_status:
                agent_status[agent['id']] = {
                    'enabled': agent['default_enabled'],
                    'priority': agent['priority'],
                    'settings': {}
                }

        # Build MCP status map
        mcp_status = {}
        for config in (mcp_configs or []):
            mcp_status[config['mcp_id']] = {
                'enabled': bool(config['enabled']),
                'settings': json.loads(config['settings']) if config['settings'] else {}
            }

        for mcp in AVAILABLE_MCPS:
            if mcp['id'] not in mcp_status:
                mcp_status[mcp['id']] = {
                    'enabled': mcp['default_enabled'],
                    'settings': {}
                }

        # Build hook status map
        hook_status = {}
        for config in (hook_configs or []):
            hook_status[config['hook_id']] = {
                'enabled': bool(config['enabled']),
                'settings': json.loads(config['settings']) if config['settings'] else {}
            }

        for hook in AVAILABLE_HOOKS:
            if hook['id'] not in hook_status:
                hook_status[hook['id']] = {
                    'enabled': hook['default_enabled'],
                    'settings': {}
                }

        return {
            "success": True,
            "project_path": project_path,
            "preferences": prefs[0] if prefs else None,
            "agents": agent_status,
            "mcps": mcp_status,
            "hooks": hook_status,
            "stats": {
                "enabled_agents": sum(1 for a in agent_status.values() if a['enabled']),
                "total_agents": len(AVAILABLE_AGENTS),
                "enabled_mcps": sum(1 for m in mcp_status.values() if m['enabled']),
                "total_mcps": len(AVAILABLE_MCPS),
                "enabled_hooks": sum(1 for h in hook_status.values() if h['enabled']),
                "total_hooks": len(AVAILABLE_HOOKS)
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/project/{project_path:path}/agent/{agent_id}")
async def update_agent_config(project_path: str, agent_id: str, request: Request):
    """Update agent configuration for a project."""
    # Normalize path to prevent duplicates from different separators
    project_path = normalize_path(project_path)
    try:
        body = await request.json()
        enabled = body.get('enabled', True)
        priority = body.get('priority', 5)
        settings = json.dumps(body.get('settings', {}))

        await db.execute_query(
            """
            INSERT INTO project_agent_config (project_path, agent_id, enabled, priority, settings)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(project_path, agent_id) DO UPDATE SET
                enabled = excluded.enabled,
                priority = excluded.priority,
                settings = excluded.settings,
                updated_at = datetime('now')
            """,
            (project_path, agent_id, 1 if enabled else 0, priority, settings)
        )
        db.conn.commit()

        return {"success": True, "agent_id": agent_id, "enabled": enabled}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/project/{project_path:path}/mcp/{mcp_id}")
async def update_mcp_config(project_path: str, mcp_id: str, request: Request):
    """Update MCP configuration for a project."""
    # Normalize path to prevent duplicates from different separators
    project_path = normalize_path(project_path)
    try:
        body = await request.json()
        enabled = body.get('enabled', True)
        settings = json.dumps(body.get('settings', {}))

        await db.execute_query(
            """
            INSERT INTO project_mcp_config (project_path, mcp_id, enabled, settings)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(project_path, mcp_id) DO UPDATE SET
                enabled = excluded.enabled,
                settings = excluded.settings,
                updated_at = datetime('now')
            """,
            (project_path, mcp_id, 1 if enabled else 0, settings)
        )
        db.conn.commit()

        return {"success": True, "mcp_id": mcp_id, "enabled": enabled}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/project/{project_path:path}/hook/{hook_id}")
async def update_hook_config(project_path: str, hook_id: str, request: Request):
    """Update hook configuration for a project."""
    # Normalize path to prevent duplicates from different separators
    project_path = normalize_path(project_path)
    try:
        body = await request.json()
        enabled = body.get('enabled', True)
        settings = json.dumps(body.get('settings', {}))

        await db.execute_query(
            """
            INSERT INTO project_hook_config (project_path, hook_id, enabled, settings)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(project_path, hook_id) DO UPDATE SET
                enabled = excluded.enabled,
                settings = excluded.settings,
                updated_at = datetime('now')
            """,
            (project_path, hook_id, 1 if enabled else 0, settings)
        )
        db.conn.commit()

        return {"success": True, "hook_id": hook_id, "enabled": enabled}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/project/{project_path:path}/preferences")
async def update_project_preferences(project_path: str, request: Request):
    """Update project preferences."""
    # Normalize path to prevent duplicates from different separators
    project_path = normalize_path(project_path)
    try:
        body = await request.json()

        await db.execute_query(
            """
            INSERT INTO project_preferences (project_path, name, description, color, icon, default_model, auto_memory, auto_checkpoint, settings)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_path) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                color = excluded.color,
                icon = excluded.icon,
                default_model = excluded.default_model,
                auto_memory = excluded.auto_memory,
                auto_checkpoint = excluded.auto_checkpoint,
                settings = excluded.settings,
                updated_at = datetime('now')
            """,
            (
                project_path,
                body.get('name'),
                body.get('description'),
                body.get('color', '#58a6ff'),
                body.get('icon', 'folder'),
                body.get('default_model', 'sonnet'),
                1 if body.get('auto_memory', True) else 0,
                1 if body.get('auto_checkpoint', True) else 0,
                json.dumps(body.get('settings', {}))
            )
        )
        db.conn.commit()

        return {"success": True, "project_path": project_path}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/project/{project_path:path}/agents/bulk")
async def bulk_update_agents(project_path: str, request: Request):
    """Bulk enable/disable agents for a project."""
    # Normalize path to prevent duplicates from different separators
    project_path = normalize_path(project_path)
    try:
        body = await request.json()
        updates = body.get('updates', {})  # {agent_id: enabled}

        for agent_id, enabled in updates.items():
            await db.execute_query(
                """
                INSERT INTO project_agent_config (project_path, agent_id, enabled)
                VALUES (?, ?, ?)
                ON CONFLICT(project_path, agent_id) DO UPDATE SET
                    enabled = excluded.enabled,
                    updated_at = datetime('now')
                """,
                (project_path, agent_id, 1 if enabled else 0)
            )

        db.conn.commit()

        return {"success": True, "updated": len(updates)}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8102)),
        reload=True
    )
