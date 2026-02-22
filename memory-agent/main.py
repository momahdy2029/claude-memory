"""
Claude Memory Agent - A2A Server with FastAPI.

Provides semantic memory storage and retrieval for Claude Code sessions.
Implements Google A2A protocol for agent-to-agent communication.
Enhanced with rich context support for cross-project memory management.
"""
import os
import json
import uuid
import asyncio
import sqlite3
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from agent_card import AGENT_CARD
from services.database import (
    DatabaseService, normalize_path,
    DatabaseError, ConnectionPoolError, QueryTimeoutError,
    RetryExhaustedError, MigrationError
)
from services.embeddings import EmbeddingService
from services.auth import get_auth_service, AuthService
from services.response_manager import fit_response

# Original memory skills
from skills.store import store_memory, store_project, store_pattern
from skills.retrieve import retrieve_memory
from skills.search import semantic_search, search_patterns, get_project_context
from skills.summarize import (
    summarize_session, auto_summarize_session, get_session_handoff,
    create_diary_entry, check_session_inactivity
)

# Timeline skills (Anti-Hallucination Layer)
from skills.timeline import timeline_log, timeline_log_batch, timeline_get, timeline_search, timeline_auto_detect, timeline_chain
from skills.state import state_get, state_update, state_init_session
from skills.checkpoint import checkpoint_create, checkpoint_load, checkpoint_list
from skills.grounding import (
    context_refresh, check_contradictions, verify_entity, mark_anchor,
    get_unresolved_conflicts, resolve_conflict, get_anchor_history, auto_resolve_conflicts
)

# CLAUDE.md management skills
from skills.claude_md import (
    claude_md_read, claude_md_add_section, claude_md_update_section,
    claude_md_add_instruction, claude_md_list_sections, claude_md_suggest_from_session
)

# Verification skills (Best-of-N, Quote Extraction)
from skills.verification import best_of_n_verify, extract_quotes, require_grounding

# Cross-session learning skills
from skills.insights import (
    run_aggregation, get_insights, suggest_improvements,
    record_insight_feedback, mark_insight_applied, get_project_insights
)

# Memory cleanup skills
from skills.cleanup import (
    memory_cleanup, get_archived_memories, restore_memory,
    get_cleanup_config, set_cleanup_config, get_cleanup_stats,
    purge_expired_archives
)

# Admin skills (embedding model management, reindexing)
from skills.admin import (
    get_embedding_status, switch_embedding_model, reindex_memories,
    get_reindex_progress, cancel_reindex, get_model_info, get_system_stats
)

# Natural language memory interface
from skills.natural_language import process_natural_command

# Session review skills (end-of-session memory verification)
from skills.session_review import (
    get_session_memories, review_session_memories,
    suggest_session_reviews, get_recent_sessions, bulk_review_by_type
)

# WebSocket service for real-time updates
from services.websocket import get_websocket_manager, broadcast_event, EventTypes

# Auto-injection service for mid-task relevance
from services.auto_inject import get_auto_injector

# Confidence scoring service
from services.confidence import get_confidence_service

# CLAUDE.md sync service
from services.claude_md_sync import get_claude_md_sync

# Cross-session awareness
from services.session_awareness import get_session_awareness

# Agent registry for dashboard
from services.agent_registry import (
    AVAILABLE_AGENTS, AVAILABLE_MCPS, AVAILABLE_HOOKS,
    AGENT_CATEGORIES, get_agents_by_category, get_agent_by_id,
    load_configured_hooks, load_configured_mcps
)

load_dotenv()

# ---------------------------------------------------------------------------
# Simple metrics tracker for search/store operations
# ---------------------------------------------------------------------------
class OperationMetrics:
    """Lightweight in-memory metrics for search and store operations.

    Tracks hit rates, result counts, and operation frequencies to answer:
    "Are stored memories actually being found?"
    """

    def __init__(self):
        self.search_total = 0
        self.search_hits = 0  # searches that returned >= 1 result
        self.search_empty = 0  # searches that returned 0 results
        self.search_result_counts: List[int] = []  # rolling window of result counts
        self.store_total = 0
        self.store_merged = 0  # dedup merges
        self.store_without_embedding = 0
        self._max_history = 1000  # keep last 1000 result counts

    def record_search(self, result_count: int):
        self.search_total += 1
        if result_count > 0:
            self.search_hits += 1
        else:
            self.search_empty += 1
        self.search_result_counts.append(result_count)
        if len(self.search_result_counts) > self._max_history:
            self.search_result_counts = self.search_result_counts[-self._max_history:]

    def record_store(self, merged: bool = False, has_embedding: bool = True):
        self.store_total += 1
        if merged:
            self.store_merged += 1
        if not has_embedding:
            self.store_without_embedding += 1

    def to_dict(self) -> dict:
        avg_results = (
            sum(self.search_result_counts) / len(self.search_result_counts)
            if self.search_result_counts else 0
        )
        hit_rate = (
            self.search_hits / self.search_total
            if self.search_total > 0 else 0
        )
        return {
            "search": {
                "total": self.search_total,
                "hits": self.search_hits,
                "empty": self.search_empty,
                "hit_rate": round(hit_rate, 3),
                "avg_result_count": round(avg_results, 1)
            },
            "store": {
                "total": self.store_total,
                "merged": self.store_merged,
                "without_embedding": self.store_without_embedding
            }
        }


metrics = OperationMetrics()

# Initialize services
db = DatabaseService()
from config import config as _cfg
embeddings = EmbeddingService(
    provider_type=_cfg.EMBEDDING_PROVIDER,
    model=_cfg.EMBEDDING_MODEL,
)

# Retry queue (imported lazily to avoid circular imports)
retry_queue = None

# Task storage
tasks: Dict[str, Dict[str, Any]] = {}


async def process_queued_request(item: Dict[str, Any]) -> bool:
    """Process a single queued request."""
    import httpx

    endpoint = item.get("endpoint", "")
    method = item.get("method", "POST")
    payload = item.get("payload", {})
    headers = item.get("headers", {})

    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(headers, str):
        headers = json.loads(headers) if headers else {}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method.upper() == "POST":
                response = await client.post(endpoint, json=payload, headers=headers)
            elif method.upper() == "GET":
                response = await client.get(endpoint, headers=headers)
            else:
                return False

            return response.status_code < 400
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global retry_queue

    await db.connect()
    await db.initialize_schema()

    # Initialize retry queue
    from services.retry_queue import get_queue
    retry_queue = get_queue()

    # Start background queue processor
    queue_task = asyncio.create_task(
        retry_queue.process_queue(
            processor=process_queued_request,
            batch_size=10,
            interval_seconds=5.0
        )
    )

    # Start curator maintenance scheduler
    from services.curator import run_curator_scheduler
    curator_interval = int(os.getenv("CURATOR_INTERVAL_HOURS", "24"))
    curator_task = asyncio.create_task(
        run_curator_scheduler(db, embeddings, interval_hours=curator_interval)
    )

    # Initialize embedding pipeline with LRU cache
    from services.embedding_pipeline import get_embedding_pipeline
    pipeline = get_embedding_pipeline(embeddings, db)

    # Start embedding pre-computation background loop
    from config import config as app_config

    async def precompute_loop():
        """Background: generate embeddings for memories missing them."""
        interval = app_config.EMBEDDING_PRECOMPUTE_INTERVAL
        while True:
            await asyncio.sleep(interval)
            try:
                result = await pipeline.precompute_missing_embeddings()
                if result.get('generated', 0) > 0:
                    logger.info(f"Pre-computed {result['generated']} embeddings")
            except Exception as e:
                logger.debug(f"Precompute loop error: {e}")

    precompute_task = asyncio.create_task(precompute_loop())

    # Start consolidation background loop
    async def consolidation_loop():
        """Background: consolidate similar warm-tier memories."""
        interval_hours = app_config.CONSOLIDATION_INTERVAL_HOURS
        while True:
            await asyncio.sleep(interval_hours * 3600)
            try:
                from services.consolidation import ConsolidationService
                consolidator = ConsolidationService(db, embeddings)
                result = await consolidator.run_consolidation()
                if result.get('consolidated', 0) > 0:
                    logger.info(
                        f"Consolidated {result['consolidated']} groups "
                        f"({result['memories_archived']} memories archived)"
                    )
            except Exception as e:
                logger.debug(f"Consolidation loop error: {e}")

    consolidation_task = asyncio.create_task(consolidation_loop())

    # Start cross-session stale cleanup loop
    async def session_cleanup_loop():
        """Background: clean up stale/expired sessions."""
        interval = app_config.SESSION_CLEANUP_INTERVAL_SECONDS
        while True:
            await asyncio.sleep(interval)
            try:
                awareness = get_session_awareness(db)
                result = await awareness.cleanup_stale(
                    idle_minutes=app_config.SESSION_IDLE_THRESHOLD_MINUTES,
                    completed_minutes=app_config.SESSION_COMPLETED_THRESHOLD_MINUTES,
                )
                total = sum(result.values())
                if total > 0:
                    logger.info(f"Session cleanup: {result}")
            except Exception as e:
                logger.debug(f"Session cleanup loop error: {e}")

    session_cleanup_task = asyncio.create_task(session_cleanup_loop())

    # Collect DB stats for splash
    auth_stats = auth_service.get_stats()
    db_stats = None
    try:
        db_stats = await db.get_stats()
    except Exception:
        pass

    # Rich terminal splash screen
    try:
        from services.terminal_ui import print_splash, setup_rich_logging

        print_splash(
            version="2.4.0",
            port=int(os.getenv("PORT", 8102)),
            auth_enabled=auth_stats.get("enabled", False),
            auth_keys=auth_stats.get("active_keys", 0),
            queue_depth=retry_queue.get_queue_depth(),
            curator_interval=curator_interval,
            embedding_cache_size=app_config.EMBEDDING_CACHE_SIZE,
            precompute_interval=app_config.EMBEDDING_PRECOMPUTE_INTERVAL,
            consolidation_threshold=app_config.CONSOLIDATION_THRESHOLD,
            consolidation_interval=app_config.CONSOLIDATION_INTERVAL_HOURS,
            db_stats=db_stats,
        )

        # Install rich logging handler for prettier output
        rich_handler = setup_rich_logging(app_config.LOG_LEVEL)
        logging.root.handlers = [rich_handler]

    except ImportError:
        # Fallback to plain output if rich unavailable
        print(f"Memory Agent v2.4.0 (CLaRa) started on port {os.getenv('PORT', 8102)}")
        if auth_stats.get("enabled"):
            print(f"Authentication: ENABLED ({auth_stats.get('active_keys', 0)} active keys)")
        else:
            print("Authentication: DISABLED")

    yield

    # Cleanup
    retry_queue.stop_processing()
    queue_task.cancel()
    curator_task.cancel()
    precompute_task.cancel()
    consolidation_task.cancel()
    session_cleanup_task.cancel()
    for task in [queue_task, curator_task, precompute_task, consolidation_task, session_cleanup_task]:
        try:
            await task
        except asyncio.CancelledError:
            pass
    retry_queue.close()
    await db.disconnect()


app = FastAPI(
    title="Claude Memory Agent",
    description="Persistent semantic memory for Claude Code sessions with cross-project support",
    version="2.4.0",
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

# Initialize auth service
auth_service = get_auth_service()


# Authentication middleware
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Validate API key for protected endpoints."""
    path = request.url.path

    # Skip auth for exempt endpoints
    if auth_service.is_exempt(path):
        return await call_next(request)

    # Skip if auth is disabled
    if not auth_service.enabled:
        return await call_next(request)

    # Get API key from header
    api_key = request.headers.get("X-Memory-Key")

    # Validate key
    valid, error, key_info = auth_service.validate_key(api_key)

    if not valid:
        return JSONResponse(
            status_code=401 if error == "Missing API key" else 403,
            content={
                "error": error,
                "code": "AUTH_REQUIRED" if error == "Missing API key" else "AUTH_FAILED"
            },
            headers={"WWW-Authenticate": "X-Memory-Key"}
        )

    # Add key info to request state for downstream use
    request.state.auth = key_info

    # Add rate limit headers to response
    response = await call_next(request)
    if key_info:
        response.headers["X-RateLimit-Remaining"] = str(key_info.get("rate_remaining", 0))

    return response


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
                "artifacts": [{"parts": [{"type": "text", "text": fit_response(result)}]}]
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
            "artifacts": [{"parts": [{"type": "text", "text": fit_response(task.get("result", {}))}]}] if task.get("result") else []
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


# ============================================================
# SKILL HANDLER FUNCTIONS
# ============================================================
# Each handler receives (query, params, session_id) and returns a dict.
# Grouped by category for maintainability.
# ============================================================


# --- Core Memory Skills ---

async def _handle_store_memory(query, params, session_id):
    result = await store_memory(
        db=db,
        embeddings=embeddings,
        content=params.get("content", query),
        memory_type=params.get("type", "chunk"),
        metadata=params.get("metadata"),
        session_id=session_id or params.get("session_id"),
        project_path=params.get("project_path"),
        project_name=params.get("project_name"),
        project_type=params.get("project_type"),
        tech_stack=params.get("tech_stack"),
        agent_type=params.get("agent_type"),
        skill_used=params.get("skill_used"),
        tools_used=params.get("tools_used"),
        outcome=params.get("outcome"),
        success=params.get("success"),
        tags=params.get("tags"),
        importance=params.get("importance", 5),
        confidence=params.get("confidence", 0.5),
        outcome_status=params.get("outcome_status", "pending"),
        fixed=params.get("fixed"),
        did_not_fix=params.get("did_not_fix"),
        caused=params.get("caused")
    )
    metrics.record_store(
        merged=result.get("action") == "merged",
        has_embedding=result.get("has_embedding", True)
    )
    try:
        await broadcast_event(
            EventTypes.MEMORY_STORED,
            {"memory_id": result.get("memory_id"), "type": params.get("type", "chunk")},
            params.get("project_path")
        )
    except Exception as e:
        logger.debug(f"Broadcast error: {e}")
    return result


async def _handle_store_project(query, params, session_id):
    return await store_project(
        db=db,
        path=params.get("path"),
        name=params.get("name"),
        project_type=params.get("project_type"),
        tech_stack=params.get("tech_stack"),
        conventions=params.get("conventions"),
        preferences=params.get("preferences")
    )


async def _handle_store_pattern(query, params, session_id):
    return await store_pattern(
        db=db,
        embeddings=embeddings,
        name=params.get("name"),
        solution=params.get("solution"),
        problem_type=params.get("problem_type"),
        tech_context=params.get("tech_context"),
        metadata=params.get("metadata")
    )


async def _handle_retrieve_memory(query, params, session_id):
    return await retrieve_memory(
        db=db,
        memory_id=params.get("memory_id"),
        memory_type=params.get("type"),
        session_id=session_id or params.get("session_id"),
        project_path=params.get("project_path"),
        limit=params.get("limit", 10)
    )


async def _handle_semantic_search(query, params, session_id):
    result = await semantic_search(
        db=db,
        embeddings=embeddings,
        query=params.get("query", query),
        limit=params.get("limit", 10),
        memory_type=params.get("type"),
        session_id=session_id or params.get("session_id"),
        project_path=params.get("project_path"),
        agent_type=params.get("agent_type"),
        success_only=params.get("success_only", False),
        threshold=params.get("threshold", 0.5),
        include_failed=params.get("include_failed", False),
        include_superseded=params.get("include_superseded", False),
        include_unreliable=params.get("include_unreliable", False),
        outcome_status=params.get("outcome_status"),
        include_graph=params.get("include_graph", True),
        temperature=params.get("temperature")
    )
    metrics.record_search(result.get("count", 0))
    return result


async def _handle_search_patterns(query, params, session_id):
    result = await search_patterns(
        db=db,
        embeddings=embeddings,
        query=params.get("query", query),
        limit=params.get("limit", 5),
        problem_type=params.get("problem_type"),
        threshold=params.get("threshold", 0.5)
    )
    metrics.record_search(result.get("count", 0))
    return result


async def _handle_get_project_context(query, params, session_id):
    return await get_project_context(
        db=db,
        embeddings=embeddings,
        project_path=params.get("project_path"),
        query=params.get("query"),
        limit=params.get("limit", 10)
    )


# --- Session Skills ---

async def _handle_summarize_session(query, params, session_id):
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


async def _handle_auto_summarize_session(query, params, session_id):
    return await auto_summarize_session(
        db=db,
        embeddings=embeddings,
        session_id=session_id or params.get("session_id"),
        project_path=params.get("project_path")
    )


async def _handle_get_session_handoff(query, params, session_id):
    return await get_session_handoff(
        db=db,
        embeddings=embeddings,
        project_path=params.get("project_path"),
        include_last_n_sessions=params.get("include_last_n_sessions", 3)
    )


async def _handle_create_diary_entry(query, params, session_id):
    return await create_diary_entry(
        db=db,
        embeddings=embeddings,
        session_id=session_id or params.get("session_id"),
        project_path=params.get("project_path"),
        user_notes=params.get("user_notes")
    )


async def _handle_check_session_inactivity(query, params, session_id):
    return await check_session_inactivity(
        db=db,
        session_id=session_id or params.get("session_id"),
        inactivity_threshold_hours=params.get("inactivity_threshold_hours", 4.0)
    )


async def _handle_get_stats(query, params, session_id):
    return await db.get_stats()


# --- Timeline Skills ---

async def _handle_timeline_log(query, params, session_id):
    result = await timeline_log(
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
    await broadcast_event(
        EventTypes.TIMELINE_LOGGED,
        {"event_id": result.get("event_id"), "event_type": params.get("event_type", "observation")},
        params.get("project_path")
    )
    return result


async def _handle_timeline_log_batch(query, params, session_id):
    result = await timeline_log_batch(
        db=db,
        embeddings=embeddings,
        session_id=params.get("session_id") or session_id or str(uuid.uuid4()),
        events=params.get("events", []),
        project_path=params.get("project_path"),
        parent_event_id=params.get("parent_event_id"),
        root_event_id=params.get("root_event_id")
    )
    if result.get("events_logged", 0) > 0:
        await broadcast_event(
            EventTypes.TIMELINE_LOGGED,
            {
                "event_ids": result.get("event_ids", []),
                "batch_size": result.get("events_logged", 0),
                "event_types": result.get("event_types", {})
            },
            params.get("project_path")
        )
    return result


async def _handle_timeline_get(query, params, session_id):
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


async def _handle_timeline_search(query, params, session_id):
    return await timeline_search(
        db=db,
        embeddings=embeddings,
        query=params.get("query", query),
        session_id=params.get("session_id") or session_id,
        limit=params.get("limit", 10),
        threshold=params.get("threshold", 0.5)
    )


async def _handle_timeline_auto_detect(query, params, session_id):
    return await timeline_auto_detect(
        db=db,
        embeddings=embeddings,
        session_id=params.get("session_id") or session_id or str(uuid.uuid4()),
        response_text=params.get("response_text", query),
        project_path=params.get("project_path"),
        parent_event_id=params.get("parent_event_id")
    )


async def _handle_timeline_chain(query, params, session_id):
    return await timeline_chain(
        db=db,
        session_id=params.get("session_id") or session_id,
        root_event_id=params.get("root_event_id"),
        include_details=params.get("include_details", False)
    )


# --- State Skills ---

async def _handle_state_get(query, params, session_id):
    return await state_get(
        db=db,
        session_id=params.get("session_id") or session_id,
        project_path=params.get("project_path")
    )


async def _handle_state_update(query, params, session_id):
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


async def _handle_state_init_session(query, params, session_id):
    return await state_init_session(
        db=db,
        embeddings=embeddings,
        project_path=params.get("project_path")
    )


# --- Checkpoint Skills ---

async def _handle_checkpoint_create(query, params, session_id):
    return await checkpoint_create(
        db=db,
        embeddings=embeddings,
        session_id=params.get("session_id") or session_id,
        summary=params.get("summary"),
        key_facts=params.get("key_facts"),
        include_state=params.get("include_state", True)
    )


async def _handle_checkpoint_load(query, params, session_id):
    return await checkpoint_load(
        db=db,
        session_id=params.get("session_id") or session_id,
        checkpoint_id=params.get("checkpoint_id"),
        project_path=params.get("project_path")
    )


async def _handle_checkpoint_list(query, params, session_id):
    return await checkpoint_list(
        db=db,
        session_id=params.get("session_id") or session_id,
        limit=params.get("limit", 10)
    )


# --- Grounding Skills (Anti-Hallucination) ---

async def _handle_context_refresh(query, params, session_id):
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


async def _handle_check_contradictions(query, params, session_id):
    return await check_contradictions(
        db=db,
        embeddings=embeddings,
        statement=params.get("statement", query),
        session_id=params.get("session_id") or session_id,
        scope=params.get("scope", "session")
    )


async def _handle_verify_entity(query, params, session_id):
    return await verify_entity(
        db=db,
        session_id=params.get("session_id") or session_id,
        entity_key=params.get("entity_key"),
        entity_type=params.get("entity_type")
    )


async def _handle_mark_anchor(query, params, session_id):
    result = await mark_anchor(
        db=db,
        embeddings=embeddings,
        session_id=params.get("session_id") or session_id,
        fact=params.get("fact", query),
        details=params.get("details"),
        project_path=params.get("project_path"),
        force=params.get("force", False)
    )
    event_type = EventTypes.ANCHOR_CONFLICT if result.get("conflict_detected") else EventTypes.ANCHOR_MARKED
    await broadcast_event(
        event_type,
        {"anchor_id": result.get("anchor_id"), "fact": params.get("fact", query)[:100]},
        params.get("project_path")
    )
    return result


async def _handle_get_unresolved_conflicts(query, params, session_id):
    return await get_unresolved_conflicts(
        db=db,
        session_id=params.get("session_id") or session_id,
        project_path=params.get("project_path"),
        limit=params.get("limit", 20)
    )


async def _handle_resolve_conflict(query, params, session_id):
    return await resolve_conflict(
        db=db,
        embeddings=embeddings,
        conflict_id=params.get("conflict_id"),
        resolution=params.get("resolution"),
        keep_anchor_id=params.get("keep_anchor_id"),
        resolved_by=params.get("resolved_by", "user")
    )


async def _handle_get_anchor_history(query, params, session_id):
    return await get_anchor_history(
        db=db,
        anchor_id=params.get("anchor_id"),
        session_id=params.get("session_id") or session_id,
        limit=params.get("limit", 50)
    )


async def _handle_auto_resolve_conflicts(query, params, session_id):
    return await auto_resolve_conflicts(
        db=db,
        embeddings=embeddings,
        session_id=params.get("session_id") or session_id
    )


# --- Self-Correcting Confidence Skills ---

async def _handle_memory_worked(query, params, session_id):
    from skills.confidence_tracker import report_solution_outcome
    result = await report_solution_outcome(
        db=db,
        memory_id=params.get("memory_id"),
        worked=True,
        context=params.get("context")
    )
    if result.get("success"):
        await broadcast_event(
            EventTypes.MEMORY_UPDATED,
            {
                "memory_id": params.get("memory_id"),
                "action": "worked",
                "new_confidence": result.get("new_confidence"),
                "reliability": result.get("reliability")
            }
        )
    return result


async def _handle_memory_failed(query, params, session_id):
    from skills.confidence_tracker import report_solution_outcome
    result = await report_solution_outcome(
        db=db,
        memory_id=params.get("memory_id"),
        worked=False,
        context=params.get("context")
    )
    if result.get("success"):
        await broadcast_event(
            EventTypes.MEMORY_UPDATED,
            {
                "memory_id": params.get("memory_id"),
                "action": "failed",
                "new_confidence": result.get("new_confidence"),
                "reliability": result.get("reliability"),
                "is_unreliable": result.get("is_unreliable")
            }
        )
    return result


async def _handle_get_reliability_stats(query, params, session_id):
    from skills.confidence_tracker import get_reliability_stats as _get_reliability_stats
    return await _get_reliability_stats(
        db=db,
        memory_id=params.get("memory_id")
    )


async def _handle_get_unreliable_memories(query, params, session_id):
    from skills.confidence_tracker import get_unreliable_memories as _get_unreliable_memories
    return await _get_unreliable_memories(
        db=db,
        project_path=params.get("project_path"),
        limit=params.get("limit", 50)
    )


async def _handle_reset_memory_reliability(query, params, session_id):
    from skills.confidence_tracker import reset_memory_reliability as _reset_memory_reliability
    return await _reset_memory_reliability(
        db=db,
        memory_id=params.get("memory_id"),
        new_confidence=params.get("confidence", 0.5)
    )


# --- CLAUDE.MD Management Skills ---

async def _handle_claude_md_read(query, params, session_id):
    return await claude_md_read(
        section=params.get("section")
    )


async def _handle_claude_md_add_section(query, params, session_id):
    return await claude_md_add_section(
        section_name=params.get("section_name"),
        content=params.get("content", query),
        position=params.get("position", "end")
    )


async def _handle_claude_md_update_section(query, params, session_id):
    return await claude_md_update_section(
        section_name=params.get("section_name"),
        content=params.get("content", query),
        mode=params.get("mode", "replace")
    )


async def _handle_claude_md_add_instruction(query, params, session_id):
    return await claude_md_add_instruction(
        section_name=params.get("section_name"),
        instruction=params.get("instruction", query),
        bullet_style=params.get("bullet_style", "-")
    )


async def _handle_claude_md_list_sections(query, params, session_id):
    return await claude_md_list_sections()


async def _handle_claude_md_suggest(query, params, session_id):
    return await claude_md_suggest_from_session(
        db=db,
        session_id=params.get("session_id") or session_id,
        min_importance=params.get("min_importance", 7)
    )


# --- Verification Skills ---

async def _handle_best_of_n_verify(query, params, session_id):
    return await best_of_n_verify(
        query=params.get("query", query),
        n=params.get("n", 3),
        context=params.get("context"),
        threshold=params.get("threshold", 0.7)
    )


async def _handle_extract_quotes(query, params, session_id):
    return await extract_quotes(
        document=params.get("document", ""),
        query=params.get("query", query),
        max_quotes=params.get("max_quotes", 5),
        min_length=params.get("min_length", 20)
    )


async def _handle_require_grounding(query, params, session_id):
    return await require_grounding(
        db=db,
        session_id=params.get("session_id") or session_id,
        statement=params.get("statement", query),
        source_type=params.get("source_type", "any")
    )


# --- Cross-Session Learning Skills ---

async def _handle_run_aggregation(query, params, session_id):
    return await run_aggregation(
        db=db,
        embeddings=embeddings,
        days_back=params.get("days_back", 30)
    )


async def _handle_get_insights(query, params, session_id):
    return await get_insights(
        db=db,
        embeddings=embeddings,
        insight_type=params.get("insight_type"),
        project_path=params.get("project_path"),
        min_confidence=params.get("min_confidence", 0.5),
        limit=params.get("limit", 10)
    )


async def _handle_suggest_improvements(query, params, session_id):
    return await suggest_improvements(
        db=db,
        embeddings=embeddings,
        min_confidence=params.get("min_confidence", 0.7)
    )


async def _handle_record_insight_feedback(query, params, session_id):
    return await record_insight_feedback(
        db=db,
        embeddings=embeddings,
        insight_id=params.get("insight_id"),
        helpful=params.get("helpful", True),
        session_id=session_id or params.get("session_id"),
        comment=params.get("comment")
    )


async def _handle_mark_insight_applied(query, params, session_id):
    return await mark_insight_applied(
        db=db,
        embeddings=embeddings,
        insight_id=params.get("insight_id")
    )


async def _handle_get_project_insights(query, params, session_id):
    return await get_project_insights(
        db=db,
        embeddings=embeddings,
        project_path=params.get("project_path"),
        include_global=params.get("include_global", True),
        limit=params.get("limit", 10)
    )


# --- Memory Cleanup Skills ---

async def _handle_memory_cleanup(query, params, session_id):
    result = await memory_cleanup(
        db=db,
        embeddings=embeddings,
        project_path=params.get("project_path"),
        dry_run=params.get("dry_run", True)
    )
    if not params.get("dry_run", True):
        await broadcast_event(
            EventTypes.CLEANUP_COMPLETED,
            {"archived": result.get("total_archived", 0), "deleted": result.get("total_deleted", 0)},
            params.get("project_path")
        )
    return result


async def _handle_get_archived_memories(query, params, session_id):
    return await get_archived_memories(
        db=db,
        embeddings=embeddings,
        project_path=params.get("project_path"),
        reason=params.get("reason"),
        limit=params.get("limit", 50)
    )


async def _handle_restore_memory(query, params, session_id):
    return await restore_memory(
        db=db,
        embeddings=embeddings,
        archive_id=params.get("archive_id")
    )


async def _handle_get_cleanup_config(query, params, session_id):
    return await get_cleanup_config(
        db=db,
        embeddings=embeddings,
        project_path=params.get("project_path")
    )


async def _handle_set_cleanup_config(query, params, session_id):
    return await set_cleanup_config(
        db=db,
        embeddings=embeddings,
        project_path=params.get("project_path"),
        retention_days=params.get("retention_days"),
        min_relevance_score=params.get("min_relevance_score"),
        keep_high_importance=params.get("keep_high_importance"),
        importance_threshold=params.get("importance_threshold"),
        dedup_enabled=params.get("dedup_enabled"),
        dedup_threshold=params.get("dedup_threshold"),
        archive_before_delete=params.get("archive_before_delete"),
        auto_cleanup_enabled=params.get("auto_cleanup_enabled")
    )


async def _handle_get_cleanup_stats(query, params, session_id):
    return await get_cleanup_stats(db=db, embeddings=embeddings)


async def _handle_purge_expired_archives(query, params, session_id):
    return await purge_expired_archives(db=db, embeddings=embeddings)


# --- Admin Skills (Embedding Model Management) ---

async def _handle_get_embedding_status(query, params, session_id):
    return await get_embedding_status(db=db, embeddings=embeddings)


async def _handle_switch_embedding_model(query, params, session_id):
    return await switch_embedding_model(
        db=db,
        embeddings=embeddings,
        model=params.get("model", "nomic-embed-text"),
        reindex_existing=params.get("reindex_existing", False)
    )


async def _handle_reindex_memories(query, params, session_id):
    return await reindex_memories(
        db=db,
        embeddings=embeddings,
        model=params.get("model"),
        project_path=params.get("project_path"),
        batch_size=params.get("batch_size", 10),
        dry_run=params.get("dry_run", False)
    )


async def _handle_get_reindex_progress(query, params, session_id):
    return await get_reindex_progress(db=db, embeddings=embeddings)


async def _handle_cancel_reindex(query, params, session_id):
    return await cancel_reindex(db=db, embeddings=embeddings)


async def _handle_get_model_info(query, params, session_id):
    return await get_model_info(
        db=db,
        embeddings=embeddings,
        model=params.get("model")
    )


async def _handle_get_system_stats(query, params, session_id):
    return await get_system_stats(db=db, embeddings=embeddings)


# --- MoltBot-Inspired Skills (Human-Readable Transparency) ---

async def _handle_daily_log_append(query, params, session_id):
    from services.daily_log import append_entry
    return await append_entry(
        project_path=params.get("project_path"),
        content=params.get("content", query),
        entry_type=params.get("entry_type", "note"),
        session_id=session_id or params.get("session_id")
    )


async def _handle_daily_log_append_session(query, params, session_id):
    from services.daily_log import append_session_summary
    return await append_session_summary(
        project_path=params.get("project_path"),
        session_id=session_id or params.get("session_id"),
        decisions=params.get("decisions"),
        accomplishments=params.get("accomplishments"),
        notes=params.get("notes"),
        errors_solved=params.get("errors_solved")
    )


async def _handle_daily_log_read(query, params, session_id):
    from services.daily_log import load_recent_logs
    return await load_recent_logs(
        project_path=params.get("project_path"),
        days=params.get("days", 2),
        max_chars=params.get("max_chars", 8000)
    )


async def _handle_daily_log_highlights(query, params, session_id):
    from services.daily_log import get_today_highlights
    return await get_today_highlights(
        project_path=params.get("project_path"),
        max_entries=params.get("max_entries", 10)
    )


async def _handle_daily_log_list(query, params, session_id):
    from services.daily_log import list_logs
    return await list_logs(
        project_path=params.get("project_path"),
        limit=params.get("limit", 30)
    )


async def _handle_sync_memory_md(query, params, session_id):
    from services.memory_md_sync import sync_to_memory_md
    return await sync_to_memory_md(
        db=db,
        project_path=params.get("project_path"),
        min_importance=params.get("min_importance", 7),
        min_pattern_success=params.get("min_pattern_success", 3)
    )


async def _handle_read_memory_md(query, params, session_id):
    from services.memory_md_sync import read_memory_md
    return await read_memory_md(
        project_path=params.get("project_path")
    )


async def _handle_get_memory_md_summary(query, params, session_id):
    from services.memory_md_sync import get_memory_md_summary
    return await get_memory_md_summary(
        project_path=params.get("project_path")
    )


async def _handle_add_memory_md_fact(query, params, session_id):
    from services.memory_md_sync import add_fact
    return await add_fact(
        project_path=params.get("project_path"),
        fact=params.get("fact", query),
        section=params.get("section", "anchors")
    )


async def _handle_check_flush_needed(query, params, session_id):
    from services.compaction_flush import check_flush_needed as _check_flush_needed
    return await _check_flush_needed(
        db=db,
        session_id=session_id or params.get("session_id"),
        event_threshold=params.get("event_threshold", 50),
        time_threshold_minutes=params.get("time_threshold_minutes", 30)
    )


async def _handle_pre_compaction_flush(query, params, session_id):
    from services.compaction_flush import execute_flush
    return await execute_flush(
        db=db,
        project_path=params.get("project_path"),
        session_id=session_id or params.get("session_id")
    )


async def _handle_list_flushes(query, params, session_id):
    from services.compaction_flush import list_flushes as _list_flushes
    return await _list_flushes(
        project_path=params.get("project_path"),
        limit=params.get("limit", 20)
    )


async def _handle_read_flush(query, params, session_id):
    from services.compaction_flush import read_flush as _read_flush
    return await _read_flush(
        project_path=params.get("project_path"),
        filename=params.get("filename")
    )


# --- Outcome Spectrum Skills ---

async def _handle_update_memory_outcome(query, params, session_id):
    result = await db.update_memory_outcome(
        memory_id=params.get("memory_id"),
        outcome_status=params.get("outcome_status"),
        fixed=params.get("fixed"),
        did_not_fix=params.get("did_not_fix"),
        caused=params.get("caused"),
        superseded_by=params.get("superseded_by")
    )
    if result.get("success"):
        await broadcast_event(
            EventTypes.MEMORY_UPDATED,
            {
                "memory_id": params.get("memory_id"),
                "outcome_status": params.get("outcome_status"),
                "action": "outcome_updated"
            }
        )
    return result


async def _handle_supersede_memory(query, params, session_id):
    result = await db.supersede_memory(
        old_memory_id=params.get("old_memory_id"),
        new_memory_id=params.get("new_memory_id"),
        reason=params.get("reason")
    )
    if result.get("success"):
        await broadcast_event(
            EventTypes.MEMORY_UPDATED,
            {
                "old_memory_id": params.get("old_memory_id"),
                "new_memory_id": params.get("new_memory_id"),
                "action": "superseded"
            }
        )
    return result


async def _handle_get_superseding_memory(query, params, session_id):
    superseding = await db.get_superseding_memory(
        memory_id=params.get("memory_id")
    )
    return {
        "success": True,
        "superseded": superseding is not None,
        "superseding_memory": superseding
    }


# --- Curator Agent Skills ---

async def _handle_curator_explore(query, params, session_id):
    from skills.curator import curator_explore as _curator_explore
    return await _curator_explore(
        db=db,
        embeddings=embeddings,
        start_node_id=params.get("start_node_id"),
        max_depth=params.get("max_depth", 3),
        mode=params.get("mode", "bfs"),
        relationship_filter=params.get("relationship_filter")
    )


async def _handle_curator_find_duplicates(query, params, session_id):
    from skills.curator import curator_find_duplicates as _curator_find_duplicates
    return await _curator_find_duplicates(
        db=db,
        embeddings=embeddings,
        project_path=params.get("project_path"),
        similarity_threshold=params.get("similarity_threshold", 0.92),
        limit=params.get("limit", 50)
    )


async def _handle_curator_suggest_links(query, params, session_id):
    from skills.curator import curator_suggest_links as _curator_suggest_links
    return await _curator_suggest_links(
        db=db,
        embeddings=embeddings,
        memory_id=params.get("memory_id"),
        project_path=params.get("project_path"),
        similarity_threshold=params.get("similarity_threshold", 0.7),
        limit=params.get("limit", 20)
    )


async def _handle_curator_merge(query, params, session_id):
    from skills.curator import curator_merge as _curator_merge
    return await _curator_merge(
        db=db,
        embeddings=embeddings,
        keep_id=params.get("keep_id"),
        remove_ids=params.get("remove_ids", []),
        merge_content=params.get("merge_content", False)
    )


async def _handle_curator_get_summary(query, params, session_id):
    from skills.curator import curator_get_summary as _curator_get_summary
    return await _curator_get_summary(
        db=db,
        embeddings=embeddings,
        query=params.get("query", query),
        project_path=params.get("project_path"),
        max_memories=params.get("max_memories", 10),
        include_graph=params.get("include_graph", True)
    )


async def _handle_curator_run_maintenance(query, params, session_id):
    from skills.curator import curator_run_maintenance as _curator_run_maintenance
    return await _curator_run_maintenance(
        db=db,
        embeddings=embeddings,
        project_path=params.get("project_path"),
        tasks=params.get("tasks")
    )


async def _handle_curator_get_report(query, params, session_id):
    from skills.curator import curator_get_report as _curator_get_report
    return await _curator_get_report(
        db=db,
        embeddings=embeddings,
        project_path=params.get("project_path")
    )


async def _handle_curator_get_status(query, params, session_id):
    from skills.curator import curator_get_status as _curator_get_status
    return await _curator_get_status(
        db=db,
        embeddings=embeddings
    )


async def _handle_curator_score_quality(query, params, session_id):
    from skills.curator import curator_score_quality as _curator_score_quality
    return await _curator_score_quality(
        db=db,
        embeddings=embeddings,
        memory_id=params.get("memory_id"),
        project_path=params.get("project_path"),
        limit=params.get("limit", 100)
    )


async def _handle_curator_find_orphans(query, params, session_id):
    from skills.curator import curator_find_orphans as _curator_find_orphans
    return await _curator_find_orphans(
        db=db,
        embeddings=embeddings,
        project_path=params.get("project_path"),
        limit=params.get("limit", 50)
    )


# --- Memory Decay Skills ---

async def _handle_decay_maintenance(query, params, session_id):
    from services.memory_decay import MemoryDecayService
    from config import config
    decay_service = MemoryDecayService(
        db=db,
        archive_threshold=config.DECAY_ARCHIVE_THRESHOLD
    )
    return await decay_service.apply_decay()


async def _handle_decay_stats(query, params, session_id):
    from services.memory_decay import MemoryDecayService
    from config import config
    decay_service = MemoryDecayService(
        db=db,
        archive_threshold=config.DECAY_ARCHIVE_THRESHOLD
    )
    return await decay_service.get_decay_stats()


async def _handle_decay_boost(query, params, session_id):
    from services.memory_decay import MemoryDecayService
    from config import config
    decay_service = MemoryDecayService(
        db=db,
        archive_threshold=config.DECAY_ARCHIVE_THRESHOLD
    )
    memory_id = params.get("memory_id")
    if not memory_id:
        return {"error": "memory_id is required"}
    return await decay_service.boost_on_access(int(memory_id))


# --- Tier 1 Auto-Generation Skill ---

async def _handle_generate_tier1(query, params, session_id):
    from services.claude_md_sync import get_claude_md_sync
    sync_service = get_claude_md_sync(db, embeddings)
    return await sync_service.write_tier1_to_claude_md(
        project_path=params.get("project_path"),
        dry_run=params.get("dry_run", False)
    )


# --- Session Review Skills ---

async def _handle_get_session_memories(query, params, session_id):
    return await get_session_memories(
        db=db,
        session_id=session_id or params.get("session_id"),
        include_patterns=params.get("include_patterns", False),
        limit=params.get("limit", 100)
    )


async def _handle_review_session_memories(query, params, session_id):
    return await review_session_memories(
        db=db,
        session_id=session_id or params.get("session_id"),
        reviews=params.get("reviews", [])
    )


async def _handle_suggest_session_reviews(query, params, session_id):
    return await suggest_session_reviews(
        db=db,
        embeddings=embeddings,
        session_id=session_id or params.get("session_id")
    )


async def _handle_get_recent_sessions(query, params, session_id):
    return await get_recent_sessions(
        db=db,
        project_path=params.get("project_path"),
        limit=params.get("limit", 10)
    )


async def _handle_bulk_review_by_type(query, params, session_id):
    return await bulk_review_by_type(
        db=db,
        session_id=session_id or params.get("session_id"),
        type_decisions=params.get("type_decisions", {})
    )


# --- Cross-Session Awareness Skills ---

async def _handle_session_register(query, params, session_id):
    awareness = get_session_awareness(db)
    return await awareness.register_session(
        session_id=session_id or params.get("session_id"),
        project_path=params.get("project_path", ""),
        goal=params.get("goal"),
        label=params.get("label"),
    )


async def _handle_session_heartbeat(query, params, session_id):
    awareness = get_session_awareness(db)
    return await awareness.heartbeat(
        session_id=session_id or params.get("session_id"),
        project_path=params.get("project_path", ""),
        files_modified=params.get("files_modified"),
        current_goal=params.get("current_goal"),
        key_decisions=params.get("key_decisions"),
        summary=params.get("summary"),
    )


async def _handle_session_deregister(query, params, session_id):
    awareness = get_session_awareness(db)
    return await awareness.deregister_session(
        session_id=session_id or params.get("session_id"),
        project_path=params.get("project_path", ""),
        final_summary=params.get("final_summary"),
    )


async def _handle_get_active_sessions(query, params, session_id):
    awareness = get_session_awareness(db)
    sessions = await db.get_active_sessions(
        project_path=params.get("project_path", ""),
        exclude_session_id=params.get("exclude_session_id"),
    )
    return {"success": True, "sessions": sessions, "count": len(sessions)}


async def _handle_session_activity_feed(query, params, session_id):
    awareness = get_session_awareness(db)
    return await awareness.get_activity_feed(
        project_path=params.get("project_path", ""),
        limit=params.get("limit", 20),
        since=params.get("since"),
        exclude_session_id=params.get("exclude_session_id"),
    )


async def _handle_session_catchup(query, params, session_id):
    awareness = get_session_awareness(db)
    return await awareness.get_catchup(
        session_id=session_id or params.get("session_id"),
        project_path=params.get("project_path", ""),
        since=params.get("since"),
    )


async def _handle_session_conflicts(query, params, session_id):
    awareness = get_session_awareness(db)
    return await awareness.check_conflicts(
        session_id=session_id or params.get("session_id"),
        project_path=params.get("project_path", ""),
    )


async def _handle_session_post_activity(query, params, session_id):
    awareness = get_session_awareness(db)
    return await awareness.post_activity(
        session_id=session_id or params.get("session_id"),
        project_path=params.get("project_path", ""),
        event_type=params.get("event_type", "decision"),
        summary=params.get("summary", ""),
        files=params.get("files"),
    )


async def _handle_session_append_file(query, params, session_id):
    return await db.append_file_modified(
        session_id=session_id or params.get("session_id"),
        file_path=params.get("file_path", ""),
    )


# ============================================================
# SKILL DISPATCH TABLE
# ============================================================
# Maps skill_id strings to their async handler functions.
# All handlers share the signature: (query, params, session_id) -> dict
# ============================================================

SKILL_DISPATCH = {
    # Core Memory
    "store_memory": _handle_store_memory,
    "store_project": _handle_store_project,
    "store_pattern": _handle_store_pattern,
    "retrieve_memory": _handle_retrieve_memory,
    "semantic_search": _handle_semantic_search,
    "search_patterns": _handle_search_patterns,
    "get_project_context": _handle_get_project_context,

    # Session
    "summarize_session": _handle_summarize_session,
    "auto_summarize_session": _handle_auto_summarize_session,
    "get_session_handoff": _handle_get_session_handoff,
    "create_diary_entry": _handle_create_diary_entry,
    "check_session_inactivity": _handle_check_session_inactivity,
    "get_stats": _handle_get_stats,

    # Timeline
    "timeline_log": _handle_timeline_log,
    "timeline_log_batch": _handle_timeline_log_batch,
    "timeline_get": _handle_timeline_get,
    "timeline_search": _handle_timeline_search,
    "timeline_auto_detect": _handle_timeline_auto_detect,
    "timeline_chain": _handle_timeline_chain,

    # State
    "state_get": _handle_state_get,
    "state_update": _handle_state_update,
    "state_init_session": _handle_state_init_session,

    # Checkpoint
    "checkpoint_create": _handle_checkpoint_create,
    "checkpoint_load": _handle_checkpoint_load,
    "checkpoint_list": _handle_checkpoint_list,

    # Grounding (Anti-Hallucination)
    "context_refresh": _handle_context_refresh,
    "check_contradictions": _handle_check_contradictions,
    "verify_entity": _handle_verify_entity,
    "mark_anchor": _handle_mark_anchor,
    "get_unresolved_conflicts": _handle_get_unresolved_conflicts,
    "resolve_conflict": _handle_resolve_conflict,
    "get_anchor_history": _handle_get_anchor_history,
    "auto_resolve_conflicts": _handle_auto_resolve_conflicts,

    # Self-Correcting Confidence
    "memory_worked": _handle_memory_worked,
    "memory_failed": _handle_memory_failed,
    "get_reliability_stats": _handle_get_reliability_stats,
    "get_unreliable_memories": _handle_get_unreliable_memories,
    "reset_memory_reliability": _handle_reset_memory_reliability,

    # CLAUDE.MD Management
    "claude_md_read": _handle_claude_md_read,
    "claude_md_add_section": _handle_claude_md_add_section,
    "claude_md_update_section": _handle_claude_md_update_section,
    "claude_md_add_instruction": _handle_claude_md_add_instruction,
    "claude_md_list_sections": _handle_claude_md_list_sections,
    "claude_md_suggest": _handle_claude_md_suggest,

    # Verification
    "best_of_n_verify": _handle_best_of_n_verify,
    "extract_quotes": _handle_extract_quotes,
    "require_grounding": _handle_require_grounding,

    # Cross-Session Learning
    "run_aggregation": _handle_run_aggregation,
    "get_insights": _handle_get_insights,
    "suggest_improvements": _handle_suggest_improvements,
    "record_insight_feedback": _handle_record_insight_feedback,
    "mark_insight_applied": _handle_mark_insight_applied,
    "get_project_insights": _handle_get_project_insights,

    # Memory Cleanup
    "memory_cleanup": _handle_memory_cleanup,
    "get_archived_memories": _handle_get_archived_memories,
    "restore_memory": _handle_restore_memory,
    "get_cleanup_config": _handle_get_cleanup_config,
    "set_cleanup_config": _handle_set_cleanup_config,
    "get_cleanup_stats": _handle_get_cleanup_stats,
    "purge_expired_archives": _handle_purge_expired_archives,

    # Admin (Embedding Model Management)
    "get_embedding_status": _handle_get_embedding_status,
    "switch_embedding_model": _handle_switch_embedding_model,
    "reindex_memories": _handle_reindex_memories,
    "get_reindex_progress": _handle_get_reindex_progress,
    "cancel_reindex": _handle_cancel_reindex,
    "get_model_info": _handle_get_model_info,
    "get_system_stats": _handle_get_system_stats,

    # MoltBot-Inspired (Human-Readable Transparency)
    "daily_log_append": _handle_daily_log_append,
    "daily_log_append_session": _handle_daily_log_append_session,
    "daily_log_read": _handle_daily_log_read,
    "daily_log_highlights": _handle_daily_log_highlights,
    "daily_log_list": _handle_daily_log_list,
    "sync_memory_md": _handle_sync_memory_md,
    "read_memory_md": _handle_read_memory_md,
    "get_memory_md_summary": _handle_get_memory_md_summary,
    "add_memory_md_fact": _handle_add_memory_md_fact,
    "check_flush_needed": _handle_check_flush_needed,
    "pre_compaction_flush": _handle_pre_compaction_flush,
    "list_flushes": _handle_list_flushes,
    "read_flush": _handle_read_flush,

    # Outcome Spectrum
    "update_memory_outcome": _handle_update_memory_outcome,
    "supersede_memory": _handle_supersede_memory,
    "get_superseding_memory": _handle_get_superseding_memory,

    # Curator Agent
    "curator_explore": _handle_curator_explore,
    "curator_find_duplicates": _handle_curator_find_duplicates,
    "curator_suggest_links": _handle_curator_suggest_links,
    "curator_merge": _handle_curator_merge,
    "curator_get_summary": _handle_curator_get_summary,
    "curator_run_maintenance": _handle_curator_run_maintenance,
    "curator_get_report": _handle_curator_get_report,
    "curator_get_status": _handle_curator_get_status,
    "curator_score_quality": _handle_curator_score_quality,
    "curator_find_orphans": _handle_curator_find_orphans,

    # Memory Decay
    "decay_maintenance": _handle_decay_maintenance,
    "decay_stats": _handle_decay_stats,
    "decay_boost": _handle_decay_boost,

    # Tier 1 Auto-Generation
    "generate_tier1": _handle_generate_tier1,

    # Session Review
    "get_session_memories": _handle_get_session_memories,
    "review_session_memories": _handle_review_session_memories,
    "suggest_session_reviews": _handle_suggest_session_reviews,
    "get_recent_sessions": _handle_get_recent_sessions,
    "bulk_review_by_type": _handle_bulk_review_by_type,

    # Cross-Session Awareness
    "session_register": _handle_session_register,
    "session_heartbeat": _handle_session_heartbeat,
    "session_deregister": _handle_session_deregister,
    "get_active_sessions": _handle_get_active_sessions,
    "session_activity_feed": _handle_session_activity_feed,
    "session_catchup": _handle_session_catchup,
    "session_conflicts": _handle_session_conflicts,
    "session_post_activity": _handle_session_post_activity,
    "session_append_file": _handle_session_append_file,
}


async def execute_skill(
    skill_id: str,
    query: str,
    params: Dict[str, Any],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Execute the specified skill with enhanced context support.

    Dispatches to the appropriate handler via SKILL_DISPATCH lookup table.
    Each handler receives (query, params, session_id) and returns a dict.
    """
    handler = SKILL_DISPATCH.get(skill_id)
    if handler is None:
        raise ValueError(f"Unknown skill: {skill_id}")
    return await handler(query, params, session_id)


# ============= REST API Endpoints =============

@app.get("/api/stats")
async def api_get_stats():
    try:
        stats = await db.get_stats()
    except DatabaseError as e:
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "error_code": "STATS_ERROR",
            "error": f"Failed to get stats: {str(e)}"
        }

    # Add timeline stats
    try:
        timeline_stats = await db.execute_query(
            "SELECT COUNT(*) as count FROM timeline_events"
        )
        stats["total_timeline_events"] = timeline_stats[0]["count"] if timeline_stats else 0
    except (DatabaseError, sqlite3.Error) as e:
        logger.warning(f"Failed to get timeline stats: {e}")
        stats["total_timeline_events"] = 0
        stats["timeline_stats_error"] = str(e)
    except Exception as e:
        logger.warning(f"Unexpected error getting timeline stats: {e}")
        stats["total_timeline_events"] = 0
        stats["timeline_stats_error"] = f"Unexpected error: {str(e)}"

    stats["success"] = True
    stats["operation_metrics"] = metrics.to_dict()
    return stats


@app.get("/api/memories")
async def api_get_memories(
    project_path: Optional[str] = None,
    memory_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get memories with optional filtering by project and type."""
    try:
        # Normalize project_path to match stored paths (handles backslash/forward slash mismatch)
        if project_path:
            project_path = normalize_path(project_path)

        query = "SELECT * FROM memories WHERE 1=1"
        params = []

        if project_path:
            # Use REPLACE to normalize stored paths for comparison
            query += " AND REPLACE(project_path, '\\', '/') = ?"
            params.append(project_path)

        if memory_type and memory_type != "all":
            query += " AND type = ?"
            params.append(memory_type)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        memories = await db.execute_query(query, params)

        # Get total count
        count_query = "SELECT COUNT(*) as count FROM memories WHERE 1=1"
        count_params = []
        if project_path:
            # Use REPLACE to normalize stored paths for comparison
            count_query += " AND REPLACE(project_path, '\\', '/') = ?"
            count_params.append(project_path)
        if memory_type and memory_type != "all":
            count_query += " AND type = ?"
            count_params.append(memory_type)

        count_result = await db.execute_query(count_query, count_params)
        total = count_result[0]["count"] if count_result else 0

        return {
            "success": True,
            "memories": memories,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Failed to get memories: {e}")
        return {"success": False, "error": str(e), "memories": []}


@app.get("/api/patterns")
async def api_get_patterns(
    project_path: Optional[str] = None,
    problem_type: Optional[str] = None,
    limit: int = 50
):
    """Get stored patterns with optional filtering."""
    try:
        query = "SELECT * FROM patterns WHERE 1=1"
        params = []

        if problem_type and problem_type != "all":
            query += " AND problem_type = ?"
            params.append(problem_type)

        query += " ORDER BY success_count DESC, created_at DESC LIMIT ?"
        params.append(limit)

        patterns = await db.execute_query(query, params)

        return {
            "success": True,
            "patterns": patterns or []
        }
    except Exception as e:
        logger.error(f"Failed to get patterns: {e}")
        return {"success": False, "error": str(e), "patterns": []}


@app.get("/api/search")
async def api_search_memories(
    query: str,
    project_path: Optional[str] = None,
    limit: int = 20
):
    """Semantic search across memories."""
    try:
        results = await semantic_search(
            db=db,
            embeddings=embeddings,
            query=query,
            project_path=project_path,
            limit=limit
        )
        return {
            "success": True,
            "results": results.get("results", []),
            "query": query
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"success": False, "error": str(e), "results": []}


@app.get("/api/timeline")
async def api_get_timeline(
    project_path: Optional[str] = None,
    session_id: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 100
):
    """Get timeline events with optional filtering."""
    try:
        # Exclude the embedding column - it's a large binary blob that makes responses huge
        query = """SELECT id, session_id, project_path, event_type, sequence_num,
                   summary, details, parent_event_id, root_event_id, entities,
                   status, outcome, confidence, is_anchor, is_reversible,
                   needs_verification, created_at
                   FROM timeline_events WHERE 1=1"""
        params = []

        if project_path:
            query += " AND project_path = ?"
            params.append(project_path)

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        if event_type and event_type != "all":
            query += " AND event_type = ?"
            params.append(event_type)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        events = await db.execute_query(query, params)

        return {
            "success": True,
            "events": events or [],
            "count": len(events) if events else 0
        }
    except Exception as e:
        logger.error(f"Failed to get timeline: {e}")
        return {"success": False, "error": str(e), "events": []}


@app.get("/dashboard")
async def serve_dashboard():
    """Serve the monitoring dashboard."""
    from fastapi.responses import FileResponse
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    return FileResponse(dashboard_path, media_type="text/html")


# ============= Outcome Spectrum API =============

class OutcomeUpdateRequest(BaseModel):
    """Request body for updating memory outcome."""
    outcome_status: Optional[str] = None
    fixed: Optional[List[str]] = None
    did_not_fix: Optional[List[str]] = None
    caused: Optional[List[str]] = None


class SupersedeRequest(BaseModel):
    """Request body for superseding a memory."""
    new_memory_id: int
    reason: Optional[str] = None


@app.put("/api/memory/{memory_id}/outcome")
async def update_memory_outcome(memory_id: int, request: OutcomeUpdateRequest):
    """Update the outcome status and details for a memory.

    This endpoint allows updating:
    - outcome_status: 'pending', 'success', 'partial', 'failed', 'superseded'
    - fixed: List of what this solution fixed (appends to existing)
    - did_not_fix: List of what remains unfixed (appends to existing)
    - caused: List of side effects (appends to existing)

    Examples:
        # Mark a solution as successful
        PUT /api/memory/123/outcome
        {"outcome_status": "success", "fixed": ["login bug", "session timeout"]}

        # Mark as partial with remaining issues
        PUT /api/memory/123/outcome
        {"outcome_status": "partial", "fixed": ["main bug"], "did_not_fix": ["edge case"]}

        # Mark as failed with side effects
        PUT /api/memory/123/outcome
        {"outcome_status": "failed", "caused": ["broke logout"]}
    """
    try:
        # Validate outcome_status if provided
        valid_statuses = {'pending', 'success', 'partial', 'failed', 'superseded'}
        if request.outcome_status and request.outcome_status not in valid_statuses:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Invalid outcome_status: {request.outcome_status}. Must be one of: {', '.join(valid_statuses)}"
                }
            )

        result = await db.update_memory_outcome(
            memory_id=memory_id,
            outcome_status=request.outcome_status,
            fixed=request.fixed,
            did_not_fix=request.did_not_fix,
            caused=request.caused
        )

        if not result.get("success"):
            return JSONResponse(
                status_code=404,
                content=result
            )

        # Broadcast real-time update
        await broadcast_event(
            EventTypes.MEMORY_UPDATED,
            {
                "memory_id": memory_id,
                "outcome_status": request.outcome_status,
                "action": "outcome_updated"
            }
        )

        return result

    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)}
        )
    except Exception as e:
        logger.error(f"Failed to update memory outcome: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/memory/{memory_id}/supersede")
async def supersede_memory(memory_id: int, request: SupersedeRequest):
    """Mark a memory as superseded by another memory.

    This endpoint:
    1. Sets the old memory's status to 'superseded'
    2. Links it to the new memory via superseded_by
    3. Optionally stores the reason for supersession

    When searching, superseded memories are excluded by default,
    and the superseding memory is shown instead.

    Examples:
        # Supersede memory 123 with memory 456
        POST /api/memory/123/supersede
        {"new_memory_id": 456, "reason": "Found better solution"}
    """
    try:
        result = await db.supersede_memory(
            old_memory_id=memory_id,
            new_memory_id=request.new_memory_id,
            reason=request.reason
        )

        if not result.get("success"):
            return JSONResponse(
                status_code=404,
                content=result
            )

        # Broadcast real-time update
        await broadcast_event(
            EventTypes.MEMORY_UPDATED,
            {
                "old_memory_id": memory_id,
                "new_memory_id": request.new_memory_id,
                "action": "superseded"
            }
        )

        return result

    except Exception as e:
        logger.error(f"Failed to supersede memory: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/api/memory/{memory_id}/superseding")
async def get_superseding_memory(memory_id: int):
    """Get the memory that supersedes the given memory.

    If the memory has been superseded, this follows the chain
    to find the latest active (non-superseded) memory.

    Returns None if the memory has not been superseded.
    """
    try:
        result = await db.get_superseding_memory(memory_id)

        if result:
            return {
                "success": True,
                "superseded": True,
                "superseding_memory": result,
                "message": f"Memory {memory_id} has been superseded by memory {result['id']}"
            }
        else:
            return {
                "success": True,
                "superseded": False,
                "superseding_memory": None,
                "message": f"Memory {memory_id} has not been superseded"
            }

    except Exception as e:
        logger.error(f"Failed to get superseding memory: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


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
    except DatabaseError as e:
        logger.error(f"Database error getting projects: {e}")
        return {
            'success': False,
            'error_code': e.error_code,
            'error': str(e),
            'projects': []
        }
    except Exception as e:
        logger.error(f"Unexpected error getting projects: {e}")
        return {
            'success': False,
            'error_code': 'PROJECTS_FETCH_ERROR',
            'error': str(e),
            'projects': []
        }


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
    except DatabaseError as e:
        logger.error(f"Database error getting sessions for {project_path}: {e}")
        return {
            'success': False,
            'error_code': e.error_code,
            'error': str(e),
            'sessions': []
        }
    except Exception as e:
        logger.error(f"Unexpected error getting sessions for {project_path}: {e}")
        return {
            'success': False,
            'error_code': 'SESSIONS_FETCH_ERROR',
            'error': str(e),
            'sessions': []
        }


# ============= Session Review API =============

class SessionReviewRequest(BaseModel):
    """Request body for reviewing session memories."""
    reviews: List[Dict[str, Any]]


class BulkReviewRequest(BaseModel):
    """Request body for bulk review by type."""
    type_decisions: Dict[str, str]


@app.get("/api/session/{session_id}/memories")
async def api_get_session_memories(
    session_id: str,
    include_patterns: bool = False,
    limit: int = 100
):
    """Get all memories created in a specific session for review.

    Returns memories with their current confidence and outcome status,
    along with a summary of memory types and statistics.

    Use this endpoint before presenting a session review UI.
    """
    try:
        result = await get_session_memories(
            db=db,
            session_id=session_id,
            include_patterns=include_patterns,
            limit=limit
        )
        return result
    except Exception as e:
        logger.error(f"Failed to get session memories: {e}")
        return {
            "success": False,
            "error": str(e),
            "memories": []
        }


@app.post("/api/session/{session_id}/review")
async def api_review_session_memories(session_id: str, request: SessionReviewRequest):
    """Submit review decisions for session memories.

    Each review in the list should contain:
    - memory_id: int - The memory to review
    - decision: str - 'keep', 'discard', or 'partial'
    - feedback: str (optional) - User feedback

    Decision effects:
    - keep: Sets confidence to 0.9, outcome_status to 'success'
    - partial: Sets confidence to 0.5, outcome_status to 'partial'
    - discard: Sets confidence to 0.1, outcome_status to 'failed'

    Example:
        POST /api/session/sess-123/review
        {
            "reviews": [
                {"memory_id": 1, "decision": "keep"},
                {"memory_id": 2, "decision": "discard", "feedback": "Not accurate"},
                {"memory_id": 3, "decision": "partial"}
            ]
        }
    """
    try:
        result = await review_session_memories(
            db=db,
            session_id=session_id,
            reviews=request.reviews
        )

        # Broadcast update if any reviews were processed
        if result.get("processed", 0) > 0:
            await broadcast_event(
                EventTypes.MEMORY_UPDATED,
                {
                    "session_id": session_id,
                    "action": "session_review",
                    "kept": result.get("kept", 0),
                    "discarded": result.get("discarded", 0),
                    "partial": result.get("partial", 0)
                }
            )

        return result
    except Exception as e:
        logger.error(f"Failed to review session memories: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/session/{session_id}/suggestions")
async def api_get_session_suggestions(session_id: str):
    """Get AI-suggested review decisions for session memories.

    Analyzes each memory and suggests whether to keep, discard,
    or mark as partial based on:
    - Memory type (decisions, errors more valuable)
    - Importance score
    - Existing outcome status

    Use this to pre-populate a review form.
    """
    try:
        result = await suggest_session_reviews(
            db=db,
            embeddings=embeddings,
            session_id=session_id
        )
        return result
    except Exception as e:
        logger.error(f"Failed to get session suggestions: {e}")
        return {
            "success": False,
            "error": str(e),
            "suggestions": []
        }


@app.get("/api/sessions/recent")
async def api_get_recent_sessions(
    project_path: Optional[str] = None,
    limit: int = 10
):
    """Get recent sessions with memory counts for review selection.

    Returns sessions ordered by most recent activity, with:
    - Memory count
    - Success/failed/pending counts
    - Average confidence

    Use this to show a list of sessions available for review.
    """
    try:
        result = await get_recent_sessions(
            db=db,
            project_path=project_path,
            limit=limit
        )
        return result
    except Exception as e:
        logger.error(f"Failed to get recent sessions: {e}")
        return {
            "success": False,
            "error": str(e),
            "sessions": []
        }


@app.post("/api/session/{session_id}/bulk-review")
async def api_bulk_review_by_type(session_id: str, request: BulkReviewRequest):
    """Apply review decisions to all memories of specific types.

    Useful for quickly processing sessions:
    - Keep all 'decision' and 'error' memories
    - Discard all 'chunk' memories

    Example:
        POST /api/session/sess-123/bulk-review
        {
            "type_decisions": {
                "decision": "keep",
                "error": "keep",
                "chunk": "discard",
                "code": "partial"
            }
        }
    """
    try:
        result = await bulk_review_by_type(
            db=db,
            session_id=session_id,
            type_decisions=request.type_decisions
        )

        # Broadcast update if any reviews were processed
        if result.get("processed", 0) > 0:
            await broadcast_event(
                EventTypes.MEMORY_UPDATED,
                {
                    "session_id": session_id,
                    "action": "bulk_review",
                    "kept": result.get("kept", 0),
                    "discarded": result.get("discarded", 0),
                    "partial": result.get("partial", 0)
                }
            )

        return result
    except Exception as e:
        logger.error(f"Failed to bulk review session: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/health")
async def health_check():
    """Comprehensive health check with component status.

    Returns status of:
    - Agent (always healthy if responding)
    - Database (SQLite connection)
    - Ollama (embedding service)
    """
    # Check database health
    db_healthy = False
    db_error = None
    try:
        # Simple query to verify DB connection
        result = await db.execute_query("SELECT 1 as test")
        db_healthy = result is not None
    except Exception as e:
        db_error = str(e)

    # Check Ollama health
    ollama_health = await embeddings.check_health()

    # Overall status
    all_healthy = db_healthy and ollama_health.get("healthy", False)
    degraded = db_healthy and not ollama_health.get("healthy", False)

    status = "healthy" if all_healthy else ("degraded" if degraded else "unhealthy")

    return {
        "status": status,
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "agent": {
                "healthy": True,
                "version": "2.0.0"
            },
            "database": {
                "healthy": db_healthy,
                "error": db_error
            },
            "ollama": ollama_health
        },
        "degraded_mode": embeddings.is_degraded(),
        "capabilities": {
            "semantic_search": ollama_health.get("healthy", False),
            "keyword_search": db_healthy,  # Fallback always available if DB healthy
            "memory_storage": db_healthy and ollama_health.get("healthy", False),
            "memory_storage_degraded": db_healthy  # Can store without embeddings
        }
    }


@app.get("/ready")
async def readiness_check():
    """Readiness probe for startup checks.

    Returns 200 only when the service is fully ready to accept requests.
    Used by orchestrators to know when to route traffic.
    """
    # Check database is connected
    try:
        result = await db.execute_query("SELECT 1 as test")
        if not result:
            return JSONResponse(
                status_code=503,
                content={"ready": False, "reason": "Database not responding"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": f"Database error: {str(e)}"}
        )

    # Note: We don't require Ollama for readiness - service can run in degraded mode
    return {
        "ready": True,
        "timestamp": datetime.now().isoformat(),
        "degraded_mode": embeddings.is_degraded()
    }


@app.get("/health/live")
async def liveness_check():
    """Simple liveness probe.

    Returns 200 if the process is alive. Used for basic health monitoring.
    """
    return {"alive": True, "timestamp": datetime.now().isoformat()}


@app.post("/health/retry")
async def retry_connection():
    """Force retry connection to Ollama.

    Clears health check cache and attempts to reconnect.
    Used by dashboard to manually trigger recovery.
    """
    # Force health check with fresh attempt
    health = await embeddings.check_health(force=True)

    # If still unhealthy, try to ping Ollama directly
    if not health.get("healthy"):
        try:
            import subprocess
            import platform

            # Try to check if Ollama is running
            if platform.system() == "Windows":
                # On Windows, try to start Ollama if not running
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq ollama.exe"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                ollama_running = "ollama.exe" in result.stdout

                if not ollama_running:
                    # Try to start Ollama
                    subprocess.Popen(
                        ["ollama", "serve"],
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    # Wait a moment for startup
                    import asyncio
                    await asyncio.sleep(2)
                    # Retry health check
                    health = await embeddings.check_health(force=True)
        except Exception as e:
            logger.warning(f"Could not auto-start Ollama: {e}")

    return {
        "success": health.get("healthy", False),
        "health": health,
        "message": "Healthy" if health.get("healthy") else "Still degraded - check if Ollama is running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/index-stats")
async def get_index_stats():
    """Get FAISS vector index statistics.

    Returns information about:
    - Whether FAISS is available
    - Index sizes for memories, patterns, and timeline
    - Search and add counts
    """
    try:
        stats = db.get_index_stats()
        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.post("/api/rebuild-indexes")
async def rebuild_indexes():
    """Rebuild all FAISS vector indexes from the database.

    Use this if indexes get out of sync with the database.
    """
    try:
        await db._rebuild_memories_index()
        await db._rebuild_patterns_index()
        await db._rebuild_timeline_index()

        return {
            "success": True,
            "message": "Indexes rebuilt successfully",
            "stats": db.get_index_stats(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# ============= Retry Queue API =============

@app.get("/api/queue/stats")
async def get_queue_stats():
    """Get retry queue statistics."""
    if retry_queue is None:
        return {"success": False, "error": "Queue not initialized"}

    return {
        "success": True,
        "stats": retry_queue.get_stats(),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/queue/pending")
async def get_pending_requests(limit: int = 20):
    """Get pending requests in the queue."""
    if retry_queue is None:
        return {"success": False, "error": "Queue not initialized"}

    items = retry_queue.get_pending(limit=limit)
    return {
        "success": True,
        "items": items,
        "count": len(items),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/queue/dead-letters")
async def get_dead_letters(limit: int = 50):
    """Get items from the dead letter queue."""
    if retry_queue is None:
        return {"success": False, "error": "Queue not initialized"}

    items = retry_queue.get_dead_letters(limit=limit)
    return {
        "success": True,
        "items": items,
        "count": len(items),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/queue/retry-dead-letter/{item_id}")
async def retry_dead_letter(item_id: int):
    """Move a dead letter back to the pending queue for retry."""
    if retry_queue is None:
        return {"success": False, "error": "Queue not initialized"}

    success = retry_queue.retry_dead_letter(item_id)
    return {
        "success": success,
        "message": "Item requeued" if success else "Item not found",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/queue/enqueue")
async def enqueue_request(request: Request):
    """Manually enqueue a request for retry.

    Useful for testing or manual recovery.
    """
    if retry_queue is None:
        return {"success": False, "error": "Queue not initialized"}

    try:
        body = await request.json()
        item_id = retry_queue.enqueue(
            endpoint=body.get("endpoint"),
            payload=body.get("payload", {}),
            method=body.get("method", "POST"),
            headers=body.get("headers")
        )
        return {
            "success": True,
            "item_id": item_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# ============= Authentication API =============

@app.get("/api/auth/stats")
async def get_auth_stats():
    """Get authentication service statistics."""
    return {
        "success": True,
        "stats": auth_service.get_stats(),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/auth/keys")
async def list_api_keys():
    """List all API keys (without the actual key values)."""
    return {
        "success": True,
        "keys": auth_service.list_keys(),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/auth/keys")
async def create_api_key(request: Request):
    """Generate a new API key.

    IMPORTANT: The key is only returned once! Store it securely.
    """
    try:
        body = await request.json()
        name = body.get("name")
        description = body.get("description", "")
        rate_limit = body.get("rate_limit", 100)

        if not name:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Name is required"}
            )

        key = auth_service.generate_key(name, description, rate_limit)
        return {
            "success": True,
            "key": key,
            "name": name,
            "warning": "Store this key securely - it will not be shown again!",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/auth/keys/{name}/revoke")
async def revoke_api_key(name: str):
    """Revoke an API key by name."""
    success = auth_service.revoke_key(name)
    return {
        "success": success,
        "message": f"Key '{name}' revoked" if success else f"Key '{name}' not found",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/auth/keys/{name}/rotate")
async def rotate_api_key(name: str):
    """Rotate an API key (revoke old, generate new with same name).

    IMPORTANT: The new key is only returned once! Store it securely.
    """
    new_key = auth_service.rotate_key(name)
    if new_key:
        return {
            "success": True,
            "key": new_key,
            "name": name,
            "warning": "Store this key securely - it will not be shown again!",
            "timestamp": datetime.now().isoformat()
        }
    return {
        "success": False,
        "error": f"Key '{name}' not found or already revoked",
        "timestamp": datetime.now().isoformat()
    }


# ============= Anchor Conflict Resolution API =============

@app.get("/api/conflicts")
async def api_get_conflicts(
    session_id: Optional[str] = None,
    project_path: Optional[str] = None,
    limit: int = 20
):
    """Get unresolved anchor conflicts."""
    return await get_unresolved_conflicts(
        db=db,
        session_id=session_id,
        project_path=project_path,
        limit=limit
    )


@app.post("/api/conflicts/{conflict_id}/resolve")
async def api_resolve_conflict(conflict_id: int, request: Request):
    """Resolve an anchor conflict."""
    try:
        body = await request.json()
        return await resolve_conflict(
            db=db,
            embeddings=embeddings,
            conflict_id=conflict_id,
            resolution=body.get("resolution"),
            keep_anchor_id=body.get("keep_anchor_id"),
            resolved_by=body.get("resolved_by", "api")
        )
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error resolving conflict {conflict_id}: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error resolving conflict {conflict_id}: {e}")
        return {
            "success": False,
            "error_code": "CONFLICT_RESOLVE_ERROR",
            "error": str(e)
        }


@app.post("/api/conflicts/auto-resolve")
async def api_auto_resolve(session_id: Optional[str] = None):
    """Attempt automatic resolution of simple conflicts."""
    return await auto_resolve_conflicts(
        db=db,
        embeddings=embeddings,
        session_id=session_id
    )


@app.get("/api/anchors/history")
async def api_anchor_history(
    anchor_id: Optional[int] = None,
    session_id: Optional[str] = None,
    limit: int = 50
):
    """Get anchor history for tracking fact evolution."""
    return await get_anchor_history(
        db=db,
        anchor_id=anchor_id,
        session_id=session_id,
        limit=limit
    )


# ============= Memory Cleanup API =============

@app.post("/api/cleanup")
async def api_run_cleanup(
    project_path: Optional[str] = None,
    dry_run: bool = True
):
    """Run memory cleanup (dry run by default)."""
    return await memory_cleanup(
        db=db,
        embeddings=embeddings,
        project_path=project_path,
        dry_run=dry_run
    )


@app.get("/api/cleanup/stats")
async def api_cleanup_stats():
    """Get cleanup statistics."""
    return await get_cleanup_stats(db=db, embeddings=embeddings)


@app.get("/api/cleanup/config")
async def api_get_cleanup_config(project_path: Optional[str] = None):
    """Get cleanup configuration."""
    return await get_cleanup_config(db=db, embeddings=embeddings, project_path=project_path)


@app.post("/api/cleanup/config")
async def api_set_cleanup_config(request: Request):
    """Update cleanup configuration."""
    try:
        body = await request.json()
        return await set_cleanup_config(
            db=db,
            embeddings=embeddings,
            project_path=body.get("project_path"),
            retention_days=body.get("retention_days"),
            min_relevance_score=body.get("min_relevance_score"),
            keep_high_importance=body.get("keep_high_importance"),
            importance_threshold=body.get("importance_threshold"),
            dedup_enabled=body.get("dedup_enabled"),
            dedup_threshold=body.get("dedup_threshold"),
            archive_before_delete=body.get("archive_before_delete"),
            auto_cleanup_enabled=body.get("auto_cleanup_enabled")
        )
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error setting cleanup config: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error setting cleanup config: {e}")
        return {
            "success": False,
            "error_code": "CLEANUP_CONFIG_ERROR",
            "error": str(e)
        }


@app.get("/api/archives")
async def api_get_archives(
    project_path: Optional[str] = None,
    reason: Optional[str] = None,
    limit: int = 50
):
    """Get archived memories."""
    return await get_archived_memories(
        db=db,
        embeddings=embeddings,
        project_path=project_path,
        reason=reason,
        limit=limit
    )


@app.post("/api/archives/{archive_id}/restore")
async def api_restore_memory(archive_id: int):
    """Restore an archived memory."""
    return await restore_memory(db=db, embeddings=embeddings, archive_id=archive_id)


@app.post("/api/archives/purge")
async def api_purge_archives():
    """Permanently delete expired archives."""
    return await purge_expired_archives(db=db, embeddings=embeddings)


# ============= Admin API (Embedding Model Management) =============

@app.get("/api/admin/embeddings")
async def api_embedding_status():
    """Get embedding service status and available models."""
    return await get_embedding_status(db=db, embeddings=embeddings)


@app.post("/api/admin/embeddings/switch")
async def api_switch_model(request: Request):
    """Switch embedding model."""
    try:
        body = await request.json()
        return await switch_embedding_model(
            db=db,
            embeddings=embeddings,
            model=body.get("model", "nomic-embed-text"),
            reindex_existing=body.get("reindex_existing", False)
        )
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error switching embedding model: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error switching embedding model: {e}")
        return {
            "success": False,
            "error_code": "MODEL_SWITCH_ERROR",
            "error": str(e)
        }


@app.get("/api/admin/embeddings/models/{model}")
async def api_model_info(model: str):
    """Get detailed info about a specific embedding model."""
    return await get_model_info(db=db, embeddings=embeddings, model=model)


@app.post("/api/admin/reindex")
async def api_start_reindex(request: Request):
    """Start background reindexing of memories."""
    try:
        body = await request.json()
        return await reindex_memories(
            db=db,
            embeddings=embeddings,
            model=body.get("model"),
            project_path=body.get("project_path"),
            batch_size=body.get("batch_size", 10),
            dry_run=body.get("dry_run", False)
        )
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error during reindex: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error during reindex: {e}")
        return {
            "success": False,
            "error_code": "REINDEX_ERROR",
            "error": str(e)
        }


@app.get("/api/admin/reindex/progress")
async def api_reindex_progress():
    """Get reindex progress."""
    return await get_reindex_progress(db=db, embeddings=embeddings)


@app.post("/api/admin/reindex/cancel")
async def api_cancel_reindex():
    """Cancel running reindex operation."""
    return await cancel_reindex(db=db, embeddings=embeddings)


@app.get("/api/admin/stats")
async def api_system_stats():
    """Get comprehensive system statistics."""
    return await get_system_stats(db=db, embeddings=embeddings)


# ============= Auto-Injection API =============

@app.post("/api/inject")
async def api_auto_inject(request: Request):
    """Get relevant context for current task (auto-injection).

    Send current query/task and get back relevant memories and patterns.
    """
    try:
        body = await request.json()
        injector = get_auto_injector(db, embeddings)

        context = await injector.get_relevant_context(
            current_query=body.get("query", ""),
            project_path=body.get("project_path"),
            task_type=body.get("task_type"),
            max_results=body.get("max_results", 3)
        )

        # Format for easy consumption
        formatted = injector.format_injection(context)

        return {
            "success": True,
            "context": context,
            "formatted": formatted
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error during auto-inject: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error during auto-inject: {e}")
        return {
            "success": False,
            "error_code": "AUTO_INJECT_ERROR",
            "error": str(e)
        }


@app.post("/api/inject/reset")
async def api_reset_injection():
    """Reset injection tracking for new session."""
    injector = get_auto_injector(db, embeddings)
    injector.reset_session()
    return {"success": True, "message": "Injection context reset"}


# ============= Confidence Scoring API =============

@app.get("/api/memory/{memory_id}/confidence")
async def api_get_confidence(memory_id: int):
    """Get confidence score for a memory."""
    confidence_svc = get_confidence_service(db, embeddings)
    return await confidence_svc.calculate_confidence(memory_id)


@app.post("/api/memory/{memory_id}/confidence")
async def api_update_confidence(memory_id: int, request: Request):
    """Update the stored confidence score for a memory.

    Request body:
        confidence: float (0.0 to 1.0)

    Confidence scoring rules:
        - New memories start at 0.5
        - Range: 0.0 (unreliable) to 1.0 (proven)
        - Higher confidence = higher rank in search results
    """
    try:
        body = await request.json()
        confidence = body.get("confidence")

        if confidence is None:
            return {
                "success": False,
                "error_code": "MISSING_CONFIDENCE",
                "error": "confidence field is required in request body"
            }

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return {
                "success": False,
                "error_code": "INVALID_CONFIDENCE",
                "error": "confidence must be a number between 0.0 and 1.0"
            }

        if confidence < 0.0 or confidence > 1.0:
            return {
                "success": False,
                "error_code": "CONFIDENCE_OUT_OF_RANGE",
                "error": f"confidence must be between 0.0 and 1.0, got {confidence}"
            }

        result = await db.update_memory_confidence(memory_id, confidence)
        return result

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error updating confidence for memory {memory_id}: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error updating confidence for memory {memory_id}: {e}")
        return {
            "success": False,
            "error_code": "CONFIDENCE_UPDATE_ERROR",
            "error": str(e)
        }


@app.post("/api/memory/{memory_id}/verify")
async def api_verify_memory(memory_id: int, request: Request):
    """Mark a memory as verified or unverified."""
    try:
        body = await request.json()
        confidence_svc = get_confidence_service(db, embeddings)
        return await confidence_svc.verify_memory(
            memory_id=memory_id,
            verified=body.get("verified", True),
            verified_by=body.get("verified_by", "user")
        )
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error verifying memory {memory_id}: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error verifying memory {memory_id}: {e}")
        return {
            "success": False,
            "error_code": "MEMORY_VERIFY_ERROR",
            "error": str(e)
        }


@app.post("/api/memory/{memory_id}/outdated")
async def api_mark_outdated(memory_id: int, request: Request):
    """Mark a memory as outdated."""
    try:
        body = await request.json()
        confidence_svc = get_confidence_service(db, embeddings)
        return await confidence_svc.mark_outdated(
            memory_id=memory_id,
            reason=body.get("reason", "manually marked")
        )
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error marking memory {memory_id} outdated: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error marking memory {memory_id} outdated: {e}")
        return {
            "success": False,
            "error_code": "MEMORY_OUTDATED_ERROR",
            "error": str(e)
        }


@app.get("/api/memory/low-confidence")
async def api_low_confidence(
    project_path: Optional[str] = None,
    threshold: float = 0.5,
    limit: int = 20
):
    """Get memories with low confidence that need verification."""
    confidence_svc = get_confidence_service(db, embeddings)
    return await confidence_svc.get_low_confidence_memories(
        project_path=project_path,
        threshold=threshold,
        limit=limit
    )


# ============= Self-Correcting Confidence API =============

@app.post("/api/memory/{memory_id}/worked")
async def api_memory_worked(memory_id: int, request: Request):
    """Report that a memory's solution worked.

    This is the core feedback mechanism for self-correcting confidence.
    When a solution works:
    - Confidence increases by 0.15 (max 1.0)
    - Consecutive failure count resets to 0
    - times_worked counter increments

    Request body (optional):
        context: dict with optional details about the usage context

    Returns:
        Updated confidence, failure_count, times_worked, times_failed, reliability status
    """
    try:
        from skills.confidence_tracker import report_solution_outcome

        body = {}
        try:
            body = await request.json()
        except json.JSONDecodeError:
            pass  # Empty body is fine

        context = body.get("context")

        result = await report_solution_outcome(db, memory_id, worked=True, context=context)

        if not result.get("success"):
            return JSONResponse(
                status_code=404 if result.get("error_code") == "MEMORY_NOT_FOUND" else 400,
                content=result
            )

        # Broadcast real-time update
        await broadcast_event(
            EventTypes.MEMORY_UPDATED,
            {
                "memory_id": memory_id,
                "action": "worked",
                "new_confidence": result.get("new_confidence"),
                "reliability": result.get("reliability")
            }
        )

        return result

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error reporting memory {memory_id} worked: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": e.error_code,
                "error": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error reporting memory {memory_id} worked: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "MEMORY_WORKED_ERROR",
                "error": str(e)
            }
        )


@app.post("/api/memory/{memory_id}/failed")
async def api_memory_failed(memory_id: int, request: Request):
    """Report that a memory's solution failed.

    This is the core feedback mechanism for self-correcting confidence.
    When a solution fails:
    - Confidence decreases by 0.2 (min 0.0)
    - Consecutive failure count increments
    - times_failed counter increments
    - After 3 consecutive failures: memory marked as unreliable
      (outcome_status='failed', confidence=0.1)

    Request body (optional):
        context: dict with optional details about the failure context

    Returns:
        Updated confidence, failure_count, times_worked, times_failed, reliability status
        is_unreliable: true if marked unreliable after this failure
    """
    try:
        from skills.confidence_tracker import report_solution_outcome

        body = {}
        try:
            body = await request.json()
        except json.JSONDecodeError:
            pass  # Empty body is fine

        context = body.get("context")

        result = await report_solution_outcome(db, memory_id, worked=False, context=context)

        if not result.get("success"):
            return JSONResponse(
                status_code=404 if result.get("error_code") == "MEMORY_NOT_FOUND" else 400,
                content=result
            )

        # Broadcast real-time update
        await broadcast_event(
            EventTypes.MEMORY_UPDATED,
            {
                "memory_id": memory_id,
                "action": "failed",
                "new_confidence": result.get("new_confidence"),
                "reliability": result.get("reliability"),
                "is_unreliable": result.get("is_unreliable")
            }
        )

        return result

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error reporting memory {memory_id} failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": e.error_code,
                "error": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error reporting memory {memory_id} failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "MEMORY_FAILED_ERROR",
                "error": str(e)
            }
        )


@app.get("/api/memory/{memory_id}/reliability")
async def api_memory_reliability(memory_id: int):
    """Get detailed reliability statistics for a memory.

    Returns:
        - confidence: Current confidence score (0.0-1.0)
        - times_worked: Total times solution worked
        - times_failed: Total times solution failed
        - success_rate: Ratio of worked to total uses
        - failure_count: Consecutive failures (resets on success)
        - reliability: Classification ('proven', 'high', 'moderate', 'low', 'unreliable', 'untested')
        - outcome_history: List of last 20 outcome reports
        - interpretation: Human-readable explanation
    """
    try:
        from skills.confidence_tracker import get_reliability_stats

        result = await get_reliability_stats(db, memory_id)

        if not result.get("success"):
            return JSONResponse(
                status_code=404 if result.get("error_code") == "MEMORY_NOT_FOUND" else 400,
                content=result
            )

        return result

    except DatabaseError as e:
        logger.error(f"Database error getting reliability for memory {memory_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": e.error_code,
                "error": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error getting reliability for memory {memory_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "RELIABILITY_GET_ERROR",
                "error": str(e)
            }
        )


@app.get("/api/memories/unreliable")
async def api_unreliable_memories(
    project_path: Optional[str] = None,
    limit: int = 50
):
    """Get all memories marked as unreliable (failure_count >= 3).

    These memories are excluded from search by default.
    """
    try:
        from skills.confidence_tracker import get_unreliable_memories

        return await get_unreliable_memories(db, project_path=project_path, limit=limit)

    except DatabaseError as e:
        logger.error(f"Database error getting unreliable memories: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e),
            "unreliable_memories": []
        }
    except Exception as e:
        logger.error(f"Unexpected error getting unreliable memories: {e}")
        return {
            "success": False,
            "error_code": "UNRELIABLE_GET_ERROR",
            "error": str(e),
            "unreliable_memories": []
        }


@app.post("/api/memory/{memory_id}/reset-reliability")
async def api_reset_reliability(memory_id: int, request: Request):
    """Reset a memory's reliability stats (admin function).

    Useful when a memory has been fixed or updated and should be
    given a fresh chance.

    Request body (optional):
        confidence: Starting confidence (default 0.5)
    """
    try:
        from skills.confidence_tracker import reset_memory_reliability

        body = {}
        try:
            body = await request.json()
        except json.JSONDecodeError:
            pass  # Empty body is fine

        new_confidence = body.get("confidence", 0.5)

        result = await reset_memory_reliability(db, memory_id, new_confidence)

        if not result.get("success"):
            return JSONResponse(
                status_code=404 if result.get("error_code") == "MEMORY_NOT_FOUND" else 400,
                content=result
            )

        # Broadcast real-time update
        await broadcast_event(
            EventTypes.MEMORY_UPDATED,
            {
                "memory_id": memory_id,
                "action": "reliability_reset",
                "new_confidence": result.get("new_confidence")
            }
        )

        return result

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error resetting reliability for memory {memory_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": e.error_code,
                "error": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error resetting reliability for memory {memory_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "RELIABILITY_RESET_ERROR",
                "error": str(e)
            }
        )


# ============= Context Tagging API =============

@app.post("/api/memory/{memory_id}/context")
async def api_add_context(memory_id: int, request: Request):
    """Add context result (success or failure) to a memory.

    Request body:
        context: dict with project_type, tech_stack, environment, file_patterns
        result: "success" or "failure"
        failure_reason: optional string explaining why it failed

    This makes the memory system context-aware - same solution may work in React but fail in Vue.
    """
    try:
        body = await request.json()

        context = body.get("context")
        result = body.get("result", "success")
        failure_reason = body.get("failure_reason")

        if not context:
            return {
                "success": False,
                "error_code": "MISSING_CONTEXT",
                "error": "context field is required in request body"
            }

        if result not in ("success", "failure"):
            return {
                "success": False,
                "error_code": "INVALID_RESULT",
                "error": "result must be 'success' or 'failure'"
            }

        from skills.context import add_context_success, add_context_failure

        if result == "success":
            return await add_context_success(db, memory_id, context)
        else:
            return await add_context_failure(db, memory_id, context, failure_reason)

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error adding context to memory {memory_id}: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error adding context to memory {memory_id}: {e}")
        return {
            "success": False,
            "error_code": "CONTEXT_ADD_ERROR",
            "error": str(e)
        }


@app.get("/api/memory/{memory_id}/context")
async def api_get_context(memory_id: int, current_project: Optional[str] = None):
    """Get context data for a memory and optionally calculate relevance score.

    Query params:
        current_project: Optional project path to calculate context match score

    Returns:
        worked_in: contexts where solution worked
        failed_in: contexts where solution failed
        context_confidence: context-specific confidence
        context_score: if current_project provided, relevance score for that context
    """
    try:
        from skills.context import get_memory_contexts, get_context_score, detect_project_context

        result = await get_memory_contexts(db, memory_id)

        if not result.get("success"):
            return result

        # If current_project provided, calculate context score
        if current_project:
            current_context = detect_project_context(current_project)
            if current_context:
                score_result = await get_context_score(db, memory_id, current_context)
                result["context_score"] = score_result
                result["current_context"] = current_context

        return result

    except DatabaseError as e:
        logger.error(f"Database error getting context for memory {memory_id}: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error getting context for memory {memory_id}: {e}")
        return {
            "success": False,
            "error_code": "CONTEXT_GET_ERROR",
            "error": str(e)
        }


@app.get("/api/context/detect")
async def api_detect_context(project_path: str):
    """Detect project context from a path.

    Query params:
        project_path: Path to the project directory

    Returns:
        project_type: detected project type (react, python, wordpress, etc.)
        tech_stack: list of detected technologies
        environment: detected environment (dev, prod, test)
        file_patterns: list of file patterns found in project
    """
    try:
        from skills.context import detect_project_context

        context = detect_project_context(project_path)

        return {
            "success": True,
            "project_path": project_path,
            "context": context
        }

    except Exception as e:
        logger.error(f"Error detecting context for {project_path}: {e}")
        return {
            "success": False,
            "error_code": "CONTEXT_DETECT_ERROR",
            "error": str(e)
        }


# ============= CLAUDE.md Sync API =============

@app.get("/api/claude-md/suggestions")
async def api_claude_md_suggestions(project_path: Optional[str] = None):
    """Get suggestions for CLAUDE.md updates."""
    sync_svc = get_claude_md_sync(db, embeddings)
    return await sync_svc.suggest_updates(project_path)


@app.post("/api/claude-md/sync")
async def api_claude_md_sync(request: Request):
    """Sync learnings to CLAUDE.md."""
    try:
        body = await request.json()
        sync_svc = get_claude_md_sync(db, embeddings)
        return await sync_svc.sync_to_claude_md(
            project_path=body.get("project_path"),
            dry_run=body.get("dry_run", True)
        )
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error syncing CLAUDE.md: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error syncing CLAUDE.md: {e}")
        return {
            "success": False,
            "error_code": "CLAUDE_MD_SYNC_ERROR",
            "error": str(e)
        }


# ============= Natural Language Interface =============

@app.post("/api/memory/natural")
async def api_natural_language(request: Request):
    """Process natural language memory commands.

    Examples:
    - "remember this: always use async/await for DB calls"
    - "what did I learn about authentication?"
    - "show me past errors"
    - "memory stats"
    """
    try:
        body = await request.json()
        result = await process_natural_command(
            db=db,
            embeddings=embeddings,
            command=body.get("command", ""),
            project_path=body.get("project_path"),
            session_id=body.get("session_id")
        )
        return result
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error processing natural command: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error processing natural command: {e}")
        return {
            "success": False,
            "error_code": "NATURAL_COMMAND_ERROR",
            "error": str(e)
        }


# ============= Session Summarization API =============

@app.post("/api/sessions/{session_id}/auto-summarize")
async def api_auto_summarize(session_id: str, project_path: Optional[str] = None):
    """Auto-summarize a session based on its timeline."""
    return await auto_summarize_session(
        db=db,
        embeddings=embeddings,
        session_id=session_id,
        project_path=project_path
    )


@app.get("/api/sessions/handoff")
async def api_session_handoff(
    project_path: Optional[str] = None,
    include_last_n: int = 3
):
    """Get context handoff from previous sessions."""
    return await get_session_handoff(
        db=db,
        embeddings=embeddings,
        project_path=project_path,
        include_last_n_sessions=include_last_n
    )


@app.post("/api/sessions/{session_id}/diary")
async def api_create_diary(
    session_id: str,
    request: Request
):
    """Create a detailed diary entry for a session."""
    try:
        body = await request.json()
        return await create_diary_entry(
            db=db,
            embeddings=embeddings,
            session_id=session_id,
            project_path=body.get("project_path"),
            user_notes=body.get("user_notes")
        )
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error creating diary for session {session_id}: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error creating diary for session {session_id}: {e}")
        return {
            "success": False,
            "error_code": "DIARY_CREATE_ERROR",
            "error": str(e)
        }


@app.get("/api/sessions/{session_id}/inactivity")
async def api_check_inactivity(
    session_id: str,
    threshold_hours: float = 4.0
):
    """Check if a session should be auto-summarized due to inactivity."""
    return await check_session_inactivity(
        db=db,
        session_id=session_id,
        inactivity_threshold_hours=threshold_hours
    )


# ============= Insights API =============

@app.get("/api/insights")
async def api_get_insights(
    insight_type: Optional[str] = None,
    project_path: Optional[str] = None,
    min_confidence: float = 0.5,
    limit: int = 20
):
    """Get cross-session learning insights."""
    from skills.insights import get_insights as get_insights_skill
    return await get_insights_skill(
        db=db,
        embeddings=embeddings,
        insight_type=insight_type,
        project_path=project_path,
        min_confidence=min_confidence,
        limit=limit
    )


@app.post("/api/insights/aggregate")
async def api_run_aggregation(days_back: int = 30):
    """Run cross-session learning aggregation."""
    from skills.insights import run_aggregation as run_agg
    return await run_agg(db=db, embeddings=embeddings, days_back=days_back)


@app.get("/api/insights/suggestions")
async def api_get_suggestions(min_confidence: float = 0.7):
    """Get CLAUDE.md improvement suggestions."""
    from skills.insights import suggest_improvements as suggest
    return await suggest(db=db, embeddings=embeddings, min_confidence=min_confidence)


@app.post("/api/insights/{insight_id}/feedback")
async def api_insight_feedback(insight_id: int, request: Request):
    """Record feedback on an insight."""
    from skills.insights import record_insight_feedback as record_fb
    try:
        body = await request.json()
        return await record_fb(
            db=db,
            embeddings=embeddings,
            insight_id=insight_id,
            helpful=body.get("helpful", True),
            session_id=body.get("session_id"),
            comment=body.get("comment")
        )
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error recording feedback for insight {insight_id}: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error recording feedback for insight {insight_id}: {e}")
        return {
            "success": False,
            "error_code": "INSIGHT_FEEDBACK_ERROR",
            "error": str(e)
        }


@app.post("/api/insights/{insight_id}/apply")
async def api_mark_insight_applied(insight_id: int):
    """Mark an insight as applied to CLAUDE.md."""
    from skills.insights import mark_insight_applied as mark_applied
    return await mark_applied(db=db, embeddings=embeddings, insight_id=insight_id)


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
    """Get all available MCP servers with live configured status."""
    mcps = load_configured_mcps()
    configured_count = sum(1 for m in mcps if m.get("configured"))
    return {
        "success": True,
        "mcps": mcps,
        "total": len(mcps),
        "configured": configured_count
    }


@app.get("/api/hooks")
async def get_all_hooks():
    """Get all available hooks with live configured status."""
    hooks = load_configured_hooks()
    configured_count = sum(1 for h in hooks if h.get("configured"))
    # Group by trigger for dashboard convenience
    by_trigger: dict = {}
    for h in hooks:
        trigger = h.get("trigger", "Unknown")
        by_trigger.setdefault(trigger, []).append(h)
    return {
        "success": True,
        "hooks": hooks,
        "total": len(hooks),
        "configured": configured_count,
        "by_trigger": by_trigger
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

        # Build MCP status map using dynamic loader
        live_mcps = load_configured_mcps(project_path)
        mcp_status = {}
        for config in (mcp_configs or []):
            mcp_status[config['mcp_id']] = {
                'enabled': bool(config['enabled']),
                'settings': json.loads(config['settings']) if config['settings'] else {}
            }

        for mcp in live_mcps:
            if mcp['id'] not in mcp_status:
                mcp_status[mcp['id']] = {
                    'enabled': mcp['default_enabled'],
                    'configured': mcp.get('configured', False),
                    'settings': {}
                }
            else:
                mcp_status[mcp['id']]['configured'] = mcp.get('configured', False)

        # Build hook status map using dynamic loader
        live_hooks = load_configured_hooks(project_path)
        hook_status = {}
        for config in (hook_configs or []):
            hook_status[config['hook_id']] = {
                'enabled': bool(config['enabled']),
                'settings': json.loads(config['settings']) if config['settings'] else {}
            }

        for hook in live_hooks:
            if hook['id'] not in hook_status:
                hook_status[hook['id']] = {
                    'enabled': hook['default_enabled'],
                    'configured': hook.get('configured', False),
                    'trigger': hook.get('trigger', ''),
                    'settings': {}
                }
            else:
                hook_status[hook['id']]['configured'] = hook.get('configured', False)
                hook_status[hook['id']]['trigger'] = hook.get('trigger', '')

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
                "enabled_mcps": sum(1 for m in mcp_status.values() if m.get('enabled')),
                "total_mcps": len(live_mcps),
                "configured_mcps": sum(1 for m in mcp_status.values() if m.get('configured')),
                "enabled_hooks": sum(1 for h in hook_status.values() if h.get('enabled')),
                "total_hooks": len(live_hooks),
                "configured_hooks": sum(1 for h in hook_status.values() if h.get('configured'))
            }
        }
    except DatabaseError as e:
        logger.error(f"Database error getting project config for {project_path}: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error getting project config for {project_path}: {e}")
        return {
            "success": False,
            "error_code": "PROJECT_CONFIG_ERROR",
            "error": str(e)
        }


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
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error updating agent config {agent_id} for {project_path}: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error updating agent config {agent_id} for {project_path}: {e}")
        return {
            "success": False,
            "error_code": "AGENT_CONFIG_UPDATE_ERROR",
            "error": str(e)
        }


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
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error updating MCP config {mcp_id} for {project_path}: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error updating MCP config {mcp_id} for {project_path}: {e}")
        return {
            "success": False,
            "error_code": "MCP_CONFIG_UPDATE_ERROR",
            "error": str(e)
        }


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
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error updating hook config {hook_id} for {project_path}: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error updating hook config {hook_id} for {project_path}: {e}")
        return {
            "success": False,
            "error_code": "HOOK_CONFIG_UPDATE_ERROR",
            "error": str(e)
        }


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
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error updating preferences for {project_path}: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error updating preferences for {project_path}: {e}")
        return {
            "success": False,
            "error_code": "PREFERENCES_UPDATE_ERROR",
            "error": str(e)
        }


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
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_code": "INVALID_JSON",
            "error": f"Invalid JSON in request body: {str(e)}"
        }
    except DatabaseError as e:
        logger.error(f"Database error in bulk agent update for {project_path}: {e}")
        return {
            "success": False,
            "error_code": e.error_code,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error in bulk agent update for {project_path}: {e}")
        return {
            "success": False,
            "error_code": "BULK_UPDATE_ERROR",
            "error": str(e)
        }


# ============= WebSocket Endpoint for Real-time Updates =============

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates.

    Clients can:
    - Receive broadcasts of all memory/timeline events
    - Subscribe to specific event types
    - Filter by project path
    """
    ws_manager = get_websocket_manager()
    client_id = await ws_manager.connect(websocket)

    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_json()
            await ws_manager.handle_message(client_id, data)

    except WebSocketDisconnect:
        await ws_manager.disconnect(client_id)
    except Exception:
        await ws_manager.disconnect(client_id)


@app.get("/api/ws/stats")
async def websocket_stats():
    """Get WebSocket connection statistics."""
    ws_manager = get_websocket_manager()
    return {
        "success": True,
        "stats": ws_manager.get_stats()
    }


# ============================================================
# KNOWLEDGE GRAPH API ENDPOINTS
# ============================================================

@app.get("/api/graph/full")
async def get_full_graph(
    project_path: Optional[str] = None,
    limit: int = 200
):
    """
    Get the full knowledge graph for visualization.
    Returns nodes (memories) and edges (relationships).
    """
    try:
        # Normalize project_path if provided
        if project_path:
            project_path = normalize_path(project_path)

        data = await db.get_graph_data(project_path=project_path, limit=limit)
        return {
            "success": True,
            "nodes": data.get("nodes", []),
            "edges": data.get("edges", []),
            "stats": data.get("stats", {})
        }
    except Exception as e:
        logger.error(f"Failed to get graph data: {e}")
        return {"success": False, "error": str(e), "nodes": [], "edges": []}


@app.get("/api/graph/node/{memory_id}")
async def get_graph_node(memory_id: int):
    """
    Get a single node with all its relationships.
    Used when clicking a node in the visualization.
    """
    try:
        # Get the memory itself
        memory = await db.get_memory(memory_id)
        if not memory:
            return {"success": False, "error": "Memory not found"}

        # Get all relationships
        outgoing = await db.get_related_memories(memory_id, direction='outgoing', depth=1)
        incoming = await db.get_related_memories(memory_id, direction='incoming', depth=1)

        return {
            "success": True,
            "node": memory,
            "relationships": {
                "outgoing": outgoing,
                "incoming": incoming
            }
        }
    except Exception as e:
        logger.error(f"Failed to get node {memory_id}: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/graph/subgraph")
async def get_subgraph(
    memory_id: int,
    depth: int = 2
):
    """
    Get a connected subgraph starting from a specific node.
    Used for focused exploration of relationships.
    """
    try:
        data = await db.get_subgraph(memory_id, depth=min(depth, 5))  # Cap depth at 5
        return {
            "success": True,
            "center_id": memory_id,
            "depth": depth,
            "nodes": data.get("nodes", []),
            "edges": data.get("edges", [])
        }
    except Exception as e:
        logger.error(f"Failed to get subgraph for {memory_id}: {e}")
        return {"success": False, "error": str(e), "nodes": [], "edges": []}


@app.get("/api/graph/stats")
async def get_graph_stats(project_path: Optional[str] = None):
    """
    Get statistics about the knowledge graph.
    Used for dashboard overview.
    """
    try:
        # Normalize project_path if provided
        if project_path:
            project_path = normalize_path(project_path)

        stats = await db.get_relationship_stats(project_path)
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get graph stats: {e}")
        return {"success": False, "error": str(e), "stats": {}}


class CreateRelationshipRequest(BaseModel):
    """Request body for creating a relationship."""
    source_id: int
    target_id: int
    relationship: str
    strength: float = 1.0


@app.post("/api/graph/relationship")
async def create_graph_relationship(request: CreateRelationshipRequest):
    """
    Create a relationship between two memories.
    Used by curator to apply suggested links.
    """
    try:
        result = await db.create_relationship(
            source_id=request.source_id,
            target_id=request.target_id,
            relationship=request.relationship,
            strength=request.strength
        )
        if result.get("error"):
            return {"success": False, "error": result["error"]}

        # Broadcast update
        await broadcast_event(
            EventTypes.MEMORY_UPDATED,
            {
                "action": "relationship_created",
                "source_id": request.source_id,
                "target_id": request.target_id,
                "relationship": request.relationship
            }
        )

        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Failed to create relationship: {e}")
        return {"success": False, "error": str(e)}


# ============= Curator Agent API =============

class CuratorExploreRequest(BaseModel):
    """Request body for graph exploration."""
    start_node_id: int
    max_depth: int = 3
    mode: str = "bfs"
    relationship_filter: Optional[List[str]] = None


class CuratorMergeRequest(BaseModel):
    """Request body for merging memories."""
    keep_id: int
    remove_ids: List[int]
    merge_content: bool = False


class CuratorConfigUpdateRequest(BaseModel):
    """Request body for updating curator config."""
    auto_dedup_enabled: Optional[bool] = None
    auto_link_enabled: Optional[bool] = None
    dedup_threshold: Optional[float] = None
    maintenance_interval_hours: Optional[int] = None
    curator_active: Optional[bool] = None


@app.post("/api/curator/explore")
async def curator_explore_endpoint(request: CuratorExploreRequest):
    """
    Explore the memory graph from a starting node.
    Uses BFS or DFS traversal to find connected nodes and clusters.
    """
    try:
        from services.curator import get_curator
        curator = get_curator(db, embeddings)

        result = await curator.explore_graph(
            start_node_id=request.start_node_id,
            max_depth=request.max_depth,
            mode=request.mode,
            relationship_filter=request.relationship_filter
        )

        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Curator explore failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/curator/duplicates")
async def curator_duplicates_endpoint(
    project_path: Optional[str] = None,
    similarity_threshold: float = 0.92,
    limit: int = 50
):
    """
    Find semantically similar (duplicate) memories.
    Returns duplicate clusters with merge suggestions.
    """
    try:
        from services.curator import get_curator
        curator = get_curator(db, embeddings)

        if project_path:
            project_path = normalize_path(project_path)

        result = await curator.find_duplicates(
            project_path=project_path,
            similarity_threshold=similarity_threshold,
            limit=limit
        )

        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Curator duplicates failed: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/curator/suggest-links")
async def curator_suggest_links_endpoint(
    memory_id: Optional[int] = None,
    project_path: Optional[str] = None,
    similarity_threshold: float = 0.7,
    limit: int = 20
):
    """
    Suggest missing relationships between memories.
    Returns potential links with confidence scores.
    """
    try:
        from services.curator import get_curator
        curator = get_curator(db, embeddings)

        if project_path:
            project_path = normalize_path(project_path)

        result = await curator.suggest_relationships(
            memory_id=memory_id,
            project_path=project_path,
            similarity_threshold=similarity_threshold,
            limit=limit
        )

        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Curator suggest-links failed: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/curator/merge")
async def curator_merge_endpoint(request: CuratorMergeRequest):
    """
    Merge duplicate memories into one.
    Transfers relationships and optionally merges content.
    """
    try:
        from services.curator import get_curator
        curator = get_curator(db, embeddings)

        result = await curator.merge_memories(
            keep_id=request.keep_id,
            remove_ids=request.remove_ids,
            merge_content=request.merge_content
        )

        # Broadcast update
        if result.get("success"):
            await broadcast_event(
                EventTypes.MEMORY_UPDATED,
                {
                    "action": "merged",
                    "kept_id": request.keep_id,
                    "merged_count": result.get("merged_count")
                }
            )

        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Curator merge failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/curator/orphans")
async def curator_orphans_endpoint(
    project_path: Optional[str] = None,
    limit: int = 50
):
    """
    Find memories with no relationships (orphans).
    """
    try:
        from services.curator import get_curator
        curator = get_curator(db, embeddings)

        if project_path:
            project_path = normalize_path(project_path)

        orphans = await curator.find_orphan_memories(
            project_path=project_path,
            limit=limit
        )

        return {
            "success": True,
            "orphans": orphans,
            "total_found": len(orphans)
        }
    except Exception as e:
        logger.error(f"Curator orphans failed: {e}")
        return {"success": False, "error": str(e), "orphans": []}


@app.get("/api/curator/status")
async def curator_status_endpoint():
    """
    Get current curator agent status.
    """
    try:
        from services.curator import get_curator
        curator = get_curator(db, embeddings)

        status = await curator.get_status()

        return {"success": True, **status}
    except Exception as e:
        logger.error(f"Curator status failed: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/curator/run")
async def curator_run_endpoint(
    project_path: Optional[str] = None,
    tasks: Optional[List[str]] = None
):
    """
    Trigger curator maintenance manually.
    Available tasks: dedup, orphans, links, decay, quality
    """
    try:
        from services.curator import get_curator
        curator = get_curator(db, embeddings)

        if project_path:
            project_path = normalize_path(project_path)

        report = await curator.run_maintenance(
            project_path=project_path,
            tasks=tasks
        )

        return {"success": True, "report": report}
    except Exception as e:
        logger.error(f"Curator run failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/curator/report")
async def curator_report_endpoint(project_path: Optional[str] = None):
    """
    Get the latest curator report.
    """
    try:
        from services.curator import get_curator
        curator = get_curator(db, embeddings)

        if project_path:
            project_path = normalize_path(project_path)

        report = await curator.get_latest_report(project_path=project_path)

        if report:
            return {"success": True, "report": report}
        return {"success": True, "report": None, "message": "No reports found"}
    except Exception as e:
        logger.error(f"Curator report failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/curator/quality")
async def curator_quality_endpoint(
    memory_id: Optional[int] = None,
    project_path: Optional[str] = None,
    limit: int = 100
):
    """
    Calculate quality scores for memories.
    """
    try:
        from services.curator import get_curator
        curator = get_curator(db, embeddings)

        if project_path:
            project_path = normalize_path(project_path)

        result = await curator.score_quality(
            memory_id=memory_id,
            project_path=project_path,
            limit=limit
        )

        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Curator quality failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/curator/summary")
async def curator_summary_endpoint(
    query: str,
    project_path: Optional[str] = None,
    max_memories: int = 10,
    include_graph: bool = True
):
    """
    Generate curated context summary for a query.
    This is what gets injected into the main Claude's context.
    """
    try:
        from services.curator import get_curator
        curator = get_curator(db, embeddings)

        if project_path:
            project_path = normalize_path(project_path)

        result = await curator.generate_summary(
            query=query,
            project_path=project_path,
            max_memories=max_memories,
            include_graph=include_graph
        )

        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Curator summary failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/curator/config")
async def curator_config_get_endpoint(project_path: Optional[str] = None):
    """
    Get curator configuration.
    """
    try:
        from services.curator import get_curator
        curator = get_curator(db, embeddings)

        if project_path:
            project_path = normalize_path(project_path)

        config = await curator.get_config(project_path=project_path)

        return {"success": True, "config": config}
    except Exception as e:
        logger.error(f"Curator config get failed: {e}")
        return {"success": False, "error": str(e)}


@app.put("/api/curator/config/{project_path:path}")
async def curator_config_update_endpoint(
    project_path: str,
    request: CuratorConfigUpdateRequest
):
    """
    Update curator configuration for a project.
    """
    try:
        from services.curator import get_curator
        curator = get_curator(db, embeddings)

        normalized = normalize_path(project_path)

        config_updates = {}
        if request.auto_dedup_enabled is not None:
            config_updates["auto_dedup_enabled"] = request.auto_dedup_enabled
        if request.auto_link_enabled is not None:
            config_updates["auto_link_enabled"] = request.auto_link_enabled
        if request.dedup_threshold is not None:
            config_updates["dedup_threshold"] = request.dedup_threshold
        if request.maintenance_interval_hours is not None:
            config_updates["maintenance_interval_hours"] = request.maintenance_interval_hours
        if request.curator_active is not None:
            config_updates["curator_active"] = request.curator_active

        config = await curator.update_config(project_path=normalized, **config_updates)

        return {"success": True, "config": config}
    except Exception as e:
        logger.error(f"Curator config update failed: {e}")
        return {"success": False, "error": str(e)}


# ============= Memory Decay API Endpoints =============

@app.post("/api/decay/run")
async def decay_run_endpoint():
    """Run memory decay maintenance - evaluate all decayable memories and archive expired ones."""
    try:
        from services.memory_decay import MemoryDecayService
        from config import config as app_config
        decay_service = MemoryDecayService(
            db=db,
            archive_threshold=app_config.DECAY_ARCHIVE_THRESHOLD
        )
        result = await decay_service.apply_decay()
        return result
    except Exception as e:
        logger.error(f"Decay maintenance failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/decay/stats")
async def decay_stats_endpoint():
    """Get memory decay statistics - permanent vs decayable counts, at-risk memories."""
    try:
        from services.memory_decay import MemoryDecayService
        from config import config as app_config
        decay_service = MemoryDecayService(
            db=db,
            archive_threshold=app_config.DECAY_ARCHIVE_THRESHOLD
        )
        result = await decay_service.get_decay_stats()
        return result
    except Exception as e:
        logger.error(f"Decay stats failed: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/decay/boost/{memory_id}")
async def decay_boost_endpoint(memory_id: int):
    """Boost a memory's access count to resist decay."""
    try:
        from services.memory_decay import MemoryDecayService
        from config import config as app_config
        decay_service = MemoryDecayService(
            db=db,
            archive_threshold=app_config.DECAY_ARCHIVE_THRESHOLD
        )
        result = await decay_service.boost_on_access(memory_id)
        return result
    except Exception as e:
        logger.error(f"Decay boost failed: {e}")
        return {"success": False, "error": str(e)}


# ============= Tier 1 Auto-Generation API Endpoint =============

@app.post("/api/tier1/generate")
async def tier1_generate_endpoint(request: Request):
    """Generate Tier 1 context from top memories and write to CLAUDE.md.

    Auto-generates a ranked summary of the most important memories and
    writes it into CLAUDE.md between marker comments. All manually-written
    content in CLAUDE.md is preserved.

    Optional JSON body:
        project_path: Filter to a specific project
        dry_run: If true, return preview without writing (default false)
    """
    try:
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass  # No body is fine, all params are optional

        from services.claude_md_sync import get_claude_md_sync
        sync_service = get_claude_md_sync(db, embeddings)
        result = await sync_service.write_tier1_to_claude_md(
            project_path=body.get("project_path"),
            dry_run=body.get("dry_run", False)
        )
        return result
    except Exception as e:
        logger.error(f"Tier 1 generation failed: {e}")
        return {"success": False, "error": str(e)}


# ============= CLaRa-Inspired Memory Enhancement Endpoints =============

@app.get("/api/tiers/stats")
async def tier_stats_endpoint():
    """Get memory distribution across tiers (hot/warm/cold)."""
    try:
        from services.tier_manager import TierManager
        tier_mgr = TierManager(db)
        return await tier_mgr.get_tier_stats()
    except Exception as e:
        logger.error(f"Tier stats failed: {e}")
        return {"error": str(e)}


@app.post("/api/tiers/maintenance")
async def tier_maintenance_endpoint(request: Request):
    """Run tier maintenance (evaluate and update all memory tiers)."""
    try:
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        from services.tier_manager import TierManager
        tier_mgr = TierManager(db)
        return await tier_mgr.run_tier_maintenance(
            skip_recent_hours=body.get("skip_recent_hours", 24)
        )
    except Exception as e:
        logger.error(f"Tier maintenance failed: {e}")
        return {"error": str(e)}


@app.post("/api/consolidation/run")
async def consolidation_run_endpoint():
    """Manually trigger memory consolidation."""
    try:
        from services.consolidation import ConsolidationService
        consolidator = ConsolidationService(db, embeddings)
        return await consolidator.run_consolidation()
    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
        return {"error": str(e)}


@app.get("/api/consolidation/candidates")
async def consolidation_candidates_endpoint():
    """Preview consolidation candidates without executing."""
    try:
        from services.consolidation import ConsolidationService
        consolidator = ConsolidationService(db, embeddings)
        groups = await consolidator.find_consolidation_candidates()
        return {
            "candidate_groups": len(groups),
            "groups": [
                {
                    "size": len(g),
                    "types": list(set(m.get('type', 'chunk') for m in g)),
                    "ids": [m['id'] for m in g],
                    "preview": g[0]['content'][:100] if g else ''
                }
                for g in groups
            ]
        }
    except Exception as e:
        logger.error(f"Consolidation candidates failed: {e}")
        return {"error": str(e)}


@app.post("/api/consolidation/{consolidated_id}/restore")
async def consolidation_restore_endpoint(consolidated_id: int):
    """Deconsolidate: restore original memories from a consolidated memory."""
    try:
        from services.consolidation import ConsolidationService
        consolidator = ConsolidationService(db, embeddings)
        return await consolidator.deconsolidate(consolidated_id)
    except Exception as e:
        logger.error(f"Deconsolidation failed: {e}")
        return {"error": str(e)}


@app.get("/api/consolidation/stats")
async def consolidation_stats_endpoint():
    """Get consolidation statistics."""
    try:
        from services.consolidation import ConsolidationService
        consolidator = ConsolidationService(db, embeddings)
        return await consolidator.get_consolidation_stats()
    except Exception as e:
        logger.error(f"Consolidation stats failed: {e}")
        return {"error": str(e)}


@app.get("/api/embedding-pipeline/stats")
async def embedding_pipeline_stats_endpoint():
    """Get embedding pipeline statistics (cache hits, batch stats)."""
    try:
        from services.embedding_pipeline import get_embedding_pipeline
        pipeline = get_embedding_pipeline()
        if pipeline:
            return pipeline.get_stats()
        return {"error": "Pipeline not initialized"}
    except Exception as e:
        logger.error(f"Embedding pipeline stats failed: {e}")
        return {"error": str(e)}


@app.post("/api/embedding-pipeline/precompute")
async def embedding_precompute_endpoint():
    """Manually trigger embedding pre-computation for memories with missing embeddings."""
    try:
        from services.embedding_pipeline import get_embedding_pipeline
        pipeline = get_embedding_pipeline()
        if pipeline:
            return await pipeline.precompute_missing_embeddings()
        return {"error": "Pipeline not initialized"}
    except Exception as e:
        logger.error(f"Embedding precompute failed: {e}")
        return {"error": str(e)}


@app.post("/api/embeddings/migrate-binary")
async def embeddings_migrate_binary_endpoint():
    """Migrate existing JSON embeddings to binary format for storage savings."""
    try:
        result = await db.migrate_embeddings_to_binary()
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Embedding migration failed: {e}")
        return {"error": str(e)}


# ============================================================
# CROSS-SESSION AWARENESS REST API
# ============================================================

@app.post("/api/sessions/register")
async def api_session_register(request: Request):
    """Register an active session."""
    try:
        body = await request.json()
        awareness = get_session_awareness(db)
        return await awareness.register_session(
            session_id=body.get("session_id", ""),
            project_path=body.get("project_path", ""),
            goal=body.get("goal"),
            label=body.get("label"),
        )
    except Exception as e:
        logger.error(f"Session register failed: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/sessions/heartbeat")
async def api_session_heartbeat(request: Request):
    """Update session heartbeat, return siblings + conflicts."""
    try:
        body = await request.json()
        awareness = get_session_awareness(db)
        return await awareness.heartbeat(
            session_id=body.get("session_id", ""),
            project_path=body.get("project_path", ""),
            files_modified=body.get("files_modified"),
            current_goal=body.get("current_goal"),
            key_decisions=body.get("key_decisions"),
            summary=body.get("summary"),
        )
    except Exception as e:
        logger.error(f"Session heartbeat failed: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/sessions/deregister")
async def api_session_deregister(request: Request):
    """Mark session as completed."""
    try:
        body = await request.json()
        awareness = get_session_awareness(db)
        return await awareness.deregister_session(
            session_id=body.get("session_id", ""),
            project_path=body.get("project_path", ""),
            final_summary=body.get("final_summary"),
        )
    except Exception as e:
        logger.error(f"Session deregister failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/sessions/active")
async def api_active_sessions(project_path: str = "", exclude_session_id: Optional[str] = None):
    """List active sessions for a project."""
    try:
        sessions = await db.get_active_sessions(project_path, exclude_session_id)
        return {"success": True, "sessions": sessions, "count": len(sessions)}
    except Exception as e:
        logger.error(f"Get active sessions failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/sessions/activity-feed")
async def api_session_activity_feed(
    project_path: str = "", limit: int = 20,
    since: Optional[str] = None, exclude_session_id: Optional[str] = None
):
    """Get recent cross-session activity events."""
    try:
        awareness = get_session_awareness(db)
        return await awareness.get_activity_feed(project_path, limit, since, exclude_session_id)
    except Exception as e:
        logger.error(f"Activity feed failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/sessions/catch-up")
async def api_session_catchup(
    session_id: str = "", project_path: str = "", since: Optional[str] = None
):
    """What happened since timestamp, grouped by session."""
    try:
        awareness = get_session_awareness(db)
        return await awareness.get_catchup(session_id, project_path, since)
    except Exception as e:
        logger.error(f"Session catch-up failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/sessions/conflicts")
async def api_session_conflicts(session_id: str = "", project_path: str = ""):
    """Check file conflicts for a session."""
    try:
        awareness = get_session_awareness(db)
        return await awareness.check_conflicts(session_id, project_path)
    except Exception as e:
        logger.error(f"Session conflicts failed: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/sessions/activity")
async def api_post_session_activity(request: Request):
    """Post an event to the cross-session activity feed."""
    try:
        body = await request.json()
        awareness = get_session_awareness(db)
        return await awareness.post_activity(
            session_id=body.get("session_id", ""),
            project_path=body.get("project_path", ""),
            event_type=body.get("event_type", "decision"),
            summary=body.get("summary", ""),
            files=body.get("files"),
        )
    except Exception as e:
        logger.error(f"Post session activity failed: {e}")
        return {"success": False, "error": str(e)}


# ============= Aggregated Grounding Context (v2) =============


@app.post("/api/grounding-context")
async def api_grounding_context(request: Request):
    """Aggregated grounding endpoint for slim hooks.

    Single call that runs all grounding queries in parallel and returns
    a compact text summary (<150 tokens target).

    Body:
        session_id: str
        project_path: str
        user_input: str (optional - enables pattern hints)

    Returns:
        {"success": true, "context": "[MEM] goal: ... | anchors | sessions ..."}
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    session_id = body.get("session_id", "")
    project_path = body.get("project_path", "")
    user_input = body.get("user_input", "")

    parts = []
    tasks_dict = {}

    # 1. Context refresh (anchors, goal, contradictions)
    if session_id:
        async def _get_grounding():
            try:
                return await context_refresh(
                    db=db,
                    embeddings=embeddings,
                    session_id=session_id,
                    include_recent_events=3,
                    include_state=True,
                    include_checkpoint=False,
                    include_relevant_memories=False,
                    check_contradictions=True,
                )
            except Exception as e:
                logger.debug(f"Grounding context_refresh failed: {e}")
                return None
        tasks_dict["grounding"] = _get_grounding()

    # 2. Session heartbeat (parallel sessions + conflicts)
    if session_id and project_path:
        async def _get_sessions():
            try:
                awareness = get_session_awareness(db)
                return await awareness.heartbeat(
                    session_id=session_id,
                    project_path=project_path,
                )
            except Exception as e:
                logger.debug(f"Grounding heartbeat failed: {e}")
                return None
        tasks_dict["sessions"] = _get_sessions()

    # 3. Pattern hints (only if user input provided)
    if user_input and len(user_input) > 10:
        async def _get_patterns():
            try:
                return await search_patterns(
                    db=db,
                    embeddings=embeddings,
                    query=user_input[:300],
                    limit=2,
                    threshold=0.65,
                )
            except Exception as e:
                logger.debug(f"Grounding pattern search failed: {e}")
                return None
        tasks_dict["patterns"] = _get_patterns()

    # 4. Curator status (lightweight)
    async def _get_curator_status():
        try:
            from services.curator import get_curator
            curator = get_curator(db, embeddings)
            return await curator.get_status()
        except Exception as e:
            logger.debug(f"Grounding curator status failed: {e}")
            return None
    tasks_dict["curator"] = _get_curator_status()

    # Run all in parallel
    if tasks_dict:
        keys = list(tasks_dict.keys())
        gathered = await asyncio.gather(
            *[tasks_dict[k] for k in keys],
            return_exceptions=True,
        )
        results = {}
        for k, v in zip(keys, gathered):
            results[k] = v if not isinstance(v, Exception) else None
    else:
        results = {}

    # -- Build compact output --
    # Goal
    grounding = results.get("grounding")
    if grounding and isinstance(grounding, dict) and grounding.get("success"):
        g = grounding.get("grounding", {})
        goal = g.get("current_goal")
        if goal:
            parts.append(f"goal: {goal[:80]}")

        anchors = g.get("anchors", [])
        if anchors:
            parts.append(f"{len(anchors)} anchor{'s' if len(anchors) != 1 else ''}")

        contradictions = g.get("contradictions", [])
        if contradictions:
            c_summaries = [c.get("content", "")[:40] for c in contradictions[:2]]
            parts.append(f"CONFLICT: {'; '.join(c_summaries)}")

    # Parallel sessions
    sessions = results.get("sessions")
    if sessions and isinstance(sessions, dict):
        siblings = sessions.get("active_siblings", [])
        conflicts = sessions.get("file_conflicts", [])
        if siblings:
            labels = [s.get("session_label", s.get("session_id", "")[:8]) for s in siblings]
            parts.append(f"sessions: {', '.join(labels)}")
        if conflicts:
            conflict_files = []
            for c in conflicts:
                conflict_files.extend(c.get("conflicting_files", []))
            if conflict_files:
                parts.append(f"FILE CONFLICT: {', '.join(conflict_files[:3])}")

    # Pattern hints
    patterns = results.get("patterns")
    if patterns and isinstance(patterns, dict):
        p_list = patterns.get("patterns", [])
        if p_list:
            best = p_list[0]
            sim = int(best.get("similarity", 0) * 100)
            name = best.get("name", "")[:30]
            parts.append(f"pattern({sim}%): {name}")

    # Curator warnings
    curator = results.get("curator")
    if curator and isinstance(curator, dict):
        orphans = curator.get("orphan_count", 0)
        if orphans > 20:
            parts.append(f"{orphans} orphans")

    if parts:
        compact = "[MEM] " + " | ".join(parts)
    else:
        compact = ""

    return {
        "success": True,
        "context": compact,
        "token_estimate": len(compact.split()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8102)),
        reload=True
    )
