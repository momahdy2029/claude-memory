"""Session state management skills."""
from typing import Dict, Any, Optional, List
from services.database import DatabaseService
from services.timeline import TimelineService, SESSION_GAP_SECONDS


async def state_get(
    db: DatabaseService,
    session_id: Optional[str] = None,
    project_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get current session state.

    Args:
        db: Database service instance
        session_id: Specific session ID (optional)
        project_path: Get latest session for project (if no session_id)

    Returns:
        Dict with session state
    """
    if session_id:
        state = await db.get_or_create_session_state(session_id, project_path)
    elif project_path:
        state = await db.get_latest_session_for_project(project_path)
        if not state:
            return {
                "success": True,
                "state": None,
                "message": "No session found for this project"
            }
    else:
        return {
            "success": False,
            "error": "Must provide either session_id or project_path"
        }

    return {
        "success": True,
        "state": state,
        "session_id": state.get("session_id"),
        "current_goal": state.get("current_goal"),
        "entity_registry": state.get("entity_registry", {}),
        "pending_questions": state.get("pending_questions", []),
        "events_since_checkpoint": state.get("events_since_checkpoint", 0)
    }


async def state_update(
    db: DatabaseService,
    session_id: str,
    current_goal: Optional[str] = None,
    pending_questions: Optional[List[str]] = None,
    add_question: Optional[str] = None,
    remove_question: Optional[str] = None,
    register_entity: Optional[Dict[str, str]] = None,
    entity_registry: Optional[Dict[str, str]] = None,
    add_decision: Optional[str] = None,
    decisions_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update session state.

    Args:
        db: Database service instance
        session_id: The session ID
        current_goal: Set current goal
        pending_questions: Replace pending questions list
        add_question: Add a single question to pending
        remove_question: Remove a question from pending
        register_entity: Add entity to registry {"key": "value"}
        entity_registry: Replace entire entity registry
        add_decision: Add a decision to the summary
        decisions_summary: Replace entire decisions summary

    Returns:
        Dict with updated state
    """
    # Get current state first
    state = await db.get_or_create_session_state(session_id)

    # Handle question modifications
    final_questions = pending_questions
    if final_questions is None:
        final_questions = state.get("pending_questions", [])

    if add_question and add_question not in final_questions:
        final_questions = list(final_questions) + [add_question]

    if remove_question and remove_question in final_questions:
        final_questions = [q for q in final_questions if q != remove_question]

    # Handle entity registry modifications
    final_registry = entity_registry
    if final_registry is None:
        final_registry = state.get("entity_registry", {})

    if register_entity:
        final_registry = {**final_registry, **register_entity}

    # Handle decisions summary
    final_decisions = decisions_summary
    if final_decisions is None:
        final_decisions = state.get("decisions_summary", "")

    if add_decision:
        if final_decisions:
            final_decisions = f"{final_decisions}\n- {add_decision}"
        else:
            final_decisions = f"- {add_decision}"

    # Perform update
    success = await db.update_session_state(
        session_id=session_id,
        current_goal=current_goal,
        pending_questions=final_questions,
        entity_registry=final_registry,
        decisions_summary=final_decisions
    )

    # Get updated state
    updated_state = await db.get_or_create_session_state(session_id)

    return {
        "success": success,
        "session_id": session_id,
        "state": updated_state,
        "message": "Session state updated"
    }


async def state_init_session(
    db: DatabaseService,
    embeddings,
    project_path: str
) -> Dict[str, Any]:
    """
    Initialize or resume a session for a project.

    Handles the 4-hour gap logic for session boundaries.

    Args:
        db: Database service instance
        embeddings: Embedding service instance
        project_path: Project path

    Returns:
        Dict with session info and context
    """
    timeline = TimelineService(db, embeddings)

    session_id, is_new, previous_state = await timeline.get_or_create_session(project_path)

    result = {
        "success": True,
        "session_id": session_id,
        "is_new_session": is_new,
        "project_path": project_path
    }

    if is_new and previous_state:
        # Load context from previous session
        prev_checkpoint = await db.get_latest_checkpoint(previous_state["session_id"])
        result["previous_session"] = {
            "session_id": previous_state["session_id"],
            "last_goal": previous_state.get("current_goal"),
            "last_activity": previous_state.get("last_activity_at"),
            "checkpoint": prev_checkpoint
        }
        result["message"] = f"New session created (previous session timed out after {SESSION_GAP_SECONDS // 3600}h gap)"
    elif not is_new:
        # Load current session context
        context = await timeline.load_session_context(session_id)
        result["state"] = context["state"]
        result["recent_events"] = context["recent_events"][:5]  # Last 5 events
        result["checkpoint"] = context["checkpoint"]
        result["message"] = "Continuing existing session"
    else:
        result["message"] = "New session created (no previous session)"

    return result
