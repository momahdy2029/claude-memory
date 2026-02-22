"""Cross-session awareness service.

Provides higher-level operations for tracking concurrent Claude Code sessions,
detecting file conflicts, and enabling session catch-up. Wraps DatabaseService
methods with business logic.

Usage:
    from services.session_awareness import get_session_awareness
    awareness = get_session_awareness(db)
    result = await awareness.register_session(session_id, project_path)
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from services.database import DatabaseService

logger = logging.getLogger(__name__)

_instance: Optional["SessionAwarenessService"] = None


def get_session_awareness(db: DatabaseService) -> "SessionAwarenessService":
    """Get or create the singleton SessionAwarenessService."""
    global _instance
    if _instance is None or _instance.db is not db:
        _instance = SessionAwarenessService(db)
    return _instance


class SessionAwarenessService:
    """High-level service for cross-session awareness.

    Wraps raw DB methods with business logic like:
    - Auto-posting activity events on register/deregister
    - Returning siblings + conflicts on heartbeat
    - Grouping catch-up events by session
    """

    def __init__(self, db: DatabaseService):
        self.db = db

    async def register_session(
        self, session_id: str, project_path: str,
        goal: Optional[str] = None, label: Optional[str] = None
    ) -> Dict[str, Any]:
        """Register a session, post session_start activity, return active siblings."""
        await self.db.register_active_session(session_id, project_path, goal, label)

        await self.db.post_session_activity(
            session_id, project_path, "session_start",
            f"Session started{': ' + goal if goal else ''}"
        )

        siblings = await self.db.get_active_sessions(project_path, exclude_session_id=session_id)

        return {
            "success": True,
            "session_id": session_id,
            "active_siblings": siblings,
            "sibling_count": len(siblings),
        }

    async def heartbeat(
        self, session_id: str, project_path: str,
        files_modified: Optional[List[str]] = None,
        current_goal: Optional[str] = None,
        key_decisions: Optional[List[str]] = None,
        summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update heartbeat and return siblings + file conflicts."""
        await self.db.heartbeat_session(
            session_id, files_modified, current_goal, key_decisions, summary
        )

        siblings = await self.db.get_active_sessions(project_path, exclude_session_id=session_id)
        conflicts = await self.db.detect_file_conflicts(session_id, project_path)

        return {
            "success": True,
            "active_siblings": siblings,
            "sibling_count": len(siblings),
            "file_conflicts": conflicts,
            "has_conflicts": len(conflicts) > 0,
        }

    async def deregister_session(
        self, session_id: str, project_path: str,
        final_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mark session completed and post session_end activity."""
        if final_summary:
            await self.db.heartbeat_session(session_id, summary=final_summary)

        await self.db.post_session_activity(
            session_id, project_path, "session_end",
            final_summary or "Session ended"
        )

        result = await self.db.deregister_session(session_id)
        return result

    async def post_activity(
        self, session_id: str, project_path: str,
        event_type: str, summary: str, files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Post an activity event to the cross-session feed."""
        return await self.db.post_session_activity(
            session_id, project_path, event_type, summary, files
        )

    async def get_activity_feed(
        self, project_path: str, limit: int = 20,
        since: Optional[str] = None, exclude_session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get recent cross-session activity feed."""
        events = await self.db.get_session_activity_feed(
            project_path, limit, since, exclude_session_id
        )
        return {
            "success": True,
            "events": events,
            "count": len(events),
        }

    async def get_catchup(
        self, session_id: str, project_path: str,
        since: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get 'what happened while I was away' grouped by session."""
        events = await self.db.get_session_activity_feed(
            project_path, limit=50, since=since, exclude_session_id=session_id
        )

        # Group events by session
        by_session: Dict[str, List[Dict[str, Any]]] = {}
        for ev in events:
            sid = ev["session_id"]
            if sid not in by_session:
                by_session[sid] = []
            by_session[sid].append(ev)

        # Get session labels/goals for context
        siblings = await self.db.get_active_sessions(project_path)
        session_info = {s["session_id"]: s for s in siblings}

        grouped = []
        for sid, session_events in by_session.items():
            info = session_info.get(sid, {})
            grouped.append({
                "session_id": sid,
                "session_label": info.get("session_label", ""),
                "current_goal": info.get("current_goal", ""),
                "status": info.get("status", "completed"),
                "events": session_events,
                "event_count": len(session_events),
            })

        return {
            "success": True,
            "sessions": grouped,
            "total_events": len(events),
        }

    async def check_conflicts(
        self, session_id: str, project_path: str
    ) -> Dict[str, Any]:
        """Check for file conflicts with sibling sessions."""
        conflicts = await self.db.detect_file_conflicts(session_id, project_path)
        return {
            "success": True,
            "conflicts": conflicts,
            "has_conflicts": len(conflicts) > 0,
        }

    async def cleanup_stale(
        self, idle_minutes: int = 10, completed_minutes: int = 30
    ) -> Dict[str, Any]:
        """Run stale session cleanup."""
        return await self.db.cleanup_stale_sessions(idle_minutes, completed_minutes)
