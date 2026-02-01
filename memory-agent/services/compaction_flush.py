"""Pre-Compaction Flush Service - Export memories before context loss.

Since Claude Code doesn't expose a pre-compaction hook, this service uses
heuristic-based flush detection:
- Flush if events_since_checkpoint > 50
- Flush if session active > 30 minutes without flush

Creates flush_YYYYMMDD_HHMMSS.md files in <project>/.claude/memory/
"""
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Flush thresholds
EVENT_THRESHOLD = 50  # Flush after this many events
TIME_THRESHOLD_MINUTES = 30  # Flush after this many minutes


def get_flush_path(project_path: str, timestamp: Optional[datetime] = None) -> Path:
    """Get the path for a flush file.

    Args:
        project_path: Root path of the project
        timestamp: Timestamp for the flush file (defaults to now)

    Returns:
        Path to the flush markdown file
    """
    if timestamp is None:
        timestamp = datetime.now()

    # Normalize project path
    project_path = project_path.replace("\\", "/").rstrip("/")

    # Create memory directory structure
    memory_dir = Path(project_path) / ".claude" / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    # Flush filename includes full timestamp
    filename = timestamp.strftime("flush_%Y%m%d_%H%M%S.md")
    return memory_dir / filename


async def check_flush_needed(
    db,
    session_id: str,
    event_threshold: int = EVENT_THRESHOLD,
    time_threshold_minutes: int = TIME_THRESHOLD_MINUTES
) -> Dict[str, Any]:
    """Check if a pre-compaction flush is needed.

    Uses heuristics since Claude Code doesn't expose compaction hooks:
    1. Event count since last checkpoint
    2. Time since last flush or session start

    Args:
        db: Database service instance
        session_id: Current session ID
        event_threshold: Number of events to trigger flush
        time_threshold_minutes: Minutes since last flush to trigger

    Returns:
        Dict with flush_needed flag and reason
    """
    cursor = db.conn.cursor()

    # Get session state
    cursor.execute("""
        SELECT last_checkpoint_id, last_flush_at, events_since_checkpoint, created_at
        FROM session_state
        WHERE session_id = ?
    """, (session_id,))
    row = cursor.fetchone()

    if not row:
        return {
            "flush_needed": False,
            "reason": "no_session_state"
        }

    state = dict(row)
    events_count = state.get("events_since_checkpoint", 0)
    last_flush = state.get("last_flush_at")
    session_start = state.get("created_at")

    reasons = []

    # Check event threshold
    if events_count >= event_threshold:
        reasons.append(f"events_threshold ({events_count} >= {event_threshold})")

    # Check time threshold
    reference_time = last_flush if last_flush else session_start
    if reference_time:
        try:
            ref_dt = datetime.fromisoformat(reference_time.replace("Z", "+00:00"))
            # Handle naive datetime
            if ref_dt.tzinfo:
                ref_dt = ref_dt.replace(tzinfo=None)
            minutes_elapsed = (datetime.now() - ref_dt).total_seconds() / 60

            if minutes_elapsed >= time_threshold_minutes:
                reasons.append(f"time_threshold ({minutes_elapsed:.1f}min >= {time_threshold_minutes}min)")
        except Exception as e:
            logger.warning(f"Failed to parse reference time: {e}")

    return {
        "flush_needed": len(reasons) > 0,
        "reasons": reasons,
        "events_since_checkpoint": events_count,
        "session_id": session_id
    }


async def format_flush_markdown(
    db,
    session_id: str,
    project_path: str
) -> str:
    """Format the flush content as markdown.

    Gathers all important session data for human-readable export.

    Args:
        db: Database service instance
        session_id: Session to flush
        project_path: Project path for context

    Returns:
        Formatted markdown content
    """
    from services.database import normalize_path
    normalized_path = normalize_path(project_path)

    cursor = db.conn.cursor()
    now = datetime.now()

    lines = [
        f"# Memory Flush - {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Session: `{session_id}`",
        f"Project: `{project_path}`",
        "",
        "---",
        ""
    ]

    # Get high-importance decisions from this session
    cursor.execute("""
        SELECT content, importance, created_at, outcome
        FROM memories
        WHERE session_id = ?
        AND type = 'decision'
        AND importance >= 7
        ORDER BY importance DESC, created_at DESC
        LIMIT 10
    """, (session_id,))
    decisions = cursor.fetchall()

    if decisions:
        lines.append("## Important Decisions")
        lines.append("")
        for d in decisions:
            d = dict(d)
            content = d.get("content", "")[:300]
            importance = d.get("importance", 5)
            outcome = d.get("outcome")
            lines.append(f"### Decision (importance: {importance})")
            lines.append(content)
            if outcome:
                lines.append(f"\n**Outcome**: {outcome}")
            lines.append("")

    # Get anchors (verified facts) from this session
    cursor.execute("""
        SELECT summary, details, created_at
        FROM timeline_events
        WHERE session_id = ?
        AND is_anchor = 1
        ORDER BY created_at DESC
        LIMIT 15
    """, (session_id,))
    anchors = cursor.fetchall()

    if anchors:
        lines.append("## Anchors (Verified Facts)")
        lines.append("")
        for a in anchors:
            a = dict(a)
            summary = a.get("summary", "")
            details = a.get("details")
            lines.append(f"- {summary}")
            if details:
                lines.append(f"  - Details: {details[:150]}")
        lines.append("")

    # Get recent events
    cursor.execute("""
        SELECT event_type, summary, created_at, status
        FROM timeline_events
        WHERE session_id = ?
        ORDER BY created_at DESC
        LIMIT 30
    """, (session_id,))
    events = cursor.fetchall()

    if events:
        lines.append("## Recent Actions")
        lines.append("")
        for e in events:
            e = dict(e)
            event_type = e.get("event_type", "unknown")
            summary = e.get("summary", "")[:100]
            timestamp = e.get("created_at", "")[:19]
            status = e.get("status", "")
            status_str = f" [{status}]" if status else ""
            lines.append(f"- [{timestamp}] **{event_type}**{status_str}: {summary}")
        lines.append("")

    # Get session state
    cursor.execute("""
        SELECT current_goal, pending_questions, decisions_summary, entity_registry
        FROM session_state
        WHERE session_id = ?
    """, (session_id,))
    state_row = cursor.fetchone()

    if state_row:
        state = dict(state_row)
        if state.get("current_goal"):
            lines.append("## Current Goal")
            lines.append(state["current_goal"])
            lines.append("")

        if state.get("pending_questions"):
            import json
            try:
                questions = json.loads(state["pending_questions"])
                if questions:
                    lines.append("## Pending Questions")
                    for q in questions:
                        lines.append(f"- {q}")
                    lines.append("")
            except Exception:
                pass

        if state.get("entity_registry"):
            import json
            try:
                registry = json.loads(state["entity_registry"])
                if registry:
                    lines.append("## Entity Registry")
                    for key, value in list(registry.items())[:20]:
                        lines.append(f"- `{key}`: {value}")
                    lines.append("")
            except Exception:
                pass

    # Get errors solved in this session
    cursor.execute("""
        SELECT content, outcome
        FROM memories
        WHERE session_id = ?
        AND type = 'error'
        AND success = 1
        ORDER BY created_at DESC
        LIMIT 5
    """, (session_id,))
    errors = cursor.fetchall()

    if errors:
        lines.append("## Errors Solved")
        lines.append("")
        for e in errors:
            e = dict(e)
            content = e.get("content", "")[:200]
            outcome = e.get("outcome", "")[:100]
            lines.append(f"- **Error**: {content}")
            if outcome:
                lines.append(f"  - **Solution**: {outcome}")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated at {now.isoformat()}*")

    return "\n".join(lines)


async def execute_flush(
    db,
    project_path: str,
    session_id: str
) -> Dict[str, Any]:
    """Execute a pre-compaction flush.

    Exports all important session data to a markdown file.

    Args:
        db: Database service instance
        project_path: Root path of the project
        session_id: Session to flush

    Returns:
        Dict with flush results
    """
    now = datetime.now()
    flush_path = get_flush_path(project_path, now)

    try:
        # Generate flush content
        content = await format_flush_markdown(db, session_id, project_path)

        # Write to file
        flush_path.write_text(content, encoding="utf-8")

        # Update session state with flush timestamp
        cursor = db.conn.cursor()
        cursor.execute("""
            UPDATE session_state
            SET last_flush_at = ?,
                events_since_checkpoint = 0
            WHERE session_id = ?
        """, (now.isoformat(), session_id))
        db.conn.commit()

        return {
            "success": True,
            "file_path": str(flush_path),
            "flushed_at": now.isoformat(),
            "session_id": session_id,
            "content_length": len(content)
        }

    except Exception as e:
        logger.error(f"Failed to execute flush: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def list_flushes(
    project_path: str,
    limit: int = 20
) -> Dict[str, Any]:
    """List available flush files for a project.

    Args:
        project_path: Root path of the project
        limit: Maximum number of flushes to list

    Returns:
        Dict with list of flush files
    """
    memory_dir = Path(project_path) / ".claude" / "memory"

    if not memory_dir.exists():
        return {
            "success": True,
            "flushes": [],
            "total_count": 0
        }

    flushes = []
    for flush_file in sorted(memory_dir.glob("flush_*.md"), reverse=True):
        if len(flushes) >= limit:
            break

        try:
            stat = flush_file.stat()
            # Parse timestamp from filename
            name = flush_file.stem  # flush_YYYYMMDD_HHMMSS
            timestamp_str = name.replace("flush_", "")

            flushes.append({
                "filename": flush_file.name,
                "path": str(flush_file),
                "timestamp": timestamp_str,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to process flush file {flush_file}: {e}")

    return {
        "success": True,
        "flushes": flushes,
        "total_count": len(flushes)
    }


async def read_flush(
    project_path: str,
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """Read a flush file.

    Args:
        project_path: Root path of the project
        filename: Specific flush filename (defaults to most recent)

    Returns:
        Dict with flush content
    """
    memory_dir = Path(project_path) / ".claude" / "memory"

    if not memory_dir.exists():
        return {
            "success": False,
            "error": "No memory directory found"
        }

    if filename:
        flush_path = memory_dir / filename
    else:
        # Get most recent flush
        flushes = sorted(memory_dir.glob("flush_*.md"), reverse=True)
        if not flushes:
            return {
                "success": False,
                "error": "No flush files found"
            }
        flush_path = flushes[0]

    if not flush_path.exists():
        return {
            "success": False,
            "error": f"Flush file not found: {flush_path}"
        }

    try:
        content = flush_path.read_text(encoding="utf-8")
        return {
            "success": True,
            "content": content,
            "file_path": str(flush_path),
            "filename": flush_path.name
        }
    except Exception as e:
        logger.error(f"Failed to read flush file: {e}")
        return {
            "success": False,
            "error": str(e)
        }
