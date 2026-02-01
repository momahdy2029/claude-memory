"""Daily Log Service - Moltbot-inspired human-readable session logs.

Creates and manages YYYY-MM-DD.md append-only files for session activity.
Provides transparent, human-readable logs that persist beyond context window.

Storage: <project>/.claude/memory/YYYY-MM-DD.md
"""
import os
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def get_log_path(project_path: str, log_date: Optional[date] = None) -> Path:
    """Get the path for a daily log file.

    Args:
        project_path: Root path of the project
        log_date: Date for the log file (defaults to today)

    Returns:
        Path to the daily log markdown file
    """
    if log_date is None:
        log_date = date.today()

    # Normalize project path
    project_path = project_path.replace("\\", "/").rstrip("/")

    # Create memory directory structure
    memory_dir = Path(project_path) / ".claude" / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    # Log filename is YYYY-MM-DD.md
    filename = log_date.strftime("%Y-%m-%d.md")
    return memory_dir / filename


def _get_log_header(log_date: date) -> str:
    """Generate header for a new daily log file."""
    return f"# Daily Log - {log_date.strftime('%Y-%m-%d')}\n\n"


async def append_entry(
    project_path: str,
    content: str,
    entry_type: str = "note",
    session_id: Optional[str] = None,
    timestamp: Optional[datetime] = None
) -> Dict[str, Any]:
    """Append an entry to today's daily log.

    Args:
        project_path: Root path of the project
        content: Content to append
        entry_type: Type of entry (decision, accomplishment, note, error, session_summary)
        session_id: Optional session ID for context
        timestamp: Optional timestamp (defaults to now)

    Returns:
        Dict with success status and file path
    """
    if timestamp is None:
        timestamp = datetime.now()

    log_path = get_log_path(project_path)

    # Create file with header if it doesn't exist
    file_existed = log_path.exists()

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            if not file_existed:
                f.write(_get_log_header(timestamp.date()))

            # Format entry based on type
            time_str = timestamp.strftime("%H:%M:%S")

            if entry_type == "session_summary":
                # Session summary gets its own section
                session_label = f" ({session_id[:8]})" if session_id else ""
                f.write(f"\n## Session{session_label} - {time_str}\n")
                f.write(content)
                f.write("\n\n---\n")
            elif entry_type == "decision":
                f.write(f"- **[{time_str}] Decision**: {content}\n")
            elif entry_type == "accomplishment":
                f.write(f"- **[{time_str}] Done**: {content}\n")
            elif entry_type == "error":
                f.write(f"- **[{time_str}] Error**: {content}\n")
            else:
                # Generic note
                f.write(f"- [{time_str}] {content}\n")

        return {
            "success": True,
            "file_path": str(log_path),
            "entry_type": entry_type,
            "timestamp": timestamp.isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to append to daily log: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def append_session_summary(
    project_path: str,
    session_id: str,
    decisions: Optional[List[str]] = None,
    accomplishments: Optional[List[str]] = None,
    notes: Optional[List[str]] = None,
    errors_solved: Optional[List[str]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """Append a full session summary to the daily log.

    Args:
        project_path: Root path of the project
        session_id: Session identifier
        decisions: List of decisions made
        accomplishments: List of things accomplished
        notes: List of notes/observations
        errors_solved: List of errors that were solved
        start_time: Session start time
        end_time: Session end time (defaults to now)

    Returns:
        Dict with success status
    """
    if end_time is None:
        end_time = datetime.now()

    # Build summary content
    lines = []

    if start_time:
        duration = end_time - start_time
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        lines.append(f"**Duration**: {hours}h {minutes}m\n")

    if decisions:
        lines.append("\n### Decisions")
        for d in decisions:
            lines.append(f"- {d}")

    if accomplishments:
        lines.append("\n### Accomplishments")
        for a in accomplishments:
            lines.append(f"- {a}")

    if errors_solved:
        lines.append("\n### Errors Solved")
        for e in errors_solved:
            lines.append(f"- {e}")

    if notes:
        lines.append("\n### Notes")
        for n in notes:
            lines.append(f"- {n}")

    content = "\n".join(lines)

    return await append_entry(
        project_path=project_path,
        content=content,
        entry_type="session_summary",
        session_id=session_id,
        timestamp=end_time
    )


async def load_recent_logs(
    project_path: str,
    days: int = 2,
    max_chars: int = 8000
) -> Dict[str, Any]:
    """Load recent daily logs.

    Args:
        project_path: Root path of the project
        days: Number of days to look back (default 2)
        max_chars: Maximum characters to return (default 8000)

    Returns:
        Dict with success status and combined log content
    """
    logs = []
    total_chars = 0

    today = date.today()

    for i in range(days):
        log_date = today - timedelta(days=i)
        log_path = get_log_path(project_path, log_date)

        if log_path.exists():
            try:
                content = log_path.read_text(encoding="utf-8")

                # Check if adding this would exceed limit
                if total_chars + len(content) > max_chars:
                    # Truncate to fit
                    remaining = max_chars - total_chars
                    if remaining > 200:  # Only include if meaningful content
                        content = content[:remaining] + "\n\n... (truncated)"
                        logs.append({
                            "date": log_date.isoformat(),
                            "content": content,
                            "truncated": True
                        })
                    break

                logs.append({
                    "date": log_date.isoformat(),
                    "content": content,
                    "truncated": False
                })
                total_chars += len(content)

            except Exception as e:
                logger.warning(f"Failed to read log {log_path}: {e}")

    return {
        "success": True,
        "logs": logs,
        "days_loaded": len(logs),
        "total_chars": total_chars
    }


async def get_today_highlights(
    project_path: str,
    max_entries: int = 10
) -> Dict[str, Any]:
    """Get today's log highlights for context injection.

    Extracts the most important entries from today's log.

    Args:
        project_path: Root path of the project
        max_entries: Maximum number of entries to return

    Returns:
        Dict with success status and highlights
    """
    log_path = get_log_path(project_path)

    if not log_path.exists():
        return {
            "success": True,
            "highlights": [],
            "has_log": False
        }

    try:
        content = log_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        highlights = []
        for line in lines:
            # Extract important entries (decisions, accomplishments)
            if "**Decision**" in line or "**Done**" in line:
                highlights.append(line.strip("- ").strip())
                if len(highlights) >= max_entries:
                    break

        return {
            "success": True,
            "highlights": highlights,
            "has_log": True,
            "entry_count": len(highlights)
        }

    except Exception as e:
        logger.error(f"Failed to get highlights: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def list_logs(
    project_path: str,
    limit: int = 30
) -> Dict[str, Any]:
    """List available daily log files.

    Args:
        project_path: Root path of the project
        limit: Maximum number of logs to list

    Returns:
        Dict with list of log files and their sizes
    """
    memory_dir = Path(project_path) / ".claude" / "memory"

    if not memory_dir.exists():
        return {
            "success": True,
            "logs": [],
            "total_count": 0
        }

    logs = []
    for log_file in sorted(memory_dir.glob("????-??-??.md"), reverse=True):
        if len(logs) >= limit:
            break

        try:
            stat = log_file.stat()
            logs.append({
                "date": log_file.stem,
                "path": str(log_file),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to stat {log_file}: {e}")

    return {
        "success": True,
        "logs": logs,
        "total_count": len(logs)
    }
