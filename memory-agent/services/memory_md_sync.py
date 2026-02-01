"""MEMORY.md Sync Service - Moltbot-inspired core facts file.

Maintains a single MEMORY.md file per project containing:
- Anchors (verified facts)
- Key decisions (importance >= 7)
- Proven patterns (success_count >= 3)
- User preferences

Storage: <project>/.claude/MEMORY.md
"""
import os
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def get_memory_md_path(project_path: str) -> Path:
    """Get the path to the MEMORY.md file for a project.

    Args:
        project_path: Root path of the project

    Returns:
        Path to MEMORY.md file
    """
    # Normalize project path
    project_path = project_path.replace("\\", "/").rstrip("/")

    # MEMORY.md lives directly in .claude folder
    claude_dir = Path(project_path) / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)

    return claude_dir / "MEMORY.md"


def _format_memory_md(
    anchors: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    patterns: List[Dict[str, Any]],
    preferences: List[Dict[str, Any]],
    last_updated: datetime
) -> str:
    """Format the MEMORY.md content.

    Args:
        anchors: List of verified facts/anchors
        decisions: List of important decisions
        patterns: List of proven solution patterns
        preferences: List of user preferences
        last_updated: Timestamp for the file

    Returns:
        Formatted markdown content
    """
    lines = [
        "# MEMORY.md - Core Facts",
        f"Last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "<!-- This file is auto-generated from the memory database. -->",
        "<!-- High-importance memories and proven patterns are synced here. -->",
        ""
    ]

    # Anchors section
    if anchors:
        lines.append("## Anchors (Verified Facts)")
        lines.append("")
        for anchor in anchors:
            fact = anchor.get("fact", anchor.get("content", ""))
            if anchor.get("created_at"):
                date_str = anchor["created_at"][:10]
                lines.append(f"- [{date_str}] {fact}")
            else:
                lines.append(f"- {fact}")
        lines.append("")

    # Decisions section
    if decisions:
        lines.append("## Key Decisions")
        lines.append("")
        for decision in decisions:
            content = decision.get("content", "")[:200]
            date_str = decision.get("created_at", "")[:10] if decision.get("created_at") else ""
            importance = decision.get("importance", 5)
            if date_str:
                lines.append(f"- [{date_str}] (imp:{importance}) {content}")
            else:
                lines.append(f"- (imp:{importance}) {content}")
        lines.append("")

    # Patterns section
    if patterns:
        lines.append("## Patterns (Proven Solutions)")
        lines.append("")
        for pattern in patterns:
            name = pattern.get("name", "Unnamed")
            solution = pattern.get("solution", "")[:150]
            success_count = pattern.get("success_count", 0)
            lines.append(f"### {name} (used: {success_count}x)")
            lines.append(f"{solution}")
            lines.append("")

    # Preferences section
    if preferences:
        lines.append("## Preferences")
        lines.append("")
        for pref in preferences:
            content = pref.get("content", "")
            lines.append(f"- {content}")
        lines.append("")

    return "\n".join(lines)


async def sync_to_memory_md(
    db,
    project_path: str,
    min_importance: int = 7,
    min_pattern_success: int = 3
) -> Dict[str, Any]:
    """Sync high-importance memories to MEMORY.md.

    Queries the database for important content and writes to MEMORY.md.

    Args:
        db: Database service instance
        project_path: Root path of the project
        min_importance: Minimum importance level for decisions (default 7)
        min_pattern_success: Minimum success count for patterns (default 3)

    Returns:
        Dict with sync results
    """
    from services.database import normalize_path
    normalized_path = normalize_path(project_path)

    # Query anchors (from timeline_events where is_anchor=1)
    cursor = db.conn.cursor()

    anchors = []
    try:
        cursor.execute("""
            SELECT summary as fact, created_at
            FROM timeline_events
            WHERE is_anchor = 1
            AND (project_path = ? OR project_path IS NULL)
            ORDER BY created_at DESC
            LIMIT 20
        """, (normalized_path,))
        anchors = [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.warning(f"Failed to query anchors: {e}")

    # Query high-importance decisions
    decisions = []
    try:
        cursor.execute("""
            SELECT content, importance, created_at
            FROM memories
            WHERE type = 'decision'
            AND importance >= ?
            AND (project_path = ? OR project_path IS NULL)
            ORDER BY importance DESC, created_at DESC
            LIMIT 15
        """, (min_importance, normalized_path))
        decisions = [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.warning(f"Failed to query decisions: {e}")

    # Query proven patterns
    patterns = []
    try:
        cursor.execute("""
            SELECT name, solution, success_count
            FROM patterns
            WHERE success_count >= ?
            ORDER BY success_count DESC
            LIMIT 10
        """, (min_pattern_success,))
        patterns = [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.warning(f"Failed to query patterns: {e}")

    # Query preferences (memories with type='preference')
    preferences = []
    try:
        cursor.execute("""
            SELECT content, importance
            FROM memories
            WHERE type = 'preference'
            AND (project_path = ? OR project_path IS NULL)
            ORDER BY importance DESC, created_at DESC
            LIMIT 10
        """, (normalized_path,))
        preferences = [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.warning(f"Failed to query preferences: {e}")

    # Format and write MEMORY.md
    now = datetime.now()
    content = _format_memory_md(anchors, decisions, patterns, preferences, now)

    memory_path = get_memory_md_path(project_path)

    try:
        # Calculate content hash for tracking
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]

        memory_path.write_text(content, encoding="utf-8")

        return {
            "success": True,
            "file_path": str(memory_path),
            "synced_at": now.isoformat(),
            "content_hash": content_hash,
            "counts": {
                "anchors": len(anchors),
                "decisions": len(decisions),
                "patterns": len(patterns),
                "preferences": len(preferences)
            }
        }

    except Exception as e:
        logger.error(f"Failed to write MEMORY.md: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def read_memory_md(project_path: str) -> Dict[str, Any]:
    """Read the MEMORY.md file for a project.

    Args:
        project_path: Root path of the project

    Returns:
        Dict with file content and metadata
    """
    memory_path = get_memory_md_path(project_path)

    if not memory_path.exists():
        return {
            "success": True,
            "exists": False,
            "content": "",
            "file_path": str(memory_path)
        }

    try:
        content = memory_path.read_text(encoding="utf-8")
        stat = memory_path.stat()

        return {
            "success": True,
            "exists": True,
            "content": content,
            "file_path": str(memory_path),
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to read MEMORY.md: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def add_fact(
    project_path: str,
    fact: str,
    section: str = "anchors"
) -> Dict[str, Any]:
    """Add a fact directly to MEMORY.md without going through the database.

    This is for quick additions that should persist immediately.

    Args:
        project_path: Root path of the project
        fact: The fact/decision/preference to add
        section: Section to add to (anchors, decisions, preferences)

    Returns:
        Dict with success status
    """
    memory_path = get_memory_md_path(project_path)

    try:
        # Read existing content or create new
        if memory_path.exists():
            content = memory_path.read_text(encoding="utf-8")
        else:
            content = f"# MEMORY.md - Core Facts\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Find the right section and append
        date_str = datetime.now().strftime("%Y-%m-%d")
        new_line = f"- [{date_str}] {fact}\n"

        section_headers = {
            "anchors": "## Anchors (Verified Facts)",
            "decisions": "## Key Decisions",
            "patterns": "## Patterns (Proven Solutions)",
            "preferences": "## Preferences"
        }

        target_header = section_headers.get(section, section_headers["anchors"])

        if target_header in content:
            # Find the section and add after the header
            parts = content.split(target_header)
            if len(parts) == 2:
                # Find the next section or end
                after_header = parts[1]
                # Insert after any blank line following header
                insert_pos = 0
                for i, char in enumerate(after_header):
                    if char == '\n':
                        insert_pos = i + 1
                        if i + 1 < len(after_header) and after_header[i + 1] not in ('\n', '-'):
                            break
                    elif char != '\n' and char != ' ':
                        break

                after_header = after_header[:insert_pos] + new_line + after_header[insert_pos:]
                content = parts[0] + target_header + after_header
        else:
            # Section doesn't exist, create it
            content += f"\n{target_header}\n\n{new_line}"

        # Update the last updated timestamp
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("Last updated:"):
                lines[i] = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                break
        content = "\n".join(lines)

        memory_path.write_text(content, encoding="utf-8")

        return {
            "success": True,
            "file_path": str(memory_path),
            "section": section,
            "fact": fact
        }

    except Exception as e:
        logger.error(f"Failed to add fact to MEMORY.md: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def get_memory_md_summary(project_path: str) -> Dict[str, Any]:
    """Get a summary of MEMORY.md for context injection.

    Returns a condensed version suitable for grounding context.

    Args:
        project_path: Root path of the project

    Returns:
        Dict with summary content
    """
    result = await read_memory_md(project_path)

    if not result.get("success") or not result.get("exists"):
        return {
            "success": True,
            "summary": "",
            "exists": False
        }

    content = result.get("content", "")

    # Extract just the essential facts (first line of each bullet)
    lines = content.split("\n")
    summary_lines = []
    in_section = False
    current_section = ""

    for line in lines:
        if line.startswith("## "):
            in_section = True
            current_section = line[3:].split("(")[0].strip()
            summary_lines.append(f"**{current_section}**:")
        elif in_section and line.startswith("- "):
            # Extract first 100 chars of the fact
            fact = line[2:].strip()[:100]
            summary_lines.append(f"  {fact}")
        elif line.startswith("### ") and in_section:
            # Pattern name
            pattern_name = line[4:].split("(")[0].strip()
            summary_lines.append(f"  Pattern: {pattern_name}")

    return {
        "success": True,
        "summary": "\n".join(summary_lines),
        "exists": True,
        "line_count": len(summary_lines)
    }
