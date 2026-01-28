"""Skills for managing CLAUDE.md instructions file."""
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


def get_claude_md_path() -> Path:
    """Get the path to the user's CLAUDE.md file."""
    # Check common locations
    home = Path.home()

    # Primary location: ~/.claude/CLAUDE.md
    primary = home / ".claude" / "CLAUDE.md"
    if primary.exists():
        return primary

    # Create if doesn't exist
    primary.parent.mkdir(parents=True, exist_ok=True)
    return primary


def read_claude_md() -> str:
    """Read the current CLAUDE.md content."""
    path = get_claude_md_path()
    if path.exists():
        return path.read_text(encoding='utf-8')
    return ""


def write_claude_md(content: str) -> bool:
    """Write content to CLAUDE.md."""
    path = get_claude_md_path()
    try:
        path.write_text(content, encoding='utf-8')
        return True
    except Exception as e:
        return False


async def claude_md_read(
    section: Optional[str] = None
) -> Dict[str, Any]:
    """
    Read CLAUDE.md content.

    Args:
        section: Optional section header to read (e.g., "Memory System")

    Returns:
        Dict with content and metadata
    """
    content = read_claude_md()

    if not content:
        return {
            "success": True,
            "exists": False,
            "content": None,
            "message": "CLAUDE.md does not exist or is empty"
        }

    if section:
        # Extract specific section
        pattern = rf'^##\s+{re.escape(section)}.*?(?=^##|\Z)'
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        if match:
            return {
                "success": True,
                "exists": True,
                "section": section,
                "content": match.group(0).strip(),
                "path": str(get_claude_md_path())
            }
        else:
            return {
                "success": True,
                "exists": True,
                "section": section,
                "content": None,
                "message": f"Section '{section}' not found"
            }

    return {
        "success": True,
        "exists": True,
        "content": content,
        "path": str(get_claude_md_path())
    }


async def claude_md_add_section(
    section_name: str,
    content: str,
    position: str = "end"
) -> Dict[str, Any]:
    """
    Add a new section to CLAUDE.md.

    Args:
        section_name: Name for the section header (without ##)
        content: Content for the section
        position: Where to add - "end", "start", or "after:<section_name>"

    Returns:
        Dict with result
    """
    current = read_claude_md()

    # Check if section already exists
    if re.search(rf'^##\s+{re.escape(section_name)}\s*$', current, re.MULTILINE | re.IGNORECASE):
        return {
            "success": False,
            "error": f"Section '{section_name}' already exists. Use claude_md_update_section to modify it."
        }

    # Build new section
    new_section = f"\n## {section_name}\n{content}\n"

    if position == "start":
        # After the title line
        if current.startswith("#"):
            lines = current.split('\n', 1)
            new_content = lines[0] + "\n" + new_section + (lines[1] if len(lines) > 1 else "")
        else:
            new_content = new_section + current
    elif position.startswith("after:"):
        after_section = position[6:]
        pattern = rf'(^##\s+{re.escape(after_section)}.*?)(?=^##|\Z)'
        match = re.search(pattern, current, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        if match:
            insert_pos = match.end()
            new_content = current[:insert_pos] + new_section + current[insert_pos:]
        else:
            new_content = current + new_section
    else:  # end
        new_content = current.rstrip() + "\n" + new_section

    if write_claude_md(new_content):
        return {
            "success": True,
            "section": section_name,
            "message": f"Added section '{section_name}' to CLAUDE.md",
            "path": str(get_claude_md_path())
        }
    else:
        return {
            "success": False,
            "error": "Failed to write CLAUDE.md"
        }


async def claude_md_update_section(
    section_name: str,
    content: str,
    mode: str = "replace"
) -> Dict[str, Any]:
    """
    Update an existing section in CLAUDE.md.

    Args:
        section_name: Name of the section to update
        content: New content
        mode: "replace" (replace entire section), "append" (add to end), "prepend" (add to start)

    Returns:
        Dict with result
    """
    current = read_claude_md()

    pattern = rf'(^##\s+{re.escape(section_name)}\s*\n)(.*?)(?=^##|\Z)'
    match = re.search(pattern, current, re.MULTILINE | re.DOTALL | re.IGNORECASE)

    if not match:
        return {
            "success": False,
            "error": f"Section '{section_name}' not found. Use claude_md_add_section to create it."
        }

    header = match.group(1)
    existing_content = match.group(2)

    if mode == "replace":
        new_section_content = content + "\n"
    elif mode == "append":
        new_section_content = existing_content.rstrip() + "\n" + content + "\n"
    elif mode == "prepend":
        new_section_content = content + "\n" + existing_content
    else:
        return {"success": False, "error": f"Unknown mode: {mode}"}

    new_content = current[:match.start()] + header + new_section_content + current[match.end():]

    if write_claude_md(new_content):
        return {
            "success": True,
            "section": section_name,
            "mode": mode,
            "message": f"Updated section '{section_name}' in CLAUDE.md",
            "path": str(get_claude_md_path())
        }
    else:
        return {
            "success": False,
            "error": "Failed to write CLAUDE.md"
        }


async def claude_md_add_instruction(
    section_name: str,
    instruction: str,
    bullet_style: str = "-"
) -> Dict[str, Any]:
    """
    Add a single instruction/rule to a section.

    Args:
        section_name: Section to add instruction to
        instruction: The instruction text
        bullet_style: Bullet character ("-", "*", or numbered like "1.")

    Returns:
        Dict with result
    """
    current = read_claude_md()

    pattern = rf'(^##\s+{re.escape(section_name)}\s*\n)(.*?)(?=^##|\Z)'
    match = re.search(pattern, current, re.MULTILINE | re.DOTALL | re.IGNORECASE)

    if not match:
        # Section doesn't exist, create it
        return await claude_md_add_section(
            section_name,
            f"{bullet_style} {instruction}"
        )

    existing_content = match.group(2).rstrip()

    # Check if instruction already exists
    if instruction.lower() in existing_content.lower():
        return {
            "success": True,
            "already_exists": True,
            "message": f"Instruction already exists in section '{section_name}'"
        }

    # Add the instruction
    new_instruction = f"\n{bullet_style} {instruction}"
    new_section_content = existing_content + new_instruction + "\n"

    new_content = current[:match.start()] + match.group(1) + new_section_content + current[match.end():]

    if write_claude_md(new_content):
        return {
            "success": True,
            "section": section_name,
            "instruction": instruction,
            "message": f"Added instruction to '{section_name}'",
            "path": str(get_claude_md_path())
        }
    else:
        return {
            "success": False,
            "error": "Failed to write CLAUDE.md"
        }


async def claude_md_list_sections() -> Dict[str, Any]:
    """
    List all sections in CLAUDE.md.

    Returns:
        Dict with list of section names
    """
    content = read_claude_md()

    if not content:
        return {
            "success": True,
            "sections": [],
            "message": "CLAUDE.md is empty or doesn't exist"
        }

    sections = re.findall(r'^##\s+(.+?)\s*$', content, re.MULTILINE)

    return {
        "success": True,
        "sections": sections,
        "count": len(sections),
        "path": str(get_claude_md_path())
    }


async def claude_md_suggest_from_session(
    db,
    session_id: str,
    min_importance: int = 7
) -> Dict[str, Any]:
    """
    Suggest CLAUDE.md additions based on session learnings.

    Analyzes anchors, high-confidence decisions, and patterns
    to suggest instructions that should be persisted.

    Args:
        db: Database service
        session_id: Session to analyze
        min_importance: Minimum importance level to consider

    Returns:
        Dict with suggestions
    """
    suggestions = []

    # Get anchors (verified facts)
    events = await db.get_timeline_events(
        session_id=session_id,
        limit=100,
        anchors_only=True
    )

    for event in events:
        if event.get("is_anchor"):
            suggestions.append({
                "type": "anchor",
                "content": event["summary"],
                "suggested_section": "Project Facts",
                "reason": "Verified fact from session"
            })

    # Get high-confidence decisions
    decisions = await db.get_timeline_events(
        session_id=session_id,
        limit=50,
        event_type="decision"
    )

    for decision in decisions:
        confidence = decision.get("confidence", 0)
        if confidence >= 0.8:
            suggestions.append({
                "type": "decision",
                "content": decision["summary"],
                "confidence": confidence,
                "suggested_section": "Development Decisions",
                "reason": "High-confidence decision"
            })

    # Get error patterns that were solved
    errors = await db.search_similar(
        embedding=None,  # Will need embedding service for this
        memory_type="error",
        success_only=True,
        limit=10
    ) if hasattr(db, 'search_similar') else []

    return {
        "success": True,
        "suggestions": suggestions[:10],  # Limit to top 10
        "count": len(suggestions),
        "message": f"Found {len(suggestions)} potential CLAUDE.md additions"
    }
