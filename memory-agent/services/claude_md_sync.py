"""Auto-sync service for CLAUDE.md updates.

Automatically detects significant learnings and updates CLAUDE.md.
Runs periodically or triggered by insight detection.

Tier 1 auto-generation: Writes the top-ranked memories directly into CLAUDE.md
so Claude reads them at session start with zero API cost. The auto-generated
section is delimited by HTML comment markers and is replaced on each run.
"""
import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# Default CLAUDE.md locations
USER_CLAUDE_MD = Path.home() / ".claude" / "CLAUDE.md"


class ClaudeMdSync:
    """Syncs learned preferences and patterns to CLAUDE.md.

    Features:
    - Detects high-importance learnings
    - Groups by category (preferences, patterns, rules)
    - Avoids duplicates
    - Preserves existing content
    """

    def __init__(self, db, embeddings):
        self.db = db
        self.embeddings = embeddings

    def _read_claude_md(self, path: Path) -> str:
        """Read current CLAUDE.md content."""
        if path.exists():
            return path.read_text(encoding='utf-8')
        return ""

    def _write_claude_md(self, path: Path, content: str):
        """Write content to CLAUDE.md."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')

    def _find_section(self, content: str, section_name: str) -> tuple[int, int]:
        """Find a section in CLAUDE.md by header name.

        Returns (start_pos, end_pos) or (-1, -1) if not found.
        """
        pattern = rf'^##\s+{re.escape(section_name)}\s*$'
        lines = content.split('\n')

        start_line = -1
        for i, line in enumerate(lines):
            if re.match(pattern, line, re.IGNORECASE):
                start_line = i
                break

        if start_line == -1:
            return -1, -1

        # Find end (next ## header or end of file)
        end_line = len(lines)
        for i in range(start_line + 1, len(lines)):
            if re.match(r'^##\s+', lines[i]):
                end_line = i
                break

        # Convert to character positions
        start_pos = sum(len(line) + 1 for line in lines[:start_line])
        end_pos = sum(len(line) + 1 for line in lines[:end_line])

        return start_pos, end_pos

    def _content_exists(self, content: str, new_item: str) -> bool:
        """Check if similar content already exists in CLAUDE.md."""
        # Normalize for comparison
        new_normalized = re.sub(r'\s+', ' ', new_item.lower().strip())
        if not new_normalized:
            return False

        for line in content.split('\n'):
            line_normalized = re.sub(r'\s+', ' ', line.lower().strip())
            # Skip empty lines
            if not line_normalized:
                continue
            if new_normalized in line_normalized or line_normalized in new_normalized:
                return True

        return False

    async def get_sync_candidates(
        self,
        project_path: Optional[str] = None,
        min_importance: int = 7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get memories that should be synced to CLAUDE.md.

        Criteria:
        - High importance (>= min_importance)
        - Type is 'decision' or 'preference'
        - Not already synced
        """
        cursor = self.db.conn.cursor()

        query = """
            SELECT id, content, type, importance, tags, created_at
            FROM memories
            WHERE importance >= ?
            AND type IN ('decision', 'preference')
            AND (metadata IS NULL OR metadata NOT LIKE '%"synced_to_claude_md": true%')
        """
        params = [min_importance]

        if project_path:
            query += " AND project_path = ?"
            params.append(project_path)

        query += " ORDER BY importance DESC, created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [
            {
                "id": row[0],
                "content": row[1],
                "type": row[2],
                "importance": row[3],
                "tags": row[4],
                "created_at": row[5]
            }
            for row in rows
        ]

    async def sync_to_claude_md(
        self,
        project_path: Optional[str] = None,
        claude_md_path: Optional[Path] = None,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Sync high-importance learnings to CLAUDE.md.

        Args:
            project_path: Filter to specific project
            claude_md_path: Path to CLAUDE.md (default: user's global)
            dry_run: If True, only preview what would be synced

        Returns:
            Sync results
        """
        path = claude_md_path or USER_CLAUDE_MD
        candidates = await self.get_sync_candidates(project_path)

        if not candidates:
            return {
                "success": True,
                "synced": 0,
                "message": "No new learnings to sync"
            }

        current_content = self._read_claude_md(path)
        additions = []

        for candidate in candidates:
            # Check if already exists
            if self._content_exists(current_content, candidate["content"]):
                continue

            # Categorize
            if candidate["type"] == "preference":
                section = "Preferences"
            elif "pattern" in (candidate.get("tags") or ""):
                section = "Learned Patterns"
            else:
                section = "Project-Specific Rules"

            additions.append({
                "section": section,
                "content": f"- {candidate['content'][:200]}",
                "memory_id": candidate["id"],
                "importance": candidate["importance"]
            })

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "would_sync": len(additions),
                "additions": additions
            }

        # Actually sync
        synced_ids = []
        for addition in additions:
            section = addition["section"]
            new_line = addition["content"]

            # Find or create section
            start, end = self._find_section(current_content, section)

            if start == -1:
                # Create new section at end
                current_content += f"\n\n## {section}\n{new_line}\n"
            else:
                # Insert into existing section
                section_content = current_content[start:end]
                # Insert before end of section
                insert_pos = end - 1 if end > 0 else len(current_content)
                current_content = (
                    current_content[:insert_pos] +
                    f"\n{new_line}" +
                    current_content[insert_pos:]
                )

            synced_ids.append(addition["memory_id"])

        # Write updated content
        self._write_claude_md(path, current_content)

        # Mark memories as synced by setting synced_to_claude_md flag in metadata JSON
        cursor = self.db.conn.cursor()
        for mem_id in synced_ids:
            cursor.execute("""
                UPDATE memories
                SET metadata = json_set(COALESCE(metadata, '{}'), '$.synced_to_claude_md', json('true'))
                WHERE id = ?
            """, [mem_id])

        self.db.conn.commit()

        return {
            "success": True,
            "synced": len(synced_ids),
            "synced_ids": synced_ids,
            "path": str(path)
        }

    # ================================================================
    # TIER 1 AUTO-GENERATION - Zero-cost startup context
    # ================================================================

    # Markers that delimit the auto-generated section in CLAUDE.md.
    # Everything between these markers is replaced on each run.
    TIER1_START_MARKER = "<!-- AUTO-MEMORY START -->"
    TIER1_END_MARKER = "<!-- AUTO-MEMORY END -->"

    # Maximum output budget
    TIER1_MAX_LINES = 150
    TIER1_MAX_CHARS = 4000
    TIER1_CONTENT_TRUNCATE = 100  # Truncate individual content strings

    # Category mapping from memory type to output section header
    TIER1_CATEGORIES = {
        "decision": "Decisions",
        "preference": "Preferences",
        "error": "Known Issues",
        "code": "Patterns",
        "chunk": "Patterns",
        "session": "Patterns",
    }

    async def auto_generate_tier1(
        self,
        project_path: Optional[str] = None,
        max_memories: int = 30
    ) -> Dict[str, Any]:
        """Generate Tier 1 context from the top-ranked memories.

        Queries the database for memories ranked by a composite score:
            score = (importance * 0.4) + (confidence * 0.3)
                  + (access_frequency * 0.2) + (recency * 0.1)

        Where:
            access_frequency = min(access_count / 10, 1.0)
            recency = max(0, 1 - age_days / 90)

        Includes ALL memory types, not just decision/preference.
        Formats the output as categorized markdown within ~150 lines.

        Args:
            project_path: Optional filter to a specific project
            max_memories: Maximum number of memories to include

        Returns:
            Dict with 'success', 'markdown', 'memory_count', 'categories'
        """
        cursor = self.db.conn.cursor()

        # Build query -- compute the composite score in SQL
        # SQLite does not have DATEDIFF, so we compute age via julianday.
        query = """
            SELECT
                id,
                type,
                content,
                importance,
                confidence,
                access_count,
                created_at,
                outcome,
                success,
                project_path,
                project_name,
                (
                    (COALESCE(importance, 5) / 10.0) * 0.4
                    + COALESCE(confidence, 0.5) * 0.3
                    + MIN(COALESCE(access_count, 0) / 10.0, 1.0) * 0.2
                    + MAX(0.0, 1.0 - (julianday('now') - julianday(COALESCE(created_at, datetime('now')))) / 90.0) * 0.1
                ) AS tier1_score
            FROM memories
            WHERE importance >= 5
              AND COALESCE(outcome_status, 'pending') != 'failed'
              AND COALESCE(failure_count, 0) < 3
        """
        params: List[Any] = []

        if project_path:
            query += " AND project_path = ?"
            params.append(project_path)

        query += " ORDER BY tier1_score DESC LIMIT ?"
        params.append(max_memories)

        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        except Exception as e:
            logger.error(f"Tier 1 query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "markdown": "",
                "memory_count": 0,
                "categories": {}
            }

        if not rows:
            empty_md = self._format_tier1_section({})
            return {
                "success": True,
                "markdown": empty_md,
                "memory_count": 0,
                "categories": {}
            }

        # Group rows by category
        categorized: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            row_dict = dict(row)
            mem_type = row_dict.get("type", "chunk")
            category = self.TIER1_CATEGORIES.get(mem_type, "Patterns")

            if category not in categorized:
                categorized[category] = []
            categorized[category].append(row_dict)

        # Build the markdown string, respecting line and char budgets
        markdown = self._format_tier1_section(categorized)

        return {
            "success": True,
            "markdown": markdown,
            "memory_count": len(rows),
            "categories": {cat: len(items) for cat, items in categorized.items()}
        }

    def _format_tier1_section(self, categorized: Dict[str, List[Dict[str, Any]]]) -> str:
        """Format categorized memories into a markdown section with markers.

        Args:
            categorized: Dict mapping category name to list of memory dicts

        Returns:
            Complete markdown string including start/end markers.
        """
        lines: List[str] = []
        lines.append(self.TIER1_START_MARKER)
        lines.append("## Auto-Generated Memory Context")
        lines.append("")
        lines.append(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
        lines.append("")

        if not categorized:
            lines.append("*No memories qualify for Tier 1 context yet.*")
            lines.append("")
            lines.append(self.TIER1_END_MARKER)
            return "\n".join(lines)

        total_chars = sum(len(l) for l in lines)

        # Ordered category output
        category_order = ["Decisions", "Patterns", "Known Issues", "Preferences"]
        for cat_name in category_order:
            if cat_name not in categorized:
                continue

            items = categorized[cat_name]
            # Budget check
            if len(lines) >= self.TIER1_MAX_LINES - 5:
                break
            if total_chars >= self.TIER1_MAX_CHARS - 200:
                break

            lines.append(f"### {cat_name}")
            lines.append("")

            for item in items:
                if len(lines) >= self.TIER1_MAX_LINES - 2:
                    break
                if total_chars >= self.TIER1_MAX_CHARS - 100:
                    break

                entry_line = self._format_tier1_entry(item)
                lines.append(entry_line)
                total_chars += len(entry_line)

            lines.append("")

        lines.append(self.TIER1_END_MARKER)
        return "\n".join(lines)

    def _format_tier1_entry(self, memory: Dict[str, Any]) -> str:
        """Format a single memory as a concise bullet point.

        Truncates content to TIER1_CONTENT_TRUNCATE chars and adds
        metadata annotations for importance and project.
        """
        content = memory.get("content", "")
        # Extract the first meaningful line (skip markdown headers)
        first_line = ""
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                first_line = stripped
                break

        if not first_line:
            first_line = content.replace("\n", " ").strip()

        # Truncate
        if len(first_line) > self.TIER1_CONTENT_TRUNCATE:
            first_line = first_line[:self.TIER1_CONTENT_TRUNCATE].rstrip() + "..."

        # Build annotations
        annotations = []
        importance = memory.get("importance", 5)
        if importance >= 8:
            annotations.append(f"imp:{importance}")

        project_name = memory.get("project_name") or ""
        if project_name:
            annotations.append(project_name)

        success = memory.get("success")
        if success == 0:
            annotations.append("failed")
        elif memory.get("type") == "error" and success == 1:
            annotations.append("fixed")

        suffix = f" [{', '.join(annotations)}]" if annotations else ""
        return f"- {first_line}{suffix}"

    async def write_tier1_to_claude_md(
        self,
        project_path: Optional[str] = None,
        claude_md_path: Optional[Path] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Generate Tier 1 context and write it into CLAUDE.md.

        Reads the existing CLAUDE.md, finds and replaces the auto-generated
        section (between TIER1_START_MARKER and TIER1_END_MARKER), or appends
        it if no markers exist. All manually-written content is preserved.

        Args:
            project_path: Optional filter to a specific project
            claude_md_path: Path to CLAUDE.md (default: user's global)
            dry_run: If True, return the result without writing

        Returns:
            Dict with 'success', 'path', 'memory_count', 'lines_written',
            'categories', and optionally 'preview' for dry_run
        """
        path = claude_md_path or USER_CLAUDE_MD

        # Generate tier 1 content
        result = await self.auto_generate_tier1(project_path=project_path)
        if not result.get("success"):
            return result

        tier1_markdown = result["markdown"]

        # Read existing file
        current_content = self._read_claude_md(path)

        # Find existing markers
        start_idx = current_content.find(self.TIER1_START_MARKER)
        end_idx = current_content.find(self.TIER1_END_MARKER)

        if start_idx != -1 and end_idx != -1:
            # Replace existing section (include the end marker length)
            end_idx += len(self.TIER1_END_MARKER)
            new_content = (
                current_content[:start_idx].rstrip("\n")
                + "\n\n"
                + tier1_markdown
                + "\n"
                + current_content[end_idx:].lstrip("\n")
            )
        else:
            # Append at the end
            separator = "\n\n" if current_content and not current_content.endswith("\n\n") else ""
            if current_content and current_content.endswith("\n"):
                separator = "\n"
            new_content = current_content + separator + tier1_markdown + "\n"

        tier1_lines = tier1_markdown.count("\n") + 1

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "path": str(path),
                "memory_count": result["memory_count"],
                "lines_written": tier1_lines,
                "categories": result["categories"],
                "preview": tier1_markdown
            }

        # Write the file
        try:
            self._write_claude_md(path, new_content)
        except Exception as e:
            logger.error(f"Failed to write CLAUDE.md: {e}")
            return {
                "success": False,
                "error": f"Write failed: {e}",
                "path": str(path)
            }

        return {
            "success": True,
            "path": str(path),
            "memory_count": result["memory_count"],
            "lines_written": tier1_lines,
            "categories": result["categories"]
        }

    async def suggest_updates(
        self,
        project_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get suggestions for CLAUDE.md updates without syncing.

        Returns formatted suggestions that can be reviewed.
        """
        candidates = await self.get_sync_candidates(project_path, min_importance=6)

        suggestions = []
        for c in candidates:
            suggestions.append({
                "type": c["type"],
                "content": c["content"],
                "importance": c["importance"],
                "suggestion": f"Add to CLAUDE.md: {c['content'][:150]}..."
            })

        return {
            "success": True,
            "suggestions": suggestions,
            "count": len(suggestions)
        }


# Global instance
_sync_service: Optional[ClaudeMdSync] = None


def get_claude_md_sync(db, embeddings) -> ClaudeMdSync:
    """Get the global CLAUDE.md sync service."""
    global _sync_service
    if _sync_service is None:
        _sync_service = ClaudeMdSync(db, embeddings)
    return _sync_service
