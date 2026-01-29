"""Auto-sync service for CLAUDE.md updates.

Automatically detects significant learnings and updates CLAUDE.md.
Runs periodically or triggered by insight detection.
"""
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


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
        self._last_sync_time = 0

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

        # Mark memories as synced
        cursor = self.db.conn.cursor()
        for mem_id in synced_ids:
            cursor.execute("""
                UPDATE memories
                SET metadata = COALESCE(metadata, '{}')
                WHERE id = ?
            """, [mem_id])
            # TODO: Properly update JSON metadata

        self.db.conn.commit()

        return {
            "success": True,
            "synced": len(synced_ids),
            "synced_ids": synced_ids,
            "path": str(path)
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
