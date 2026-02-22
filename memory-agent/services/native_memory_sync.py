"""One-way sync: Claude Code's native MEMORY.md -> MCP vector DB.

Native MEMORY.md is owned exclusively by Claude Code's auto memory.
This module ingests its contents into the MCP vector DB at session end
so they become searchable via semantic search.

The MCP-to-Native direction has been removed to avoid competing with
Claude Code for the 200-line MEMORY.md budget.

Dedup: markdown_syncs.content_hash prevents duplicate imports.
"""
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from services.native_memory_paths import get_native_memory_md, list_native_memory_files

logger = logging.getLogger(__name__)

# Source tag used for native->MCP memories
NATIVE_SOURCE_TAG = "source=native_memory_md"

# Legacy markers (used only for stripping during import)
_MCP_SYNC_START = "<!-- MCP-SYNCED START -->"
_MCP_SYNC_END = "<!-- MCP-SYNCED END -->"


def _ensure_markdown_syncs_table(conn):
    """Create the markdown_syncs table if it doesn't exist yet."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS markdown_syncs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            memory_id INTEGER,
            project_path TEXT,
            synced_at TEXT DEFAULT (datetime('now')),
            content_hash TEXT,
            FOREIGN KEY (memory_id) REFERENCES memories(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_markdown_syncs_type ON markdown_syncs(file_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_markdown_syncs_project ON markdown_syncs(project_path)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_markdown_syncs_memory ON markdown_syncs(memory_id)")
    conn.commit()


def _content_hash(text: str) -> str:
    """Generate a short hash for dedup tracking."""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()[:16]


def _strip_mcp_section(content: str) -> str:
    """Remove any legacy MCP-SYNCED fenced section from content."""
    pattern = re.compile(
        re.escape(_MCP_SYNC_START) + r".*?" + re.escape(_MCP_SYNC_END),
        re.DOTALL,
    )
    stripped = pattern.sub("", content)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.rstrip("\n") + "\n" if stripped.strip() else ""


def _parse_markdown_entries(content: str, source_file: str = "MEMORY.md") -> List[Dict[str, Any]]:
    """Parse markdown into discrete entries for import.

    Splits on ## headers. Each section becomes one entry.
    Bullet groups under a section are kept together.
    Very short entries (< 20 chars) are skipped.
    """
    if not content.strip():
        return []

    entries = []
    lines = content.splitlines()

    current_section = ""
    current_lines: list = []

    for line in lines:
        if line.startswith("## "):
            # Flush previous section
            if current_lines:
                text = "\n".join(current_lines).strip()
                if len(text) >= 20:
                    entries.append({
                        "content": text,
                        "section": current_section,
                        "source_file": source_file,
                    })
            current_section = line[3:].strip()
            current_lines = [line]
        elif line.startswith("# ") and not current_lines:
            continue
        else:
            current_lines.append(line)

    # Flush last section
    if current_lines:
        text = "\n".join(current_lines).strip()
        if len(text) >= 20:
            entries.append({
                "content": text,
                "section": current_section,
                "source_file": source_file,
            })

    return entries


# ── Native -> MCP (the only sync direction) ──────────────────────────


async def sync_native_to_mcp(
    db,
    embeddings,
    project_path: str,
) -> Dict[str, Any]:
    """Sync native MEMORY.md content into the MCP vector DB.

    1. Read native MEMORY.md + topic files
    2. Parse into entries (## sections or bullet groups)
    3. Strip any legacy MCP-SYNCED section (avoid circular import)
    4. Hash each entry, check markdown_syncs for file_type='native_to_mcp'
    5. For new/changed entries: generate embedding, store in MCP DB
    6. Record in markdown_syncs

    Args:
        db: DatabaseService instance
        embeddings: EmbeddingService instance
        project_path: Absolute project path

    Returns:
        Dict with sync results
    """
    from services.database import normalize_path
    norm_path = normalize_path(project_path)

    _ensure_markdown_syncs_table(db.conn)

    all_files = list_native_memory_files(project_path)
    if not all_files:
        return {"success": True, "synced": 0, "reason": "no native memory files"}

    # 1. Read and parse all files
    entries = []
    for fpath in all_files:
        try:
            raw = fpath.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Native->MCP: failed to read {fpath}: {e}")
            continue

        # Strip any legacy MCP-synced section
        clean = _strip_mcp_section(raw)
        parsed = _parse_markdown_entries(clean, source_file=fpath.name)
        entries.extend(parsed)

    if not entries:
        return {"success": True, "synced": 0, "reason": "no parseable entries"}

    # 2. Check what's already synced
    cursor = db.conn.cursor()
    already_synced_hashes = set()
    try:
        cursor.execute("""
            SELECT content_hash FROM markdown_syncs
            WHERE file_type = 'native_to_mcp'
            AND project_path = ?
        """, (norm_path,))
        already_synced_hashes = {row["content_hash"] for row in cursor.fetchall()}
    except Exception as e:
        logger.warning(f"Native->MCP: failed to check synced hashes: {e}")

    # 3. Filter to new/changed entries
    new_entries = []
    for entry in entries:
        h = _content_hash(entry["content"])
        if h not in already_synced_hashes:
            entry["hash"] = h
            new_entries.append(entry)

    if not new_entries:
        return {"success": True, "synced": 0, "reason": "all already synced"}

    # 4. Generate embeddings and store
    now = datetime.now().isoformat()
    synced_count = 0
    errors = []

    for entry in new_entries:
        try:
            emb_result = await embeddings.generate_embedding(entry["content"])
            if hasattr(emb_result, "ok") and not emb_result.ok:
                logger.warning(f"Native->MCP: embedding failed for entry: {emb_result.error_message}")
                errors.append(entry["content"][:50])
                continue

            embedding = emb_result.embedding if hasattr(emb_result, "embedding") else emb_result

            tags = [NATIVE_SOURCE_TAG]
            if entry.get("source_file"):
                tags.append(f"file={entry['source_file']}")
            if entry.get("section"):
                tags.append(f"section={entry['section']}")

            embedding_json = json.dumps(
                embedding if isinstance(embedding, list) else embedding.tolist()
                if hasattr(embedding, "tolist") else list(embedding)
            )

            cursor.execute("""
                INSERT INTO memories (type, content, embedding, project_path, importance, tags, created_at)
                VALUES ('chunk', ?, ?, ?, 6, ?, ?)
            """, (
                entry["content"],
                embedding_json,
                norm_path,
                json.dumps(tags),
                now,
            ))
            memory_id = cursor.lastrowid

            cursor.execute("""
                INSERT INTO markdown_syncs (file_type, file_path, memory_id, project_path, synced_at, content_hash)
                VALUES ('native_to_mcp', ?, ?, ?, ?, ?)
            """, (
                entry.get("source_file", "MEMORY.md"),
                memory_id,
                norm_path,
                now,
                entry["hash"],
            ))
            synced_count += 1

        except Exception as e:
            logger.error(f"Native->MCP: failed to sync entry: {e}")
            errors.append(str(e))

    db.conn.commit()

    return {
        "success": True,
        "synced": synced_count,
        "total_entries": len(entries),
        "new_entries": len(new_entries),
        "errors": errors if errors else None,
    }
