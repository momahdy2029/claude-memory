"""Bidirectional sync between MCP vector DB and Claude Code's native auto memory.

Direction A (MCP -> Native): Fast, no embeddings needed.
    High-importance MCP memories -> fenced section in native MEMORY.md

Direction B (Native -> MCP): Needs embeddings, runs at session end.
    Native MEMORY.md sections -> MCP DB as type='chunk', tagged source=native_memory_md

Dedup: markdown_syncs.content_hash prevents duplicates in both directions.
"""
import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from services.native_memory_paths import get_native_memory_md, list_native_memory_files

logger = logging.getLogger(__name__)


def _ensure_markdown_syncs_table(conn):
    """Create the markdown_syncs table if it doesn't exist yet.

    Lightweight safeguard: the full initialize_schema() may fail on older DBs
    due to unrelated column mismatches, so we create just the table we need.
    """
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


# Markers for the MCP-synced section in native MEMORY.md
MCP_SYNC_START = "<!-- MCP-SYNCED START -->"
MCP_SYNC_END = "<!-- MCP-SYNCED END -->"

# Budget: native auto memory loads first 200 lines, leave 20 headroom
MAX_NATIVE_LINES = 180

# Source tag used for native->MCP memories
NATIVE_SOURCE_TAG = "source=native_memory_md"


def _content_hash(text: str) -> str:
    """Generate a short hash for dedup tracking."""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()[:16]


# ── Direction A: MCP -> Native ────────────────────────────────────────────


async def sync_mcp_to_native(
    db,
    project_path: str,
    min_importance: int = 7,
) -> Dict[str, Any]:
    """Sync high-importance MCP memories into the native MEMORY.md fenced section.

    1. Query MCP DB for important memories for this project
    2. Skip already-synced (via markdown_syncs with file_type='mcp_to_native')
    3. Read native MEMORY.md, check line budget
    4. Replace content between MCP-SYNCED markers (non-destructive)
    5. Record synced entries in markdown_syncs

    Args:
        db: DatabaseService instance
        project_path: Absolute project path (e.g. C:\\xampp\\htdocs\\server)
        min_importance: Minimum importance threshold (default 7)

    Returns:
        Dict with sync results
    """
    from services.database import normalize_path
    norm_path = normalize_path(project_path)

    _ensure_markdown_syncs_table(db.conn)
    cursor = db.conn.cursor()

    # 1. Query high-importance decisions, preferences, and successful errors
    try:
        cursor.execute("""
            SELECT id, type, content, importance, success, created_at
            FROM memories
            WHERE importance >= ?
            AND (project_path = ? OR project_path IS NULL)
            AND (
                type IN ('decision', 'preference')
                OR (type = 'error' AND success = 1)
            )
            ORDER BY importance DESC, created_at DESC
            LIMIT 30
        """, (min_importance, norm_path))
        memories = [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"MCP->Native: failed to query memories: {e}")
        return {"success": False, "error": str(e)}

    if not memories:
        return {"success": True, "synced": 0, "reason": "no qualifying memories"}

    # 2. Check which are already synced
    already_synced_ids = set()
    try:
        cursor.execute("""
            SELECT memory_id FROM markdown_syncs
            WHERE file_type = 'mcp_to_native'
            AND project_path = ?
        """, (norm_path,))
        already_synced_ids = {row["memory_id"] for row in cursor.fetchall()}
    except Exception as e:
        logger.warning(f"MCP->Native: failed to check synced IDs: {e}")

    # Also get all synced content hashes for this direction
    already_synced_hashes = set()
    try:
        cursor.execute("""
            SELECT content_hash FROM markdown_syncs
            WHERE file_type = 'mcp_to_native'
            AND project_path = ?
        """, (norm_path,))
        already_synced_hashes = {row["content_hash"] for row in cursor.fetchall()}
    except Exception:
        pass

    # Filter to new memories only
    new_memories = []
    for mem in memories:
        h = _content_hash(mem["content"])
        if mem["id"] not in already_synced_ids and h not in already_synced_hashes:
            new_memories.append(mem)

    if not new_memories:
        return {"success": True, "synced": 0, "reason": "all already synced"}

    # 3. Read native MEMORY.md
    native_md_path = get_native_memory_md(project_path)

    if native_md_path.exists():
        existing_content = native_md_path.read_text(encoding="utf-8")
    else:
        existing_content = ""

    # Count lines outside the MCP-synced section
    user_content = _strip_mcp_section(existing_content)
    user_line_count = len(user_content.splitlines())

    if user_line_count >= MAX_NATIVE_LINES:
        logger.warning(
            f"MCP->Native: MEMORY.md already at {user_line_count} lines "
            f"(budget {MAX_NATIVE_LINES}), skipping sync"
        )
        return {
            "success": True,
            "synced": 0,
            "reason": f"line budget exceeded ({user_line_count}/{MAX_NATIVE_LINES})",
        }

    # 4. Build the fenced section content
    # Include previously synced memories too (rebuild full section)
    all_for_section = memories  # all qualifying, not just new
    section_lines = _build_mcp_section(all_for_section)

    # Check combined line count
    total_lines = user_line_count + len(section_lines)
    if total_lines > MAX_NATIVE_LINES:
        # Trim section to fit budget
        budget = MAX_NATIVE_LINES - user_line_count
        if budget < 5:
            return {
                "success": True,
                "synced": 0,
                "reason": "not enough line budget for MCP section",
            }
        section_lines = section_lines[:budget]

    # 5. Replace or append the fenced section
    section_text = "\n".join(section_lines)
    updated_content = _replace_mcp_section(user_content, section_text)

    # Write back
    native_md_path.parent.mkdir(parents=True, exist_ok=True)
    native_md_path.write_text(updated_content, encoding="utf-8")

    # 6. Record in markdown_syncs
    now = datetime.now().isoformat()
    synced_count = 0
    for mem in new_memories:
        h = _content_hash(mem["content"])
        try:
            cursor.execute("""
                INSERT INTO markdown_syncs (file_type, file_path, memory_id, project_path, synced_at, content_hash)
                VALUES ('mcp_to_native', ?, ?, ?, ?, ?)
            """, (str(native_md_path), mem["id"], norm_path, now, h))
            synced_count += 1
        except Exception as e:
            logger.warning(f"MCP->Native: failed to record sync for memory {mem['id']}: {e}")

    db.conn.commit()

    return {
        "success": True,
        "synced": synced_count,
        "total_in_section": len(all_for_section),
        "file": str(native_md_path),
        "user_lines": user_line_count,
        "total_lines": len(updated_content.splitlines()),
    }


def _build_mcp_section(memories: list) -> List[str]:
    """Build the fenced MCP-SYNCED section lines."""
    lines = [
        MCP_SYNC_START,
        "## Synced from MCP Memory DB",
        "",
    ]
    for mem in memories:
        mtype = mem.get("type", "chunk")
        content = mem.get("content", "").replace("\n", " ").strip()
        # Truncate long entries
        if len(content) > 200:
            content = content[:197] + "..."
        importance = mem.get("importance", 5)
        lines.append(f"- [{mtype}] {content} (importance: {importance})")

    lines.append("")
    lines.append(MCP_SYNC_END)
    return lines


def _strip_mcp_section(content: str) -> str:
    """Remove the MCP-SYNCED fenced section from content."""
    pattern = re.compile(
        re.escape(MCP_SYNC_START) + r".*?" + re.escape(MCP_SYNC_END),
        re.DOTALL,
    )
    stripped = pattern.sub("", content)
    # Clean up extra blank lines left behind
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.rstrip("\n") + "\n" if stripped.strip() else ""


def _replace_mcp_section(user_content: str, section_text: str) -> str:
    """Append (or replace) the MCP section at the end of user content."""
    # Ensure user content ends with a newline
    if user_content and not user_content.endswith("\n"):
        user_content += "\n"

    return user_content + "\n" + section_text + "\n"


# ── Direction B: Native -> MCP ────────────────────────────────────────────


async def sync_native_to_mcp(
    db,
    embeddings,
    project_path: str,
) -> Dict[str, Any]:
    """Sync native MEMORY.md content into the MCP vector DB.

    1. Read native MEMORY.md + topic files
    2. Parse into entries (## sections or bullet groups)
    3. Skip the MCP-SYNCED section (avoid circular sync)
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

        # Strip the MCP-synced section to avoid circular import
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
            # Generate embedding
            emb_result = await embeddings.generate_embedding(entry["content"])
            if hasattr(emb_result, "ok") and not emb_result.ok:
                logger.warning(f"Native->MCP: embedding failed for entry: {emb_result.error_message}")
                errors.append(entry["content"][:50])
                continue

            # Extract the embedding vector
            embedding = emb_result.embedding if hasattr(emb_result, "embedding") else emb_result

            # Build tags
            tags = [NATIVE_SOURCE_TAG]
            if entry.get("source_file"):
                tags.append(f"file={entry['source_file']}")
            if entry.get("section"):
                tags.append(f"section={entry['section']}")

            # Store in MCP DB
            import json
            embedding_json = json.dumps(
                embedding if isinstance(embedding, list) else embedding.tolist()
                if hasattr(embedding, "tolist") else list(embedding)
            )

            # Use only columns guaranteed to exist in the active DB schema
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

            # Record in markdown_syncs
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
    current_lines = []

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
            # Top-level heading, skip (it's the file title)
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


# ── Combined sync ─────────────────────────────────────────────────────────


async def sync_bidirectional(
    db,
    embeddings,
    project_path: str,
) -> Dict[str, Any]:
    """Run both sync directions. Used at session end.

    Args:
        db: DatabaseService instance
        embeddings: EmbeddingService instance (needed for native->MCP)
        project_path: Absolute project path

    Returns:
        Dict with results from both directions
    """
    results = {}

    # Direction A: MCP -> Native (fast)
    try:
        results["mcp_to_native"] = await sync_mcp_to_native(db, project_path)
    except Exception as e:
        logger.error(f"Bidirectional sync MCP->Native failed: {e}")
        results["mcp_to_native"] = {"success": False, "error": str(e)}

    # Direction B: Native -> MCP (needs embeddings)
    try:
        results["native_to_mcp"] = await sync_native_to_mcp(db, embeddings, project_path)
    except Exception as e:
        logger.error(f"Bidirectional sync Native->MCP failed: {e}")
        results["native_to_mcp"] = {"success": False, "error": str(e)}

    return results
