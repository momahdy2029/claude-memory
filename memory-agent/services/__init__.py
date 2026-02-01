from .database import DatabaseService
from .embeddings import EmbeddingService

# Moltbot-inspired transparency services
from .daily_log import (
    append_entry as daily_log_append,
    append_session_summary as daily_log_append_session,
    load_recent_logs as daily_log_read,
    get_today_highlights as daily_log_highlights,
    list_logs as daily_log_list,
    get_log_path
)
from .memory_md_sync import (
    sync_to_memory_md,
    read_memory_md,
    add_fact as add_memory_md_fact,
    get_memory_md_summary,
    get_memory_md_path
)
from .compaction_flush import (
    check_flush_needed,
    execute_flush as pre_compaction_flush,
    list_flushes,
    read_flush,
    get_flush_path
)

__all__ = [
    "DatabaseService",
    "EmbeddingService",
    # Daily log
    "daily_log_append",
    "daily_log_append_session",
    "daily_log_read",
    "daily_log_highlights",
    "daily_log_list",
    "get_log_path",
    # MEMORY.md
    "sync_to_memory_md",
    "read_memory_md",
    "add_memory_md_fact",
    "get_memory_md_summary",
    "get_memory_md_path",
    # Compaction flush
    "check_flush_needed",
    "pre_compaction_flush",
    "list_flushes",
    "read_flush",
    "get_flush_path",
]
