#!/usr/bin/env python3
"""
Claude Memory System - Update & Migration Script
================================================
This script detects the current installation state and performs
all necessary migrations to bring it to the latest version.

Usage:
    python update_system.py [--dry-run] [--verbose]

Options:
    --dry-run   Show what would be done without making changes
    --verbose   Show detailed progress information
"""

import sqlite3
import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

# Version history and required migrations
VERSION_HISTORY = {
    "1.0.0": "Initial release - basic memories table",
    "1.1.0": "Added patterns table",
    "1.2.0": "Added timeline_events and session_state",
    "1.3.0": "Added project configurations (agent, mcp, hook configs)",
    "1.4.0": "Added insights and memory_archive",
    "1.5.0": "Added anchor_conflicts and anchor_history",
    "2.0.0": "Path normalization fix, cleanup system",
    "2.1.0": "Full feature set with confidence scoring",
    "2.2.0": "Outcome spectrum (pending/success/partial/failed/superseded)",
    "2.3.0": "Context tagging (worked_in/failed_in/context_confidence)",
    "2.4.0": "CLaRa memory tiers, consolidation, markdown sync",
    "2.5.0": "Cross-session awareness, MCP server, knowledge graph",
    "3.0.0": "Slim MCP proxy, soul layer tables",
    "3.1.0": "Database migration safety - user data in ~/.claude-memory/",
}

CURRENT_VERSION = "3.1.0"

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_step(text):
    print(f"{Colors.CYAN}[STEP]{Colors.ENDC} {text}")

def print_success(text):
    print(f"{Colors.GREEN}[OK]{Colors.ENDC} {text}")

def print_warning(text):
    print(f"{Colors.YELLOW}[WARN]{Colors.ENDC} {text}")

def print_error(text):
    print(f"{Colors.RED}[ERROR]{Colors.ENDC} {text}")

def print_info(text):
    print(f"{Colors.BLUE}[INFO]{Colors.ENDC} {text}")


class MigrationManager:
    def __init__(self, db_path: str, dry_run: bool = False, verbose: bool = False):
        self.db_path = db_path
        self.dry_run = dry_run
        self.verbose = verbose
        self.conn = None
        self.cursor = None
        self.migrations_run = []
        self.detected_version = None

    def connect(self):
        """Connect to the database"""
        if not os.path.exists(self.db_path):
            print_error(f"Database not found: {self.db_path}")
            return False
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        return True

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def get_tables(self) -> list:
        """Get list of all tables"""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [r[0] for r in self.cursor.fetchall()]

    def get_columns(self, table: str) -> dict:
        """Get columns for a table"""
        self.cursor.execute(f"PRAGMA table_info({table})")
        return {r[1]: r[2] for r in self.cursor.fetchall()}

    def table_exists(self, table: str) -> bool:
        """Check if a table exists"""
        return table in self.get_tables()

    def column_exists(self, table: str, column: str) -> bool:
        """Check if a column exists in a table"""
        if not self.table_exists(table):
            return False
        return column in self.get_columns(table)

    def detect_version(self) -> str:
        """Detect current installation version based on database structure"""
        tables = self.get_tables()

        # Check from newest to oldest features
        if 'anchor_history' in tables and 'cleanup_config' in tables:
            # Check for latest column additions
            if self.column_exists('memories', 'embedding_model'):
                return "2.1.0"
            return "2.0.0"

        if 'anchor_conflicts' in tables:
            return "1.5.0"

        if 'insights' in tables or 'memory_archive' in tables:
            return "1.4.0"

        if 'project_agent_config' in tables:
            return "1.3.0"

        if 'timeline_events' in tables:
            return "1.2.0"

        if 'patterns' in tables:
            return "1.1.0"

        if 'memories' in tables:
            return "1.0.0"

        return "0.0.0"  # Fresh install

    def backup_database(self):
        """Create a backup of the database before migration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.db_path}.backup_{timestamp}"

        if self.dry_run:
            print_info(f"Would backup database to: {backup_path}")
            return backup_path

        print_step(f"Creating backup: {backup_path}")
        shutil.copy2(self.db_path, backup_path)
        print_success("Backup created successfully")
        return backup_path

    def execute(self, sql: str, params: tuple = None):
        """Execute SQL with dry-run support"""
        if self.verbose:
            print_info(f"SQL: {sql[:100]}...")

        if self.dry_run:
            return

        if params:
            self.cursor.execute(sql, params)
        else:
            self.cursor.execute(sql)

    def commit(self):
        """Commit changes with dry-run support"""
        if not self.dry_run:
            self.conn.commit()

    # =========================================
    # Migration Functions
    # =========================================

    def migrate_create_base_tables(self):
        """Create base tables if they don't exist (v1.0.0)"""
        if not self.table_exists('memories'):
            print_step("Creating memories table...")
            self.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT DEFAULT 'chunk',
                    content TEXT NOT NULL,
                    embedding TEXT,
                    project_path TEXT,
                    project_name TEXT,
                    project_type TEXT,
                    tech_stack TEXT,
                    session_id TEXT,
                    chat_id TEXT,
                    agent_type TEXT,
                    skill_used TEXT,
                    tools_used TEXT,
                    outcome TEXT,
                    success INTEGER,
                    user_feedback TEXT,
                    tags TEXT,
                    metadata TEXT,
                    importance INTEGER DEFAULT 5,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 0,
                    decay_factor REAL DEFAULT 1.0,
                    embedding_model TEXT
                )
            """)
            self.migrations_run.append("Created memories table")

    def migrate_create_patterns(self):
        """Create patterns table (v1.1.0)"""
        if not self.table_exists('patterns'):
            print_step("Creating patterns table...")
            self.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    problem_type TEXT,
                    solution TEXT NOT NULL,
                    embedding TEXT,
                    tech_context TEXT,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.migrations_run.append("Created patterns table")

    def migrate_create_timeline_session(self):
        """Create timeline and session tables (v1.2.0)"""
        if not self.table_exists('timeline_events'):
            print_step("Creating timeline_events table...")
            self.execute("""
                CREATE TABLE IF NOT EXISTS timeline_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    project_path TEXT,
                    event_type TEXT NOT NULL,
                    sequence_num INTEGER,
                    summary TEXT,
                    details TEXT,
                    embedding TEXT,
                    parent_event_id INTEGER,
                    root_event_id INTEGER,
                    entities TEXT,
                    status TEXT DEFAULT 'active',
                    outcome TEXT,
                    confidence REAL DEFAULT 1.0,
                    is_anchor INTEGER DEFAULT 0,
                    is_reversible INTEGER DEFAULT 1,
                    needs_verification INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_event_id) REFERENCES timeline_events(id)
                )
            """)
            self.migrations_run.append("Created timeline_events table")

        if not self.table_exists('session_state'):
            print_step("Creating session_state table...")
            self.execute("""
                CREATE TABLE IF NOT EXISTS session_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    project_path TEXT,
                    current_goal TEXT,
                    pending_questions TEXT,
                    entity_registry TEXT,
                    decisions_summary TEXT,
                    last_checkpoint_id INTEGER,
                    events_since_checkpoint INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_activity_at TEXT
                )
            """)
            self.migrations_run.append("Created session_state table")

        if not self.table_exists('checkpoints'):
            print_step("Creating checkpoints table...")
            self.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    project_path TEXT,
                    checkpoint_type TEXT DEFAULT 'auto',
                    state_snapshot TEXT,
                    summary TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.migrations_run.append("Created checkpoints table")

    def migrate_create_project_configs(self):
        """Create project configuration tables (v1.3.0)"""
        config_tables = [
            ('projects', """
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    name TEXT,
                    type TEXT,
                    tech_stack TEXT,
                    conventions TEXT,
                    preferences TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """),
            ('project_agent_config', """
                CREATE TABLE IF NOT EXISTS project_agent_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_path TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    config TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(project_path, agent_name)
                )
            """),
            ('project_mcp_config', """
                CREATE TABLE IF NOT EXISTS project_mcp_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_path TEXT NOT NULL,
                    server_name TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    config TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(project_path, server_name)
                )
            """),
            ('project_hook_config', """
                CREATE TABLE IF NOT EXISTS project_hook_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_path TEXT NOT NULL,
                    hook_name TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    config TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(project_path, hook_name)
                )
            """),
            ('project_preferences', """
                CREATE TABLE IF NOT EXISTS project_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_path TEXT UNIQUE NOT NULL,
                    preferences TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """),
        ]

        for table_name, create_sql in config_tables:
            if not self.table_exists(table_name):
                print_step(f"Creating {table_name} table...")
                self.execute(create_sql)
                self.migrations_run.append(f"Created {table_name} table")

    def migrate_create_insights_archive(self):
        """Create insights and archive tables (v1.4.0)"""
        if not self.table_exists('insights'):
            print_step("Creating insights table...")
            self.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_path TEXT,
                    insight_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_memories TEXT,
                    confidence REAL DEFAULT 0.5,
                    status TEXT DEFAULT 'pending',
                    user_feedback TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.migrations_run.append("Created insights table")

        if not self.table_exists('insight_feedback'):
            print_step("Creating insight_feedback table...")
            self.execute("""
                CREATE TABLE IF NOT EXISTS insight_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight_id INTEGER NOT NULL,
                    feedback_type TEXT NOT NULL,
                    comment TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (insight_id) REFERENCES insights(id)
                )
            """)
            self.migrations_run.append("Created insight_feedback table")

        if not self.table_exists('memory_archive'):
            print_step("Creating memory_archive table...")
            self.execute("""
                CREATE TABLE IF NOT EXISTS memory_archive (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_id INTEGER,
                    type TEXT,
                    content TEXT,
                    project_path TEXT,
                    importance INTEGER,
                    archive_reason TEXT,
                    archived_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    original_created_at TEXT
                )
            """)
            self.migrations_run.append("Created memory_archive table")

    def migrate_create_anchor_tables(self):
        """Create anchor tracking tables (v1.5.0)"""
        if not self.table_exists('anchor_conflicts'):
            print_step("Creating anchor_conflicts table...")
            self.execute("""
                CREATE TABLE IF NOT EXISTS anchor_conflicts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    anchor_event_id INTEGER,
                    conflicting_action TEXT,
                    resolution TEXT,
                    resolved INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.migrations_run.append("Created anchor_conflicts table")

        if not self.table_exists('anchor_history'):
            print_step("Creating anchor_history table...")
            self.execute("""
                CREATE TABLE IF NOT EXISTS anchor_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_id INTEGER,
                    action TEXT NOT NULL,
                    previous_state TEXT,
                    new_state TEXT,
                    reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.migrations_run.append("Created anchor_history table")

    def migrate_create_cleanup_tables(self):
        """Create cleanup system tables (v2.0.0)"""
        if not self.table_exists('cleanup_config'):
            print_step("Creating cleanup_config table...")
            self.execute("""
                CREATE TABLE IF NOT EXISTS cleanup_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_key TEXT UNIQUE NOT NULL,
                    config_value TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.migrations_run.append("Created cleanup_config table")

        if not self.table_exists('cleanup_log'):
            print_step("Creating cleanup_log table...")
            self.execute("""
                CREATE TABLE IF NOT EXISTS cleanup_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cleanup_type TEXT NOT NULL,
                    items_processed INTEGER DEFAULT 0,
                    items_removed INTEGER DEFAULT 0,
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.migrations_run.append("Created cleanup_log table")

    def migrate_add_missing_columns(self):
        """Add any missing columns to existing tables"""
        # Memories table columns
        memories_columns = {
            'embedding_model': 'TEXT',
            'decay_factor': 'REAL DEFAULT 1.0',
            'access_count': 'INTEGER DEFAULT 0',
            'last_accessed': 'TEXT',
            'skill_used': 'TEXT',
            'chat_id': 'TEXT',
            'confidence': 'REAL DEFAULT 0.5',
            # Outcome spectrum columns (v2.2.0)
            'outcome_status': "TEXT DEFAULT 'pending'",
            'fixed': 'TEXT',  # JSON array
            'did_not_fix': 'TEXT',  # JSON array
            'caused': 'TEXT',  # JSON array
            'superseded_by': 'INTEGER',  # FK to memories.id
        }

        for col, col_type in memories_columns.items():
            if not self.column_exists('memories', col):
                print_step(f"Adding column memories.{col}...")
                self.execute(f"ALTER TABLE memories ADD COLUMN {col} {col_type}")
                self.migrations_run.append(f"Added memories.{col} column")

        # Session state columns
        session_columns = {
            'last_activity_at': 'TEXT',
            'last_flush_at': 'TEXT',
        }

        for col, col_type in session_columns.items():
            if self.table_exists('session_state') and not self.column_exists('session_state', col):
                print_step(f"Adding column session_state.{col}...")
                self.execute(f"ALTER TABLE session_state ADD COLUMN {col} {col_type}")
                self.migrations_run.append(f"Added session_state.{col} column")

    def migrate_normalize_paths(self):
        """Normalize all paths to use forward slashes (v2.0.0 fix)"""
        print_step("Normalizing paths in all tables...")

        # Tables without unique constraints on path - safe to update directly
        simple_tables = [
            ('memories', 'project_path'),
            ('session_state', 'project_path'),
            ('timeline_events', 'project_path'),
            ('checkpoints', 'project_path'),
            ('insights', 'project_path'),
            ('memory_archive', 'project_path'),
        ]

        # Tables with unique constraints - need special handling
        unique_tables = [
            ('projects', 'path'),
            ('project_agent_config', 'project_path'),
            ('project_mcp_config', 'project_path'),
            ('project_hook_config', 'project_path'),
            ('project_preferences', 'project_path'),
        ]

        # Process simple tables first
        for table, column in simple_tables:
            if self.table_exists(table) and self.column_exists(table, column):
                self.cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} LIKE '%\\%'")
                count = self.cursor.fetchone()[0]

                if count > 0:
                    print_info(f"  Normalizing {count} paths in {table}.{column}")
                    if not self.dry_run:
                        self.execute(f"""
                            UPDATE {table}
                            SET {column} = REPLACE({column}, '\\', '/')
                            WHERE {column} LIKE '%\\%'
                        """)
                    self.migrations_run.append(f"Normalized {count} paths in {table}.{column}")

        # Process tables with unique constraints - delete duplicates first
        for table, column in unique_tables:
            if self.table_exists(table) and self.column_exists(table, column):
                self.cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} LIKE '%\\%'")
                count = self.cursor.fetchone()[0]

                if count > 0:
                    print_info(f"  Normalizing {count} paths in {table}.{column} (handling duplicates)")
                    if not self.dry_run:
                        # Find paths that would create duplicates after normalization
                        self.cursor.execute(f"""
                            SELECT {column}, REPLACE({column}, '\\', '/') as normalized
                            FROM {table}
                            WHERE {column} LIKE '%\\%'
                        """)
                        to_normalize = self.cursor.fetchall()

                        duplicates_removed = 0
                        for row in to_normalize:
                            old_path = row[0]
                            new_path = row[1]

                            # Check if normalized path already exists
                            self.cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (new_path,))
                            exists = self.cursor.fetchone()[0] > 0

                            if exists:
                                # Delete the row with backslashes (keep the one with forward slashes)
                                self.execute(f"DELETE FROM {table} WHERE {column} = ?", (old_path,))
                                duplicates_removed += 1
                            else:
                                # Safe to update
                                self.execute(f"UPDATE {table} SET {column} = ? WHERE {column} = ?", (new_path, old_path))

                        if duplicates_removed > 0:
                            print_info(f"    Removed {duplicates_removed} duplicate entries")

                    self.migrations_run.append(f"Normalized paths in {table}.{column}")

    def migrate_create_indexes(self):
        """Create indexes for performance"""
        indexes = [
            ('idx_memories_project', 'memories', 'project_path'),
            ('idx_memories_type', 'memories', 'type'),
            ('idx_memories_created', 'memories', 'created_at'),
            ('idx_memories_importance', 'memories', 'importance'),
            ('idx_memories_outcome_status', 'memories', 'outcome_status'),
            ('idx_memories_superseded_by', 'memories', 'superseded_by'),
            ('idx_timeline_session', 'timeline_events', 'session_id'),
            ('idx_timeline_project', 'timeline_events', 'project_path'),
            ('idx_timeline_type', 'timeline_events', 'event_type'),
            ('idx_session_project', 'session_state', 'project_path'),
            ('idx_patterns_type', 'patterns', 'problem_type'),
        ]

        # Get existing indexes
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        existing = [r[0] for r in self.cursor.fetchall()]

        for idx_name, table, column in indexes:
            if self.table_exists(table) and self.column_exists(table, column):
                if idx_name not in existing:
                    print_step(f"Creating index {idx_name}...")
                    self.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})")
                    self.migrations_run.append(f"Created index {idx_name}")

    def migrate_set_version(self):
        """Store the current version in the database"""
        # Create system_info table if it doesn't exist
        if not self.table_exists('system_info'):
            print_step("Creating system_info table...")
            self.execute("""
                CREATE TABLE IF NOT EXISTS system_info (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.migrations_run.append("Created system_info table")

        # Store version
        self.execute("""
            INSERT OR REPLACE INTO system_info (key, value, updated_at)
            VALUES ('version', ?, CURRENT_TIMESTAMP)
        """, (CURRENT_VERSION,))
        self.migrations_run.append(f"Set system version to {CURRENT_VERSION}")

    def run_migrations(self):
        """Run all necessary migrations"""
        print_header("Claude Memory System - Update Script")

        if self.dry_run:
            print_warning("DRY RUN MODE - No changes will be made\n")

        # Connect to database
        if not self.connect():
            return False

        # Detect current version
        self.detected_version = self.detect_version()
        print_info(f"Detected version: {self.detected_version}")
        print_info(f"Target version: {CURRENT_VERSION}\n")

        # Check if any maintenance is needed even at current version
        needs_maintenance = False

        # Check for un-normalized paths
        for table in ['memories', 'session_state', 'timeline_events']:
            if self.table_exists(table):
                self.cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE project_path LIKE '%\\%'")
                count = self.cursor.fetchone()[0]
                if count > 0:
                    needs_maintenance = True
                    break

        if self.detected_version == CURRENT_VERSION and not needs_maintenance:
            print_success("System is already up to date!")
            self.close()
            return True

        if self.detected_version == CURRENT_VERSION:
            print_warning("Version is current but maintenance is needed\n")

        # Create backup
        backup_path = self.backup_database()

        try:
            # Run migrations in order
            print_header("Running Migrations")

            # v1.0.0 - Base tables
            self.migrate_create_base_tables()

            # v1.1.0 - Patterns
            self.migrate_create_patterns()

            # v1.2.0 - Timeline & Session
            self.migrate_create_timeline_session()

            # v1.3.0 - Project configs
            self.migrate_create_project_configs()

            # v1.4.0 - Insights & Archive
            self.migrate_create_insights_archive()

            # v1.5.0 - Anchor tables
            self.migrate_create_anchor_tables()

            # v2.0.0 - Cleanup system
            self.migrate_create_cleanup_tables()

            # Add missing columns
            self.migrate_add_missing_columns()

            # Normalize paths (critical fix)
            self.migrate_normalize_paths()

            # Create indexes
            self.migrate_create_indexes()

            # Set version
            self.migrate_set_version()

            # Commit all changes
            self.commit()

            # Print summary
            print_header("Migration Summary")

            if self.migrations_run:
                print_success(f"Completed {len(self.migrations_run)} migrations:\n")
                for i, migration in enumerate(self.migrations_run, 1):
                    print(f"  {i}. {migration}")
            else:
                print_info("No migrations were necessary")

            print(f"\n{Colors.GREEN}Update completed successfully!{Colors.ENDC}")
            print(f"  From version: {self.detected_version}")
            print(f"  To version: {CURRENT_VERSION}")

            if backup_path and not self.dry_run:
                print(f"\n  Backup saved: {backup_path}")

            return True

        except Exception as e:
            print_error(f"Migration failed: {e}")
            print_info(f"Database backup is available at: {backup_path}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            self.close()

    def show_status(self):
        """Show current system status without making changes"""
        print_header("Claude Memory System - Status Check")

        if not self.connect():
            return

        self.detected_version = self.detect_version()
        tables = self.get_tables()

        print_info(f"Database: {self.db_path}")
        print_info(f"Detected version: {self.detected_version}")
        print_info(f"Latest version: {CURRENT_VERSION}")
        print_info(f"Tables found: {len(tables)}")

        print("\nTables:")
        for table in sorted(tables):
            cols = self.get_columns(table)
            # Count rows
            self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = self.cursor.fetchone()[0]
            print(f"  - {table}: {len(cols)} columns, {count} rows")

        # Check for issues
        print("\nHealth Check:")
        issues = []

        # Check for un-normalized paths
        for table in ['memories', 'session_state', 'timeline_events']:
            if self.table_exists(table):
                self.cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE project_path LIKE '%\\%'")
                count = self.cursor.fetchone()[0]
                if count > 0:
                    issues.append(f"  - {table} has {count} paths with backslashes")

        if issues:
            print_warning("Issues found:")
            for issue in issues:
                print(issue)
            print(f"\nRun 'python update_system.py' to fix these issues.")
        else:
            print_success("No issues found!")

        self.close()


def main():
    # Parse arguments
    dry_run = '--dry-run' in sys.argv
    verbose = '--verbose' in sys.argv
    status_only = '--status' in sys.argv

    # Find database — check multiple locations in priority order
    script_dir = Path(__file__).parent
    db_path = None

    # 1. DATABASE_PATH env var (explicit user config)
    env_db = os.getenv("DATABASE_PATH")
    if env_db and Path(env_db).exists():
        db_path = Path(env_db)
        print_info(f"Using database from DATABASE_PATH env: {db_path}")

    # 2. ~/.claude-memory/memories.db (new default)
    if db_path is None:
        new_default = Path.home() / ".claude-memory" / "memories.db"
        if new_default.exists():
            db_path = new_default
            print_info(f"Using database at new default location: {db_path}")

    # 3. script_dir/memories.db (old fallback)
    if db_path is None:
        old_fallback = script_dir / "memories.db"
        if old_fallback.exists():
            db_path = old_fallback
            print_info(f"Using database at old location: {db_path}")

    if db_path is None:
        print_error("Database not found at any known location:")
        print_info(f"  - DATABASE_PATH env var: {env_db or '(not set)'}")
        print_info(f"  - {Path.home() / '.claude-memory' / 'memories.db'}")
        print_info(f"  - {script_dir / 'memories.db'}")
        sys.exit(1)

    manager = MigrationManager(str(db_path), dry_run=dry_run, verbose=verbose)

    if status_only:
        manager.show_status()
    else:
        success = manager.run_migrations()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
