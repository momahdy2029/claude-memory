# Claude Memory System

A comprehensive memory and anti-hallucination layer for Claude Code with automatic timeline tracking, causal chain logging, and persistent context management.

## Features

### 🧠 Memory Layer
- **Semantic search** using Ollama embeddings (nomic-embed-text)
- **Cross-project learning** - patterns that worked elsewhere surface automatically
- **Decision tracking** with confidence scores and reasoning
- **Anchor system** - verified facts that Claude cannot contradict

### 📊 Timeline & Causal Chains
- **Automatic event logging** - every user request, tool use, and Claude response
- **Causal chain linking** - see how decisions flow from user requests
- **Event types**: user_request → decision → observation → action → outcome → anchor
- **Visual timeline** in the dashboard with tree view

### 🛡️ Anti-Hallucination System
- **Grounding context injection** - relevant facts injected before every response
- **Entity registry** - tracks correct file paths to prevent confusion
- **Anchor verification** - blocks actions that contradict verified facts
- **Staleness warnings** - alerts when context is old

### 🚀 Auto-Start
- **Zero manual setup** - everything starts when you use Claude Code
- **Ollama** - auto-starts if not running
- **Memory Agent** - auto-starts on port 8102
- **Dashboard** - opens in browser automatically

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Claude Code                                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐    ┌───────────────────┐    ┌──────────────────┐
│ UserPrompt    │    │ PostToolUse       │    │ Stop             │
│ Submit Hook   │    │ Hook              │    │ Hook             │
├───────────────┤    ├───────────────────┤    ├──────────────────┤
│ • Log request │    │ • Log tool usage  │    │ • Extract        │
│ • Inject      │    │ • Link to request │    │   decisions      │
│   grounding   │    │ • Track files     │    │ • Extract        │
│ • Auto-start  │    │                   │    │   observations   │
│   services    │    │                   │    │ • Create anchors │
└───────┬───────┘    └─────────┬─────────┘    └────────┬─────────┘
        │                      │                       │
        └──────────────────────┼───────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   Memory Agent      │
                    │   (Port 8102)       │
                    ├─────────────────────┤
                    │ • Timeline DB       │
                    │ • Session State     │
                    │ • Semantic Search   │
                    │ • Dashboard         │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                                 ▼
    ┌─────────────────┐              ┌─────────────────┐
    │     Ollama      │              │     SQLite      │
    │ • Embeddings    │              │ • memories.db   │
    │ • LLM Analysis  │              │ • Timeline      │
    │   (llama3.2:3b) │              │ • Sessions      │
    └─────────────────┘              └─────────────────┘
```

## Prerequisites

1. **Python 3.10+**
2. **Ollama** with models:
   - `nomic-embed-text` (embeddings)
   - `llama3.2:3b` (insight extraction)

## Quick Setup

### 1. Install Ollama Models

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

### 2. Set Up Memory Agent

```powershell
cd memory-agent
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Hooks

Copy the hooks to your Claude config:

```powershell
# Copy hooks to ~/.claude/hooks/
Copy-Item -Path "C:\Users\moham\.claude\hooks\*" -Destination "$env:USERPROFILE\.claude\hooks\" -Recurse
```

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {"type": "command", "command": "python ~/.claude/hooks/enhanced-timeline.py"},
          {"type": "command", "command": "python ~/.claude/hooks/grounding-hook.py"}
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write|Bash|Read|Glob|Grep",
        "hooks": [
          {"type": "command", "command": "python ~/.claude/hooks/enhanced-timeline.py"}
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {"type": "command", "command": "python ~/.claude/hooks/extract-insights.py"}
        ]
      }
    ]
  }
}
```

### 4. Start Using

Just start Claude Code! Everything auto-starts on your first message.

Or manually: `start-memory-agent.bat`

## Dashboard

Access at: **http://localhost:8102/dashboard**

Features:
- **Project selector** - switch between projects
- **Timeline view** - see causal chains as tree structure
- **Hook configuration** - enable/disable hooks
- **Memory stats** - view stored memories and patterns
- **Session state** - current goal, entity registry, anchors

## Timeline Event Types

| Type | Description | Example |
|------|-------------|---------|
| `user_request` | User's message (root of chain) | "Fix the login bug" |
| `decision` | Claude's choices with reasoning | "Use JWT instead of sessions" |
| `observation` | Findings during analysis | "Found null check missing" |
| `action` | Tool usage (Read, Edit, Bash) | "Edited auth.py" |
| `outcome` | Result of actions | "SUCCESS - Bug fixed" |
| `anchor` | Verified facts (cannot contradict) | "Config is in /etc/app.conf" |

## Hooks

| Hook | Trigger | Purpose |
|------|---------|---------|
| `grounding-hook.py` | UserPromptSubmit | Inject context, auto-start services |
| `enhanced-timeline.py` | UserPromptSubmit, PostToolUse | Log events to timeline |
| `extract-insights.py` | Stop | Use LLM to extract decisions/observations |
| `detect-correction.py` | UserPromptSubmit | Detect when user corrects Claude |
| `pre-tool-check.py` | PreToolUse | Verify actions against anchors |

## MCP Tools

Available when memory agent is running:

| Tool | Purpose |
|------|---------|
| `memory_store` | Store memories with embeddings |
| `memory_search` | Semantic search across memories |
| `memory_context` | Load session context |
| `memory_search_patterns` | Find solution patterns |

## Anti-Hallucination Protocol

Add to your `~/.claude/CLAUDE.md`:

```markdown
## Anti-Hallucination Protocol (MANDATORY)
Before ANY file edit or significant action, I MUST:
1. Check entity registry for correct file path
2. Verify action doesn't contradict anchors
3. State which registered entity I'm using and why
4. If no entity registered, register it first
5. If uncertain, verify with context_refresh
```

## File Locations

| Component | Location |
|-----------|----------|
| Memory Agent | `./memory-agent/` |
| Active Hooks | `~/.claude/hooks/` |
| Database | `./memory-agent/memories.db` |
| Dashboard | `http://localhost:8102/dashboard` |
| Logs | `./memory-agent/memory-agent.log` |

## Configuration

### Memory Agent
- **Port**: 8102 (set via `PORT` env var)
- **Database**: SQLite (memories.db)
- **Ollama**: localhost:11434

### Hooks
All hooks read `MEMORY_AGENT_URL` env var (default: `http://localhost:8102`)

## Troubleshooting

### Services won't auto-start
1. Check `~/.claude/hooks/grounding-hook.py` has correct `AGENT_DIR` path
2. Verify Python is in PATH
3. Check `memory-agent/startup-error.log`

### Timeline not showing events
1. Verify hooks are configured in `~/.claude/settings.json`
2. Check memory agent is running: `curl http://localhost:8102/health`
3. Select correct project in dashboard

### Insights not extracting
1. Verify Ollama is running: `curl http://localhost:11434/api/tags`
2. Ensure `llama3.2:3b` model is pulled
3. Check `extract-insights.py` is in Stop hooks

### Settings file errors (extraKnownMarketplaces)
If you see an error like:
```
extraKnownMarketplaces
  └ claude-code-lsps: Expected object, but received boolean
```

This means a marketplace is configured incorrectly. The value must be an **object**, not a boolean:

```json
// ❌ Wrong
"extraKnownMarketplaces": {
  "claude-code-lsps": true
}

// ✅ Correct
"extraKnownMarketplaces": {
  "claude-code-lsps": {
    "source": {
      "source": "github",
      "repo": "Piebald-AI/claude-code-lsps"
    }
  }
}
```

## Updating from Previous Versions

If you have an older installation of the Claude Memory System, use the built-in update script to migrate your database and configuration to the latest version.

### Check Current Version

```powershell
cd memory-agent
python update_system.py --status
```

This shows:
- Your current detected version
- Latest available version
- Database table counts
- Any issues that need fixing (like path format inconsistencies)

### Run the Update

```powershell
cd memory-agent
python update_system.py
```

**Options:**
- `--dry-run` - Preview changes without modifying anything
- `--verbose` - Show detailed SQL operations
- `--status` - Check status only (no changes)

### What the Update Does

The update script automatically:

1. **Detects your version** based on database schema structure
2. **Creates a backup** (`memories.db.backup_YYYYMMDD_HHMMSS`) before any changes
3. **Creates missing tables** for features added in newer versions
4. **Adds missing columns** to existing tables
5. **Normalizes file paths** - fixes Windows backslash/forward slash inconsistencies
6. **Handles duplicates** safely when paths create conflicts
7. **Creates performance indexes** for faster queries
8. **Records the version** in the database for future updates

### Version History

| Version | Features Added |
|---------|----------------|
| 1.0.0 | Initial release - memories table |
| 1.1.0 | Patterns table for solution tracking |
| 1.2.0 | Timeline events and session state |
| 1.3.0 | Project configurations (agents, MCP, hooks) |
| 1.4.0 | Insights extraction and memory archive |
| 1.5.0 | Anchor conflict tracking and history |
| 2.0.0 | Path normalization fix, cleanup system |
| 2.1.0 | Current - full feature set, system info tracking |

### Example Update Output

```
============================================================
         Claude Memory System - Update Script
============================================================

[INFO] Detected version: 1.3.0
[INFO] Target version: 2.1.0

[STEP] Creating backup: memories.db.backup_20260130_125539
[OK] Backup created successfully

============================================================
               Running Migrations
============================================================

[STEP] Creating insights table...
[STEP] Creating insight_feedback table...
[STEP] Creating memory_archive table...
[STEP] Creating anchor_conflicts table...
[STEP] Creating anchor_history table...
[STEP] Creating cleanup_config table...
[STEP] Adding column memories.embedding_model...
[STEP] Normalizing paths in all tables...
[INFO]   Normalizing 35 paths in memories.project_path
[INFO]   Normalizing 137 paths in timeline_events.project_path
[STEP] Creating index idx_memories_project...

============================================================
                 Migration Summary
============================================================

[OK] Completed 15 migrations:

  1. Created insights table
  2. Created insight_feedback table
  3. Created memory_archive table
  4. Created anchor_conflicts table
  5. Created anchor_history table
  6. Created cleanup_config table
  7. Added memories.embedding_model column
  8. Normalized 35 paths in memories.project_path
  9. Normalized 137 paths in timeline_events.project_path
  10. Created index idx_memories_project
  ...

Update completed successfully!
  From version: 1.3.0
  To version: 2.1.0

  Backup saved: memories.db.backup_20260130_125539
```

### Restoring from Backup

If something goes wrong, restore from the backup:

```powershell
cd memory-agent
# Stop the server first
taskkill /F /PID (Get-Content memory-agent.pid)

# Restore the backup
Copy-Item memories.db.backup_YYYYMMDD_HHMMSS memories.db -Force

# Restart the server
python main.py
```

### Manual Migration (Advanced)

If the automatic update fails, you can manually fix common issues:

**Fix path normalization:**
```sql
-- In SQLite
UPDATE memories SET project_path = REPLACE(project_path, '\', '/');
UPDATE timeline_events SET project_path = REPLACE(project_path, '\', '/');
UPDATE session_state SET project_path = REPLACE(project_path, '\', '/');
```

**Check for issues:**
```sql
-- Find paths with backslashes
SELECT COUNT(*) FROM memories WHERE project_path LIKE '%\%';
SELECT COUNT(*) FROM timeline_events WHERE project_path LIKE '%\%';
```

## License

MIT
