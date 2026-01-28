# Claude Memory System

A persistent semantic memory layer for Claude Code using A2A protocol, Ollama embeddings, and PostgreSQL vector storage.

## Architecture

```
┌─────────────┐       MCP        ┌─────────────────┐       A2A        ┌─────────────────┐
│ Claude Code │ ◄──────────────► │  MCP-A2A Bridge │ ◄──────────────► │  Memory Agent   │
│             │                  │  (Node.js)      │                  │  (Python)       │
└─────────────┘                  └─────────────────┘                  └────────┬────────┘
                                                                               │
                                 ┌─────────────────────────────────────────────┤
                                 │                                             │
                        ┌────────▼────────┐                           ┌────────▼────────┐
                        │     Ollama      │                           │   PostgreSQL    │
                        │ nomic-embed-text│                           │   + pgvector    │
                        └─────────────────┘                           └─────────────────┘
```

## Prerequisites

1. **PostgreSQL 17** with pgvector extension
2. **Ollama** with `nomic-embed-text` model
3. **Python 3.10+**
4. **Node.js 18+**

## Quick Setup

### 1. Install PostgreSQL (if not installed)

```powershell
winget install PostgreSQL.PostgreSQL.17
```

After installation, add to PATH:
```powershell
$env:Path += ";C:\Program Files\PostgreSQL\17\bin"
```

### 2. Create Database

```powershell
$env:PGPASSWORD = "your_password"
psql -U postgres -c "CREATE DATABASE claude_memory;"
psql -U postgres -d claude_memory -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 3. Pull Ollama Model

```powershell
ollama pull nomic-embed-text
```

### 4. Start Services

**Terminal 1 - Ollama:**
```powershell
ollama serve
```

**Terminal 2 - Memory Agent:**
```powershell
cd "<install-dir>"
.\start-memory-agent.bat
```

**Claude Code** will automatically start the MCP Bridge.

## Usage in Claude Code

Once running, you'll have access to these MCP tools:

### `memory_store`
Store information with semantic embedding.

```
memory_store(
  content: "The project uses React with TypeScript",
  type: "decision",
  tags: ["react", "typescript"]
)
```

### `memory_search`
Search memories using natural language.

```
memory_search(
  query: "What framework does the project use?",
  limit: 5,
  threshold: 0.5
)
```

### `memory_retrieve`
Get specific memories by ID or type.

```
memory_retrieve(memory_id: 42)
memory_retrieve(type: "decision", limit: 10)
```

### `memory_summarize`
Store session summaries with key decisions.

```
memory_summarize(
  summary: "Implemented user authentication with JWT",
  session_id: "session-123",
  key_decisions: ["Chose JWT over sessions", "Used bcrypt for hashing"],
  code_patterns: ["Custom useAuth hook for React"]
)
```

## Memory Types

| Type | Purpose |
|------|---------|
| `session` | End-of-session summaries |
| `decision` | Architectural choices and preferences |
| `code` | Important code patterns and snippets |
| `chunk` | General conversation chunks |

## REST API

The Memory Agent also exposes a REST API:

- `POST /api/store` - Store a memory
- `GET /api/search?query=...` - Semantic search
- `GET /api/memory/{id}` - Get specific memory
- `GET /api/stats` - Get memory statistics
- `GET /health` - Health check

## Configuration

### Memory Agent (.env)
```
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/claude_memory
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
HOST=0.0.0.0
PORT=8100
```

### MCP Bridge (.env)
```
MEMORY_AGENT_URL=http://localhost:8100
```

## File Locations

| Component | Location |
|-----------|----------|
| Memory Agent | `<install-dir>/memory-agent/` |
| MCP Bridge | `<install-dir>/mcp-bridge/` |
| Startup Script | `<install-dir>/start-memory-agent.bat` |

## Auto-Use Setup (Recommended)

The memory system works best when Claude automatically queries it. Add this to your `~/.claude/CLAUDE.md`:

```markdown
## Memory System (AUTO-USE)
- **Session start**: Call `memory_context` to load relevant project memories, decisions, and patterns
- **Before solving bugs/errors**: Call `memory_search_patterns` with the problem description
- **When encountering errors**: Call `memory_search` with type="error" to find past solutions
- **After solving significant problems**: Call `memory_store` to save the solution for future reference
- **After making architectural decisions**: Store with type="decision" and importance 7+
- These memory calls should happen automatically without user prompting
- **Before writing code**: Search memories for decisions/patterns related to the area being modified
- **Transparency**: When a solution comes from stored memory, mention it (e.g., "Based on a previous fix we stored...")
- **Learning loop**: If a stored solution doesn't work, update the memory with what actually worked
```

This makes Claude:
- Load context at session start
- Search for patterns before solving problems
- Tell you when using stored knowledge
- Save solutions for future sessions

## Troubleshooting

### Memory Agent won't start
1. Ensure PostgreSQL is running
2. Verify pgvector extension is installed
3. Check database connection string in `.env`

### Search returns no results
1. Lower the `threshold` parameter (default 0.5)
2. Verify memories were stored (use `memory_retrieve`)
3. Check Ollama is running for embeddings

### MCP tools not appearing
1. Restart Claude Code
2. Check `~/.claude.json` for memory server config
3. Verify Memory Agent is running on port 8100
