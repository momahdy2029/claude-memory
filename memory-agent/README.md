# Claude Memory Agent

A persistent semantic memory system for Claude Code sessions. Stores memories, tracks decisions, and provides anti-hallucination grounding context.

## Features

- **Semantic Memory Storage** - Store and retrieve memories using natural language
- **Session Timeline** - Track decisions, actions, and observations
- **Anti-Hallucination** - Grounding context injection to prevent contradictions
- **Pattern Learning** - Store and reuse successful solution patterns
- **Dashboard** - Web UI for browsing and managing memories
- **Claude Code Integration** - Hooks for automatic session tracking

## Quick Start

### 1. Prerequisites

- Node.js 16+ (for npm installation)
- Python 3.9+
- [Ollama](https://ollama.ai/) with `nomic-embed-text` model

```bash
# Install Ollama, then pull the embedding model
ollama pull nomic-embed-text
ollama serve  # Start Ollama server
```

### 2. Installation

#### Option A: Install via npm (Recommended)

```bash
# Install globally
npm install -g claude-memory-agent

# Run the setup wizard
claude-memory-agent install

# Start the agent
claude-memory-agent start
```

#### Option B: Install via npx (No global install)

```bash
# Run directly without installing
npx claude-memory-agent install
npx claude-memory-agent start
```

#### Option C: Clone from GitHub

```bash
# Clone the repository
git clone https://github.com/yourusername/claude-memory-agent.git
cd claude-memory-agent

# Run the installation wizard
python install.py
```

The installer will:
- Install Python dependencies
- Create `.env` configuration file
- Configure Claude Code MCP settings
- Set up session tracking hooks
- Create platform-specific startup scripts

### 3. Start the Agent

**Windows:**
```bash
# Option 1: Double-click start-memory-agent.vbs
# Option 2: Use CLI
memory-agent start
```

**Mac/Linux:**
```bash
./start-memory-agent.sh
# or
python memory-agent start
```

### 4. Open Dashboard

```bash
memory-agent dashboard
# or open http://localhost:8102/dashboard
```

## CLI Commands

### If installed via npm:

```bash
claude-memory-agent install    # Run setup wizard
claude-memory-agent start      # Start agent in background
claude-memory-agent stop       # Stop running agent
claude-memory-agent status     # Check if agent is running
claude-memory-agent dashboard  # Open web dashboard
claude-memory-agent logs       # View recent log output
```

### If cloned from GitHub:

```bash
python memory-agent start      # Start agent in background
python memory-agent stop       # Stop running agent
python memory-agent status     # Check if agent is running
python memory-agent dashboard  # Open web dashboard
python memory-agent logs       # View recent log output
python memory-agent install    # Re-run installation wizard
```

## Configuration

Configuration is stored in `.env` (created during installation):

```bash
# Server
PORT=8102
HOST=0.0.0.0
MEMORY_AGENT_URL=http://localhost:8102

# Ollama / Embeddings
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text

# Database
DATABASE_PATH=./memories.db
USE_VECTOR_INDEX=true

# Logging
LOG_LEVEL=INFO
```

See `.env.example` for all available options.

## MCP Tools

The agent provides these tools to Claude Code:

| Tool | Description |
|------|-------------|
| `memory_store` | Store a memory with semantic embedding |
| `memory_search` | Search memories by natural language |
| `memory_context` | Load session context and relevant memories |
| `memory_search_patterns` | Find reusable solution patterns |
| `memory_store_pattern` | Save a successful solution pattern |
| `memory_get_project` | Get project-specific information |
| `memory_store_project` | Store project configuration |
| `memory_stats` | Get memory system statistics |
| `memory_dashboard` | Open the web dashboard |

## Project Structure

```
claude-memory-agent/
├── main.py              # FastAPI server
├── config.py            # Centralized configuration
├── install.py           # Installation wizard
├── memory-agent         # CLI tool
├── services/            # Core services
│   ├── database.py      # SQLite with connection pooling
│   ├── embeddings.py    # Ollama integration
│   ├── vector_index.py  # FAISS similarity search
│   └── websocket.py     # Real-time updates
├── skills/              # A2A skill implementations
├── hooks/               # Claude Code hooks
├── .env.example         # Configuration template
└── .gitignore           # Excludes user-specific files
```

## Uninstall

```bash
python install.py --uninstall
```

This removes Claude Code integration but preserves your data.

## Troubleshooting

**Agent won't start:**
- Check if Ollama is running: `ollama serve`
- Check if port is in use: `netstat -an | grep 8102`
- Check logs: `memory-agent logs`

**No embeddings:**
- Pull the model: `ollama pull nomic-embed-text`
- Check Ollama host in `.env`

**Hooks not working:**
- Restart Claude Code after installation
- Check `~/.claude/hooks.json` configuration

## License

MIT
