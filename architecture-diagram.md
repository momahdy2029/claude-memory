# Slim MCP Proxy Architecture

## High-Level Flow

```
+------------------------------------------------------------------+
|                        CLAUDE CODE (Main Chat)                    |
|                                                                   |
|  Context Window Budget:                                           |
|  +------------------------------------------------------------+  |
|  | MCP Tool Definitions: ~300 tok (was ~2,500)                 |  |
|  |   - memory_ask(query, type_hint?, project_path?, limit?)    |  |
|  |   - memory_store(content, memory_type, importance, ...)     |  |
|  |   - memory_status(project_path?)                            |  |
|  +------------------------------------------------------------+  |
|  | Grounding Hook Injection: ~50-150 tok (was ~375-1,150)      |  |
|  |   [MEM] goal: Fix auth | 3 anchors | sessions: frontend    |  |
|  +------------------------------------------------------------+  |
|  | Native MEMORY.md: ~500-2,000 tok (unchanged, Claude-owned)  |  |
|  +------------------------------------------------------------+  |
|                                                                   |
|  Total: ~850-2,450 tok/prompt (was ~3,375-5,650)                 |
+------------------------------------------------------------------+
        |                    |                      |
        | MCP stdio          | HTTP (hook)          | Auto-loaded
        | JSON-RPC           | Single POST          | at session start
        v                    v                      v
+----------------+  +-------------------+  +------------------+
| mcp_proxy.py   |  | grounding-v2.py   |  | MEMORY.md        |
| (stdio server) |  | (UserPromptSubmit)|  | (native auto     |
|                |  |                   |  |  memory, 200     |
| NO embedding   |  | 3-sec timeout     |  |  lines max)      |
| NO database    |  | Silent fail       |  |                  |
| Just HTTP      |  | Compact output    |  | Claude Code owns |
| forwarding     |  |                   |  | exclusively      |
+-------+--------+  +---------+---------+  +--------+---------+
        |                     |                      |
        | httpx async         | requests POST        | Stop hook
        | 5-sec timeout       |                      | (sync_native_to_mcp)
        |                     |                      |
        v                     v                      v
+==================================================================+
|                     main.py :8102                                 |
|                  (HTTP Backend Server)                            |
|                                                                   |
|  +-----------------------+  +-------------------------------+    |
|  | REST API Endpoints    |  | A2A Skill Dispatch (70+ skills)|   |
|  |                       |  |                               |    |
|  | GET /api/search       |  | store_memory                  |    |
|  | GET /api/stats        |  | store_pattern                 |    |
|  | GET /api/patterns     |  | store_project                 |    |
|  | GET /api/sessions/*   |  | search_patterns               |    |
|  | POST /api/grounding-  |  | get_project_context           |    |
|  |       context  [NEW]  |  | context_refresh               |    |
|  |                       |  | curator_get_status             |    |
|  +-----------------------+  | session_heartbeat              |    |
|                              | ... 60+ more                  |    |
|  +-------------------------+ +-------------------------------+    |
|  | Services                                                  |    |
|  |  - DatabaseService (SQLite + WAL)                         |    |
|  |  - EmbeddingService (sentence-transformers)               |    |
|  |  - SessionAwareness (cross-session tracking)              |    |
|  |  - Curator (dedup, orphans, graph)                        |    |
|  |  - NativeMemorySync (one-way: MEMORY.md -> vector DB)     |    |
|  +-----------------------------------------------------------+    |
+==================================================================+


## Hook Architecture (UserPromptSubmit)

BEFORE (7 hooks, 4-6 HTTP calls):            AFTER (3 hooks, 1 HTTP call):

+---------------------------+                +---------------------------+
| 1. session_start.py       | REMOVED        | 1. grounding-hook-v2.py   |
|    - 6+ A2A calls         |-------+        |    - 1 POST to            |
|    - loads full context    |       |        |      /api/grounding-      |
+---------------------------+       |        |      context              |
| 2. debug-hook.py          | REMOVED        |    - aggregates all       |
|    - writes to log file   |-------+        |      server-side          |
+---------------------------+       |        +---------------------------+
| 3. detect-correction.py   | KEPT   ------->| 2. detect-correction.py   |
|    - lightweight regex    |                |    - lightweight regex     |
+---------------------------+                +---------------------------+
| 4. problem-detector.py    | REMOVED        | 3. enhanced-timeline.py   |
|    - pattern matching     |-------+        |    - timeline logging      |
+---------------------------+       |        +---------------------------+
| 5. memory-first-reminder  | REMOVED
|    - A2A pattern search   |-------+
+---------------------------+       |
| 6. enhanced-timeline.py   | KEPT   ------->  (see above)
|    - timeline logging     |
+---------------------------+
| 7. grounding-hook.py      | REPLACED
|    - 4-6 HTTP calls       |-------+
|    - verbose output       |       |
+---------------------------+       |
                                    |
                  All replaced by grounding-hook-v2.py
                  which calls /api/grounding-context once


## /api/grounding-context Endpoint (NEW)

Single POST runs 4 queries in parallel via asyncio.gather:

                POST /api/grounding-context
                {session_id, project_path, user_input}
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
    context_refresh   heartbeat    search_patterns   curator_status
    (anchors, goal,   (parallel    (if user_input    (orphan count,
     contradictions)   sessions,    >10 chars,        connectivity)
                       conflicts)   top 2 matches)
              |             |             |               |
              +------+------+------+------+               |
                     |             |                       |
                     v             v                       v
              +-------------------------------------------------+
              | Compact output (<150 tokens):                    |
              | [MEM] goal: Fix auth | 3 anchors |              |
              |       sessions: frontend | pattern(85%): JWT    |
              +-------------------------------------------------+


## Data Flow: memory_ask (Proxy Tool)

  Claude calls: memory_ask("auth error", type_hint="error")
                            |
                            v
                    +----------------+
                    | mcp_proxy.py   |
                    +-------+--------+
                            |
              +-------------+-------------+
              |                           |
              v                           v
    GET /api/search               A2A search_patterns
    ?query=auth+error             {query: "auth error",
    &memory_type=error             limit: 5}
    &limit=10
              |                           |
              v                           v
        {memories: [...]}          {patterns: [...]}
              |                           |
              +-------------+-------------+
                            |
                            v
                  Merged JSON response:
                  {
                    query: "auth error",
                    memories: [...top 10...],
                    patterns: [...top 5...],
                    success: true
                  }


## Data Flow: memory_store (Proxy Tool)

  memory_store("Use JWT for auth", memory_type="decision", importance=8)
       |
       |  (no pattern_name, no project_type -> routes to store_memory)
       v
  A2A store_memory {content, memory_type, importance, ...}
       |
       v
  main.py -> skills/store.py -> embeddings + SQLite INSERT
       |
       v
  PostToolUse hook fires: sync-native-memory.py
       |
       v
  sync_native_to_mcp() runs at Stop hook (ingests MEMORY.md -> vector DB)


## Native Memory Sync (Simplified)

BEFORE (bidirectional):              AFTER (one-way):

  MCP Vector DB                       MCP Vector DB
       |                                   ^
       | sync_mcp_to_native                |
       | (importance >= 7)                 | sync_native_to_mcp
       v                                   | (Stop hook)
  MEMORY.md <--- Claude Code          MEMORY.md <--- Claude Code
       |         writes here                       owns exclusively
       | sync_native_to_mcp
       | (session end)
       v
  MCP Vector DB

  REMOVED: MCP -> Native direction
  REMOVED: <!-- MCP-SYNCED --> markers
  KEPT:    Native -> MCP (MEMORY.md content becomes searchable vectors)


## Optional: Team Mode Agent

  +------------------+     SendMessage     +------------------+
  | Main Chat        | ------------------> | memory-agent     |
  | (team lead)      |                     | (teammate)       |
  |                  | <------------------ |                  |
  | No memory tools  |     Results         | Has:             |
  | in context       |                     |  - memory_ask    |
  +------------------+                     |  - memory_store  |
                                           |  - memory_status |
  Only spawned on demand.                  |  - Read          |
  Zero context cost unless active.         +------------------+
