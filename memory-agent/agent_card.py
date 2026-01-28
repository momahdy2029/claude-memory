"""A2A Agent Card definition for Claude Memory Agent."""

AGENT_CARD = {
    "name": "Claude Memory Agent",
    "description": "Persistent semantic memory for Claude Code sessions with session timeline tracking for anti-hallucination. Stores memories, tracks session events, manages checkpoints, and provides grounding context.",
    "version": "2.0.0",
    "url": "http://localhost:8100",
    "documentationUrl": "https://github.com/anthropics/claude-code",
    "capabilities": {
        "streaming": False,
        "pushNotifications": False
    },
    "authentication": {
        "schemes": []
    },
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "skills": [
        # ============================================================
        # MEMORY SKILLS (Original)
        # ============================================================
        {
            "id": "store_memory",
            "name": "Store Memory",
            "description": "Store a piece of information with semantic embedding for later retrieval. Supports different memory types: session, decision, code, chunk.",
            "tags": ["memory", "storage", "embedding"],
            "examples": [
                "Store this code pattern for future reference",
                "Remember this architectural decision"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "retrieve_memory",
            "name": "Retrieve Memory",
            "description": "Retrieve specific memories by ID or filter by type and session.",
            "tags": ["memory", "retrieval"],
            "examples": [
                "Get memory with ID 42",
                "List all decisions from this session"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "semantic_search",
            "name": "Semantic Search",
            "description": "Search through stored memories using natural language. Returns semantically similar content ranked by relevance.",
            "tags": ["memory", "search", "semantic"],
            "examples": [
                "Find memories about authentication",
                "Search for code patterns related to caching"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "summarize_session",
            "name": "Summarize Session",
            "description": "Store a comprehensive session summary including key decisions and code patterns discovered during the session.",
            "tags": ["memory", "session", "summary"],
            "examples": [
                "Summarize this coding session",
                "Store session digest with key decisions"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        # ============================================================
        # TIMELINE SKILLS (New - Anti-Hallucination)
        # ============================================================
        {
            "id": "timeline_log",
            "name": "Timeline Log",
            "description": "Log an event to the session timeline. Events include: user_request, decision, action, observation, error, checkpoint. Tracks causal chains and entities.",
            "tags": ["timeline", "logging", "events", "grounding"],
            "examples": [
                "Log a decision to use React over Vue",
                "Record an action taken on a file"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "timeline_get",
            "name": "Timeline Get",
            "description": "Retrieve timeline events for a session. Can filter by event type, get anchors only, or events since a specific ID.",
            "tags": ["timeline", "retrieval", "events"],
            "examples": [
                "Get the last 20 events in this session",
                "Get all decisions made so far"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "timeline_search",
            "name": "Timeline Search",
            "description": "Semantic search across timeline events. Find events related to a specific topic or query.",
            "tags": ["timeline", "search", "semantic"],
            "examples": [
                "Search timeline for authentication changes",
                "Find events related to database work"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "timeline_auto_detect",
            "name": "Timeline Auto-Detect",
            "description": "Auto-detect and log decisions/observations from a response text. Uses pattern matching to find implicit decisions and observations.",
            "tags": ["timeline", "auto", "detection"],
            "examples": [
                "Analyze Claude's response for decisions",
                "Auto-log detected observations"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        # ============================================================
        # STATE SKILLS (New)
        # ============================================================
        {
            "id": "state_get",
            "name": "State Get",
            "description": "Get current session state including goal, entity registry, pending questions, and checkpoint info.",
            "tags": ["state", "session", "context"],
            "examples": [
                "Get current session state",
                "What's the current goal?"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "state_update",
            "name": "State Update",
            "description": "Update session state. Set current goal, add entities to registry, manage pending questions, add decisions.",
            "tags": ["state", "session", "update"],
            "examples": [
                "Set current goal to fix auth bug",
                "Register auth.py as the auth_file entity"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "state_init_session",
            "name": "Initialize Session",
            "description": "Initialize or resume a session for a project. Handles 4-hour session boundaries automatically.",
            "tags": ["state", "session", "init"],
            "examples": [
                "Start or resume session for this project",
                "Initialize session context"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        # ============================================================
        # CHECKPOINT SKILLS (New)
        # ============================================================
        {
            "id": "checkpoint_create",
            "name": "Checkpoint Create",
            "description": "Create a checkpoint snapshot of the current session state. Captures key facts, decisions, entities, and pending items.",
            "tags": ["checkpoint", "snapshot", "save"],
            "examples": [
                "Create a checkpoint before major changes",
                "Save session progress"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "checkpoint_load",
            "name": "Checkpoint Load",
            "description": "Load context from a checkpoint for session resumption. Returns grounding summary with goal, facts, decisions.",
            "tags": ["checkpoint", "load", "resume"],
            "examples": [
                "Load the last checkpoint",
                "Resume from previous session"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "checkpoint_list",
            "name": "Checkpoint List",
            "description": "List all checkpoints for a session. Shows checkpoint summaries and timestamps.",
            "tags": ["checkpoint", "list"],
            "examples": [
                "List all checkpoints in this session",
                "Show checkpoint history"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        # ============================================================
        # GROUNDING SKILLS (New - Anti-Hallucination)
        # ============================================================
        {
            "id": "context_refresh",
            "name": "Context Refresh",
            "description": "Pre-response grounding check. Returns current goal, entity registry, recent events, anchors, decisions, and potential contradictions. CALL THIS BEFORE COMPLEX RESPONSES.",
            "tags": ["grounding", "context", "anti-hallucination"],
            "examples": [
                "Ground myself before responding",
                "Check context before file changes"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "check_contradictions",
            "name": "Check Contradictions",
            "description": "Check if a statement contradicts known facts or decisions. Returns potential conflicts with established anchors and decisions.",
            "tags": ["grounding", "verification", "anti-hallucination"],
            "examples": [
                "Check if this contradicts earlier decisions",
                "Verify statement against known facts"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "verify_entity",
            "name": "Verify Entity",
            "description": "Verify an entity reference against the registry. Use when referencing files, variables, or other entities to ensure correctness.",
            "tags": ["grounding", "entity", "verification"],
            "examples": [
                "Verify the auth_file entity",
                "Check if this is the right file"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "mark_anchor",
            "name": "Mark Anchor",
            "description": "Mark a statement as a verified anchor fact. Anchors are high-confidence facts that should not be contradicted.",
            "tags": ["grounding", "anchor", "fact"],
            "examples": [
                "Establish this as a verified fact",
                "Mark this decision as final"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        # ============================================================
        # CLAUDE.MD MANAGEMENT SKILLS
        # ============================================================
        {
            "id": "claude_md_read",
            "name": "Read CLAUDE.md",
            "description": "Read the CLAUDE.md instructions file. Can read entire file or a specific section.",
            "tags": ["claude_md", "instructions", "read"],
            "examples": [
                "Read my CLAUDE.md file",
                "Show the Memory System section"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "claude_md_add_section",
            "name": "Add CLAUDE.md Section",
            "description": "Add a new section to CLAUDE.md. Use this to create new instruction categories.",
            "tags": ["claude_md", "instructions", "write"],
            "examples": [
                "Add a new section for testing guidelines",
                "Create an API conventions section"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "claude_md_update_section",
            "name": "Update CLAUDE.md Section",
            "description": "Update an existing section in CLAUDE.md. Can replace, append, or prepend content.",
            "tags": ["claude_md", "instructions", "write"],
            "examples": [
                "Update the Memory System section",
                "Append to the debugging guidelines"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "claude_md_add_instruction",
            "name": "Add CLAUDE.md Instruction",
            "description": "Add a single instruction/rule to a section. Creates the section if it doesn't exist.",
            "tags": ["claude_md", "instructions", "write"],
            "examples": [
                "Add a rule about file naming",
                "Add an instruction to the debugging section"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "claude_md_list_sections",
            "name": "List CLAUDE.md Sections",
            "description": "List all sections in the CLAUDE.md file.",
            "tags": ["claude_md", "instructions", "list"],
            "examples": [
                "What sections are in my CLAUDE.md?",
                "List all instruction categories"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "claude_md_suggest",
            "name": "Suggest CLAUDE.md Updates",
            "description": "Suggest CLAUDE.md additions based on session learnings. Analyzes anchors and decisions to recommend persistent instructions.",
            "tags": ["claude_md", "instructions", "suggest"],
            "examples": [
                "What should I add to CLAUDE.md from this session?",
                "Suggest instructions based on our work"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        # ============================================================
        # VERIFICATION SKILLS (Anti-Hallucination)
        # ============================================================
        {
            "id": "best_of_n_verify",
            "name": "Best-of-N Verification",
            "description": "Run a query N times and check consistency. Inconsistent outputs indicate potential hallucination. Use for critical facts.",
            "tags": ["verification", "anti-hallucination", "consistency"],
            "examples": [
                "Verify this fact with multiple runs",
                "Check if this answer is consistent"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "extract_quotes",
            "name": "Extract Quotes",
            "description": "Extract exact, word-for-word quotes from a document relevant to a query. Forces verbatim grounding instead of paraphrasing.",
            "tags": ["verification", "grounding", "quotes"],
            "examples": [
                "Extract relevant quotes from this document",
                "Find exact quotes about authentication"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        },
        {
            "id": "require_grounding",
            "name": "Require Grounding",
            "description": "Verify that a statement is grounded in stored facts (anchors or memories) before accepting it. Returns grounding sources or warning.",
            "tags": ["verification", "grounding", "anti-hallucination"],
            "examples": [
                "Is this statement grounded in our facts?",
                "Verify this claim against stored knowledge"
            ],
            "inputModes": ["text"],
            "outputModes": ["text"]
        }
    ]
}
