# Claude Code Hooks for Automatic Grounding

These hooks make the anti-hallucination system **automatic** - no relying on Claude to remember to call tools.

## How It Works

1. **UserPromptSubmit** hook fires when you send a message
2. `log-user-request.py` logs your message to the timeline
3. `grounding-hook.py` fetches current context and outputs it
4. Claude Code injects the output into Claude's context automatically
5. Claude sees the grounding context BEFORE processing your message

## Installation

### 1. Make hooks executable (Unix/Mac)

```bash
chmod +x ~/.claude/hooks/grounding-hook.py
chmod +x ~/.claude/hooks/log-user-request.py
```

### 2. Add to Claude Code settings

Edit `~/.claude/settings.json` and add:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python ~/.claude/hooks/detect-correction.py"
          },
          {
            "type": "command",
            "command": "python ~/.claude/hooks/log-user-request.py"
          },
          {
            "type": "command",
            "command": "python ~/.claude/hooks/grounding-hook.py"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write|Bash|Read",
        "hooks": [
          {
            "type": "command",
            "command": "python ~/.claude/hooks/log-tool-use.py"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python ~/.claude/hooks/auto-detect-response.py"
          }
        ]
      }
    ]
  }
}
```

### 3. Copy hooks to Claude directory

```bash
# Create hooks directory
mkdir -p ~/.claude/hooks

# Copy hooks
cp hooks/grounding-hook.py ~/.claude/hooks/
cp hooks/log-user-request.py ~/.claude/hooks/
cp hooks/auto-detect-response.py ~/.claude/hooks/
```

### 4. Install Python dependencies

```bash
pip install requests
```

## What Gets Injected

Before every response, Claude sees:

```
[GROUNDING CONTEXT - VERIFY BEFORE RESPONDING]
CURRENT GOAL: Fix authentication bug in login.py
ENTITY REGISTRY (use these exact references):
  - auth_file: src/auth.py
  - config: config/settings.json
ANCHORS (verified facts - DO NOT CONTRADICT):
  - Bug is in the token validation function
  - User confirmed error happens on line 45
RECENT DECISIONS:
  - Use JWT tokens (not sessions)
RECENT EVENTS:
  - [user_request] Fix the login bug
  - [action] Read src/auth.py
  - [observation] Found null check missing
[/GROUNDING CONTEXT]
```

## Why This Works Better

| Approach | Problem |
|----------|---------|
| Claude calls tools | Claude forgets to call them when hallucinating |
| Automatic injection | Context is there whether Claude remembers or not |

The hallucination happens during generation. By injecting context BEFORE generation, we give Claude the grounding information when it matters.

## Troubleshooting

### Memory agent not running
The hooks will silently fail if the memory agent isn't running. Start it:
```bash
cd memory-agent
python main.py
```

### No session file
The grounding hook creates a `.claude_session` file in your project directory. If you want to start fresh, delete it:
```bash
rm .claude_session
```

### Check if hooks are running
Add some debug output to see if hooks are being called:
```bash
# In grounding-hook.py, add at the start:
print("[DEBUG] Grounding hook called", file=sys.stderr)
```

## Advanced: Auto-Detect from Responses

The `Stop` hook runs after Claude responds. We can use it to auto-detect decisions and observations:

```python
# auto-detect-response.py
# Parses Claude's response and logs any detected decisions/observations
```

This closes the loop - user requests are logged, context is injected, and Claude's responses are analyzed for implicit decisions.
