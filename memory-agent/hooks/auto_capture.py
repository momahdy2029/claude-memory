#!/usr/bin/env python3
"""Auto-capture hook for Claude Code.

This hook automatically captures:
- Tool executions and their outcomes
- Errors encountered during sessions
- Decisions and their rationale
- File modifications

Configure in Claude Code settings:
{
  "hooks": {
    "PostToolUse": ["python /path/to/auto_capture.py"],
    "Notification": ["python /path/to/auto_capture.py --notification"]
  }
}
"""
import os
import sys
import json
import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

# Configure logging to stderr (important for Claude Code hooks)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

MEMORY_AGENT_URL = os.getenv("MEMORY_AGENT_URL", "http://localhost:8102")
API_KEY = os.getenv("MEMORY_API_KEY", "")

# Tool categories for smart capture
ERROR_INDICATORS = ["error", "failed", "exception", "traceback", "cannot", "unable"]
DECISION_INDICATORS = ["chose", "decided", "selected", "using", "approach", "strategy"]
IMPORTANT_TOOLS = ["Write", "Edit", "Bash", "Task"]


def should_capture(tool_name: str, output: str, exit_code: Optional[int] = None) -> tuple[bool, str, int]:
    """Determine if this tool execution should be captured.

    Returns: (should_capture, memory_type, importance)
    """
    output_lower = output.lower() if output else ""

    # Always capture errors
    if exit_code and exit_code != 0:
        return True, "error", 8

    if any(indicator in output_lower for indicator in ERROR_INDICATORS):
        return True, "error", 7

    # Capture important tool executions
    if tool_name in IMPORTANT_TOOLS:
        # File writes are decisions
        if tool_name in ["Write", "Edit"]:
            return True, "decision", 6
        # Bash commands with meaningful output
        if tool_name == "Bash" and len(output) > 50:
            return True, "chunk", 5
        # Task tool usage
        if tool_name == "Task":
            return True, "decision", 6

    # Skip routine operations
    if tool_name in ["Read", "Glob", "Grep"] and len(output) < 500:
        return False, "", 0

    return False, "", 0


async def send_to_memory(
    content: str,
    memory_type: str,
    importance: int,
    metadata: Dict[str, Any],
    project_path: Optional[str] = None
):
    """Send captured data to memory agent."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-Memory-Key"] = API_KEY

    payload = {
        "jsonrpc": "2.0",
        "method": "skills/call",
        "params": {
            "skill_id": "store_memory",
            "params": {
                "content": content,
                "type": memory_type,
                "importance": importance,
                "project_path": project_path,
                "tags": metadata.get("tags", []),
                "metadata": metadata,
                "outcome": metadata.get("outcome"),
                "success": metadata.get("success", True)
            }
        },
        "id": f"auto-capture-{datetime.now().isoformat()}"
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{MEMORY_AGENT_URL}/a2a",
                json=payload,
                headers=headers
            )
            return response.status_code == 200
    except httpx.RequestError as e:
        # Silently fail - don't interrupt Claude's work
        logger.debug(f"Memory agent request failed: {e}")
        return False
    except httpx.HTTPStatusError as e:
        logger.debug(f"Memory agent returned error status: {e}")
        return False


async def post_session_activity(session_id: str, project_path: str, event_type: str, summary: str, files: List[str] = None):
    """Post a cross-session activity event and track modified files."""
    if not session_id or not project_path:
        return

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-Memory-Key"] = API_KEY

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            # Post activity event
            await client.post(
                f"{MEMORY_AGENT_URL}/api/sessions/activity",
                json={
                    "session_id": session_id,
                    "project_path": project_path,
                    "event_type": event_type,
                    "summary": summary,
                    "files": files or [],
                },
                headers=headers,
            )

            # Append files to session's modified files list
            if files:
                for f in files:
                    await client.post(
                        f"{MEMORY_AGENT_URL}/a2a",
                        json={
                            "jsonrpc": "2.0",
                            "method": "skills/call",
                            "params": {
                                "skill_id": "session_append_file",
                                "params": {
                                    "session_id": session_id,
                                    "file_path": f,
                                }
                            },
                            "id": f"auto-capture-file-{datetime.now().isoformat()}"
                        },
                        headers=headers,
                    )
    except Exception as e:
        logger.debug(f"Cross-session activity post failed: {e}")


async def capture_tool_use(hook_data: Dict[str, Any]):
    """Capture a tool execution event."""
    tool_name = hook_data.get("tool_name", "Unknown")
    tool_input = hook_data.get("tool_input", {})
    tool_output = hook_data.get("tool_output", "")
    exit_code = hook_data.get("exit_code")
    session_id = hook_data.get("session_id")
    project_path = hook_data.get("project_path")

    should_cap, mem_type, importance = should_capture(tool_name, tool_output, exit_code)

    if not should_cap:
        return

    # Build content summary
    if mem_type == "error":
        content = f"Error in {tool_name}: {tool_output[:500]}"
        tags = ["error", tool_name.lower(), "auto-captured"]
        success = False
    else:
        # Summarize the action
        if tool_name == "Write":
            file_path = tool_input.get("file_path", "unknown")
            content = f"Created/updated file: {file_path}"
            tags = ["file-change", "auto-captured"]
        elif tool_name == "Edit":
            file_path = tool_input.get("file_path", "unknown")
            content = f"Edited file: {file_path} - {tool_input.get('old_string', '')[:100]} -> {tool_input.get('new_string', '')[:100]}"
            tags = ["file-edit", "auto-captured"]
        elif tool_name == "Bash":
            cmd = tool_input.get("command", "")[:200]
            content = f"Executed: {cmd}\nResult: {tool_output[:300]}"
            tags = ["command", "auto-captured"]
        elif tool_name == "Task":
            content = f"Delegated task: {tool_input.get('prompt', '')[:300]}"
            tags = ["delegation", "auto-captured"]
        else:
            content = f"{tool_name}: {str(tool_input)[:200]}\nOutput: {tool_output[:300]}"
            tags = [tool_name.lower(), "auto-captured"]
        success = True

    metadata = {
        "tool_name": tool_name,
        "session_id": session_id,
        "exit_code": exit_code,
        "auto_captured": True,
        "timestamp": datetime.now().isoformat(),
        "tags": tags,
        "outcome": "success" if success else "error",
        "success": success
    }

    await send_to_memory(content, mem_type, importance, metadata, project_path)

    # ============================================================
    # CROSS-SESSION AWARENESS: Post file changes to activity feed
    # ============================================================
    if tool_name in ("Write", "Edit") and session_id and project_path:
        file_path = tool_input.get("file_path", "")
        if file_path:
            event_type = "file_change"
            summary = f"{'Created' if tool_name == 'Write' else 'Edited'} {file_path}"
            await post_session_activity(session_id, project_path, event_type, summary, [file_path])


async def capture_notification(hook_data: Dict[str, Any]):
    """Capture a notification/error event."""
    notification_type = hook_data.get("type", "info")
    message = hook_data.get("message", "")
    project_path = hook_data.get("project_path")

    if notification_type == "error":
        content = f"Session error: {message}"
        await send_to_memory(
            content,
            "error",
            8,
            {
                "notification_type": notification_type,
                "auto_captured": True,
                "timestamp": datetime.now().isoformat(),
                "tags": ["error", "notification", "auto-captured"],
                "success": False
            },
            project_path
        )


def read_stdin_hook_data() -> Dict[str, Any]:
    """Read hook data from stdin (Claude Code passes data this way)."""
    try:
        if not sys.stdin.isatty():
            data = sys.stdin.read()
            if data:
                return json.loads(data)
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse hook input JSON from stdin: {e}")
    except (IOError, OSError) as e:
        logger.debug(f"Failed to read from stdin: {e}")
    return {}


def main():
    parser = argparse.ArgumentParser(description="Auto-capture hook for Claude Code")
    parser.add_argument("--notification", action="store_true", help="Handle notification event")
    parser.add_argument("--test", action="store_true", help="Test mode with sample data")
    args = parser.parse_args()

    if args.test:
        # Test with sample data
        test_data = {
            "tool_name": "Bash",
            "tool_input": {"command": "python test.py"},
            "tool_output": "Error: ModuleNotFoundError: No module named 'foo'",
            "exit_code": 1,
            "project_path": "/test/project"
        }
        asyncio.run(capture_tool_use(test_data))
        print("Test capture sent", file=sys.stderr)
        return

    hook_data = read_stdin_hook_data()

    if not hook_data:
        # No data from stdin, might be direct invocation
        return

    if args.notification:
        asyncio.run(capture_notification(hook_data))
    else:
        asyncio.run(capture_tool_use(hook_data))


if __name__ == "__main__":
    main()
