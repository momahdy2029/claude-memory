#!/usr/bin/env python3
"""
Problem Detector Hook for Claude Code

This script runs on UserPromptSubmit event and detects when the user
is describing a problem that might benefit from memory search.

When a problem is detected, it:
1. Updates session state with problem_solving_mode = true
2. Extracts keywords from the problem description
3. Outputs a reminder to stdout for injection into Claude's context

Configure in Claude Code settings:
{
  "hooks": {
    "UserPromptSubmit": ["python /path/to/problem-detector.py"]
  }
}
"""

import os
import sys
import json
import re
import logging
import requests
from pathlib import Path
from typing import Any, Optional, List

# Configure logging to stderr (important for Claude Code hooks)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Configuration from environment
MEMORY_AGENT_URL = os.getenv("MEMORY_AGENT_URL", "http://localhost:8102")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

# Problem detection patterns (comprehensive)
PROBLEM_PATTERNS = [
    # Direct error mentions
    r"(?i)(?:error|exception|bug|issue|problem|fail|crash|broke|doesn't work|not working|won't work)",
    # Help requests
    r"(?i)(?:how (?:do|can|to)|help me|fix|solve|debug|troubleshoot|figure out)",
    # Error messages (common programming errors)
    r"(?i)(?:TypeError|SyntaxError|ReferenceError|NameError|AttributeError|ValueError|KeyError|IndexError|Warning|Fatal|Exception|Traceback|undefined|null pointer|segfault|ENOENT|ECONNREFUSED|404|500|403)",
    # Frustration signals
    r"(?i)(?:again|still|keeps|won't|can't|unable|stuck|tried everything)",
    # Code references with issues
    r"(?i)(?:this code|my (?:code|script|function|app)|the (?:bug|error|problem))",
]

# Patterns that indicate simple questions (not problems)
SIMPLE_QUESTION_PATTERNS = [
    r"(?i)^what (?:is|are|does)",
    r"(?i)^explain\s",
    r"(?i)^show me\s",
    r"(?i)^list\s",
    r"(?i)^describe\s",
    r"(?i)^tell me about\s",
]

# Keywords to extract from problem descriptions
KEYWORD_PATTERNS = [
    # Error types
    r"\b(TypeError|SyntaxError|ReferenceError|NameError|AttributeError|ValueError|KeyError|IndexError|Exception|Error)\b",
    # Technology keywords
    r"\b(python|javascript|typescript|react|node|npm|pip|docker|git|webpack|vite|laravel|php|mysql|postgres|redis|api|http|https|ssl|cors|auth|token|jwt)\b",
    # Action keywords
    r"\b(import|export|install|build|compile|run|start|stop|deploy|test|debug|connect|load|save|read|write|create|delete|update)\b",
    # Common error codes
    r"\b(404|500|403|401|ENOENT|ECONNREFUSED|ETIMEDOUT|EPERM|EACCES)\b",
]


def safe_get(data: Any, *keys, default: Any = None) -> Any:
    """
    Safely navigate nested data structures (dicts and lists).

    Args:
        data: The data structure to navigate
        *keys: Keys (str for dict) or indices (int for list) to traverse
        default: Value to return if path doesn't exist

    Returns:
        The value at the path, or default if not found
    """
    for key in keys:
        if data is None:
            return default
        if isinstance(data, dict):
            data = data.get(key, default)
        elif isinstance(data, list) and isinstance(key, int):
            if 0 <= key < len(data):
                data = data[key]
            else:
                return default
        else:
            return default
    return data


def get_project_path():
    """Get current working directory as project path."""
    return os.getcwd()


def load_session_data() -> Optional[dict]:
    """Load session data from JSON file."""
    session_file = Path(get_project_path()) / ".claude_session"
    if session_file.exists():
        try:
            content = session_file.read_text().strip()
            # Try JSON format first
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error, trying legacy format: {e}")
            # Fall back to legacy plain text format (just session_id)
            try:
                content = session_file.read_text().strip()
                return {"session_id": content}
            except (IOError, OSError) as read_err:
                logger.warning(f"Failed to read session file: {read_err}")
                return None
        except (IOError, OSError) as e:
            logger.warning(f"Failed to read session file: {e}")
            return None
    return None


def save_session_data(data: dict):
    """Save session data to JSON file."""
    session_file = Path(get_project_path()) / ".claude_session"
    try:
        session_file.write_text(json.dumps(data, indent=2))
    except (IOError, OSError) as e:
        logger.warning(f"Failed to save session data: {e}")


def call_memory_agent(skill_id: str, params: dict) -> Optional[dict]:
    """Call the memory agent API."""
    try:
        response = requests.post(
            f"{MEMORY_AGENT_URL}/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "problem-detector-hook",
                "method": "tasks/send",
                "params": {
                    "message": {"parts": [{"type": "text", "text": ""}]},
                    "metadata": {
                        "skill_id": skill_id,
                        "params": params
                    }
                }
            },
            timeout=API_TIMEOUT
        )
        result = response.json()

        # Safely extract the artifact text using safe_get
        artifact_text = safe_get(result, "result", "artifacts", 0, "parts", 0, "text")
        if artifact_text:
            try:
                return json.loads(artifact_text)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse artifact text as JSON for skill '{skill_id}': {e}")
                return None
        return None

    except requests.RequestException as e:
        # Silently fail - don't break Claude Code if memory agent is down
        logger.debug(f"Memory agent request failed for skill '{skill_id}': {e}")
        return None
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to decode memory agent response for skill '{skill_id}': {e}")
        return None


def is_simple_question(text: str) -> bool:
    """
    Check if the text is a simple question that doesn't need problem-solving mode.

    Returns:
        True if this is a simple question, False otherwise
    """
    for pattern in SIMPLE_QUESTION_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def detect_problem(text: str) -> bool:
    """
    Detect if the user's message describes a problem.

    Args:
        text: The user's message

    Returns:
        True if a problem is detected, False otherwise
    """
    # Skip very short messages
    if len(text) < 15:
        return False

    # Skip simple questions
    if is_simple_question(text):
        return False

    # Check for problem patterns
    match_count = 0
    for pattern in PROBLEM_PATTERNS:
        if re.search(pattern, text):
            match_count += 1

    # Require at least 1 match for problem detection
    # Be conservative to avoid false positives
    return match_count >= 1


def extract_keywords(text: str) -> List[str]:
    """
    Extract relevant keywords from the problem description.

    Args:
        text: The user's message

    Returns:
        List of extracted keywords
    """
    keywords = set()

    for pattern in KEYWORD_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            keywords.add(match.lower())

    # Also extract words that appear near "error", "problem", "issue", etc.
    context_patterns = [
        r"(?:error|problem|issue|bug)\s+(?:with|in|when|while)\s+(\w+)",
        r"(\w+)\s+(?:error|problem|issue|bug)",
        r"(?:can't|cannot|won't|doesn't)\s+(\w+)",
    ]

    for pattern in context_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) > 2:  # Skip very short words
                keywords.add(match.lower())

    return list(keywords)[:10]  # Limit to 10 keywords


def format_reminder_output() -> str:
    """
    Format the reminder message for stdout injection.

    Returns:
        Formatted reminder string
    """
    return """[MEMORY HEARTBEAT - PROBLEM DETECTED]
Problem-solving mode activated.
BEFORE searching externally, check your memory:
  memory_search_patterns("your problem description")
  memory_search(query="...", type="error")
Past solutions are faster and verified to work.
[/MEMORY HEARTBEAT]"""


def main():
    """Main entry point for the hook."""
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse hook input JSON: {e}")
        sys.exit(0)
    except (IOError, OSError) as e:
        logger.debug(f"Failed to read stdin: {e}")
        sys.exit(0)

    # Get user message from hook input
    user_message = hook_input.get("user_prompt", "")
    if not user_message:
        # Try alternative format
        session_messages = hook_input.get("session_messages", [])
        if session_messages:
            last_msg = session_messages[-1]
            if last_msg.get("role") == "user":
                user_message = last_msg.get("content", "")

    if not user_message:
        sys.exit(0)

    # Detect if this is a problem description
    if not detect_problem(user_message):
        # Not a problem - exit silently
        sys.exit(0)

    # Extract keywords from the problem
    keywords = extract_keywords(user_message)

    logger.info(f"Problem detected with keywords: {keywords}")

    # Load session data
    session_data = load_session_data()
    if not session_data:
        session_data = {}

    session_id = session_data.get("session_id")

    # Update session state with heartbeat information
    if "heartbeat" not in session_data:
        session_data["heartbeat"] = {}

    session_data["heartbeat"]["problem_solving_mode"] = True
    session_data["heartbeat"]["problem_keywords"] = keywords

    # Save updated session data
    save_session_data(session_data)

    # Also update memory agent state if session exists
    if session_id:
        call_memory_agent("state_update", {
            "session_id": session_id,
            "heartbeat": {
                "problem_solving_mode": True,
                "problem_keywords": keywords
            }
        })

    # Output reminder to stdout for injection into Claude's context
    print(format_reminder_output())

    sys.exit(0)


if __name__ == "__main__":
    main()
