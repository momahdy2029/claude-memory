"""
Agent Registry - Loads agent/MCP/hook catalogs from JSON data file.

Static display metadata lives in agent_catalog.json (~880 entries).
Dynamic loading (hooks, MCPs) reads Claude Code settings files at runtime
to determine which entries are truly configured.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths to Claude Code settings files
# ---------------------------------------------------------------------------
_HOME = Path.home()
_GLOBAL_SETTINGS_PATH = _HOME / ".claude" / "settings.json"

# Project root is two levels up from this file:
# memory-agent/services/agent_registry.py -> memory-agent -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOCAL_SETTINGS_PATH = _PROJECT_ROOT / ".claude" / "settings.local.json"

# ---------------------------------------------------------------------------
# Load static catalog from JSON
# ---------------------------------------------------------------------------
_CATALOG_PATH = Path(__file__).resolve().parent / "agent_catalog.json"


def _load_catalog() -> dict:
    """Load the agent catalog from JSON file."""
    try:
        with open(_CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load agent catalog from {_CATALOG_PATH}: {e}")
        return {"categories": {}, "agents": [], "mcps": [], "hooks": []}


_catalog = _load_catalog()

AGENT_CATEGORIES = _catalog.get("categories", {})
AVAILABLE_AGENTS = _catalog.get("agents", [])
AVAILABLE_MCPS = _catalog.get("mcps", [])
AVAILABLE_HOOKS = _catalog.get("hooks", [])


# ---------------------------------------------------------------------------
# Helper: derive a hook ID from its command string
# ---------------------------------------------------------------------------
def _hook_id_from_command(command: str) -> str:
    """Extract a stable identifier from a hook command path."""
    basename = Path(command.split()[-1]).stem
    return basename.replace("_", "-")


# ---------------------------------------------------------------------------
# Dynamic loader: read configured hooks from settings files
# ---------------------------------------------------------------------------
def _read_json_safe(path: Path) -> dict:
    """Read a JSON file, returning empty dict on any error."""
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
    return {}


def _extract_hooks_from_settings(data: dict) -> Dict[str, Dict[str, Any]]:
    """Parse the 'hooks' section of a Claude Code settings file.

    Returns a dict keyed by hook-id with trigger, matcher, and command.
    """
    result: Dict[str, Dict[str, Any]] = {}
    hooks_section = data.get("hooks", {})

    _trigger_suffix = {
        "PostToolUse": "-post",
        "PreToolUse": "-pre",
        "PreCompact": "-compact",
        "SessionEnd": "-sessionend",
    }

    for trigger, entries in hooks_section.items():
        if not isinstance(entries, list):
            continue

        for entry in entries:
            matcher = entry.get("matcher", "")
            hook_items = entry.get("hooks", [entry])
            for item in hook_items:
                if item.get("type") != "command":
                    continue
                command = item.get("command", "")
                if not command:
                    continue
                hook_id = _hook_id_from_command(command)

                if hook_id in result and result[hook_id]["trigger"] != trigger:
                    suffix = _trigger_suffix.get(trigger, f"-{trigger.lower()}")
                    hook_id = hook_id + suffix

                result[hook_id] = {
                    "trigger": trigger,
                    "matcher": matcher,
                    "command": command
                }

    return result


def _extract_mcps_from_settings(data: dict) -> Dict[str, Dict[str, Any]]:
    """Parse the 'mcpServers' section of a Claude Code settings file."""
    result: Dict[str, Dict[str, Any]] = {}
    mcp_section = data.get("mcpServers", {})

    for server_id, config in mcp_section.items():
        if isinstance(config, dict):
            result[server_id] = {
                "command": config.get("command", ""),
                "args": config.get("args", []),
                "env": config.get("env", {})
            }

    return result


def load_configured_hooks(
    project_root: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Load hooks with live configured status from settings files."""
    global_data = _read_json_safe(_GLOBAL_SETTINGS_PATH)
    local_path = (
        Path(project_root) / ".claude" / "settings.local.json"
        if project_root
        else _LOCAL_SETTINGS_PATH
    )
    local_data = _read_json_safe(local_path)

    global_hooks = _extract_hooks_from_settings(global_data)
    local_hooks = _extract_hooks_from_settings(local_data)

    enriched: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for hook in AVAILABLE_HOOKS:
        hook_copy = dict(hook)
        hook_id = hook_copy["id"]
        seen_ids.add(hook_id)

        if hook_id in local_hooks:
            hook_copy["configured"] = True
            hook_copy["settings_source"] = "project"
            hook_copy["command"] = local_hooks[hook_id].get("command", "")
        elif hook_id in global_hooks:
            hook_copy["configured"] = True
            hook_copy["settings_source"] = "global"
            hook_copy["command"] = global_hooks[hook_id].get("command", "")
        else:
            hook_copy["configured"] = False
            hook_copy["settings_source"] = None
            hook_copy["command"] = ""

        enriched.append(hook_copy)

    # Append hooks from settings not in the static catalog
    for hook_id, info in {**global_hooks, **local_hooks}.items():
        if hook_id in seen_ids:
            continue
        source = "project" if hook_id in local_hooks else "global"
        enriched.append({
            "id": hook_id,
            "name": hook_id.replace("-", " ").replace("_", " ").title(),
            "description": f"Discovered hook from {source} settings.",
            "trigger": info.get("trigger", "Unknown"),
            "matcher": info.get("matcher", ""),
            "icon": "cpu",
            "color": "#8b949e",
            "default_enabled": True,
            "source": source,
            "configured": True,
            "settings_source": source,
            "command": info.get("command", "")
        })

    return enriched


def load_configured_mcps(
    project_root: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Load MCP servers with live configured status from settings files."""
    global_data = _read_json_safe(_GLOBAL_SETTINGS_PATH)
    local_path = (
        Path(project_root) / ".claude" / "settings.local.json"
        if project_root
        else _LOCAL_SETTINGS_PATH
    )
    local_data = _read_json_safe(local_path)

    global_mcps = _extract_mcps_from_settings(global_data)
    local_mcps = _extract_mcps_from_settings(local_data)

    enriched: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for mcp in AVAILABLE_MCPS:
        mcp_copy = dict(mcp)
        mcp_id = mcp_copy["id"]
        seen_ids.add(mcp_id)

        if mcp_id in local_mcps:
            mcp_copy["configured"] = True
            mcp_copy["settings_source"] = "project"
        elif mcp_id in global_mcps:
            mcp_copy["configured"] = True
            mcp_copy["settings_source"] = "global"
        else:
            mcp_copy["configured"] = False
            mcp_copy["settings_source"] = None

        enriched.append(mcp_copy)

    for mcp_id, info in {**global_mcps, **local_mcps}.items():
        if mcp_id in seen_ids:
            continue
        source = "project" if mcp_id in local_mcps else "global"
        enriched.append({
            "id": mcp_id,
            "name": mcp_id.replace("-", " ").replace("_", " ").title(),
            "description": f"Discovered MCP server from {source} settings.",
            "icon": "box",
            "color": "#8b949e",
            "default_enabled": True,
            "source": source,
            "configured": True,
            "settings_source": source
        })

    return enriched


# ---------------------------------------------------------------------------
# Helper functions (unchanged interface)
# ---------------------------------------------------------------------------

def get_agents_by_category():
    """Group agents by category."""
    result = {}
    for agent in AVAILABLE_AGENTS:
        cat = agent["category"]
        if cat not in result:
            result[cat] = []
        result[cat].append(agent)
    return result


def get_agent_by_id(agent_id: str):
    """Get agent by ID."""
    for agent in AVAILABLE_AGENTS:
        if agent["id"] == agent_id:
            return agent
    return None
