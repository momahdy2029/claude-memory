"""
Agent Registry - Discovers real agents from disk + loads MCP/hook catalogs.

Agents are discovered from ~/.claude/agents/ (global) and
<project>/.claude/agents/ (per-project) by scanning .md files with YAML
frontmatter.  Enabled agents live in the root; disabled agents live in
_disabled/<category>/ subdirectories.

MCP and hook metadata still comes from agent_catalog.json; dynamic loading
reads Claude Code settings files at runtime to check configured status.
"""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml

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

# Global agents directory
_GLOBAL_AGENTS_DIR = _HOME / ".claude" / "agents"

# ---------------------------------------------------------------------------
# Load static catalog from JSON (MCPs + hooks only now)
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

# Categories are now auto-generated from discovered agents, but keep catalog
# ones as fallback for MCPs/hooks
AGENT_CATEGORIES = _catalog.get("categories", {})
AVAILABLE_MCPS = _catalog.get("mcps", [])
AVAILABLE_HOOKS = _catalog.get("hooks", [])

# ---------------------------------------------------------------------------
# Category metadata (colors + icons for known categories)
# ---------------------------------------------------------------------------
_CATEGORY_META = {
    "design":              {"name": "Design",              "color": "#f778ba", "icon": "palette"},
    "engineering":         {"name": "Engineering",         "color": "#58a6ff", "icon": "code"},
    "marketing":           {"name": "Marketing",           "color": "#f0883e", "icon": "bullhorn"},
    "product":             {"name": "Product",             "color": "#a371f7", "icon": "lightbulb"},
    "project-management":  {"name": "Project Management",  "color": "#3fb950", "icon": "tasks"},
    "spatial-computing":   {"name": "Spatial Computing",   "color": "#79c0ff", "icon": "cube"},
    "specialized":         {"name": "Specialized",         "color": "#d2a8ff", "icon": "star"},
    "support":             {"name": "Support",             "color": "#56d364", "icon": "headset"},
    "testing":             {"name": "Testing",             "color": "#e3b341", "icon": "vial"},
}

# Color palette for agent frontmatter colors
_COLOR_MAP = {
    "blue":    "#58a6ff",
    "green":   "#3fb950",
    "red":     "#f85149",
    "purple":  "#a371f7",
    "orange":  "#f0883e",
    "yellow":  "#e3b341",
    "pink":    "#f778ba",
    "cyan":    "#79c0ff",
    "teal":    "#2ea043",
}


# ---------------------------------------------------------------------------
# Agent discovery from disk
# ---------------------------------------------------------------------------
def _parse_frontmatter(file_path: Path) -> Optional[Dict[str, Any]]:
    """Parse YAML frontmatter from an agent .md file."""
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return None

    # Match --- delimited frontmatter
    match = re.match(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
    if not match:
        return None

    try:
        data = yaml.safe_load(match.group(1))
        if not isinstance(data, dict):
            return None
        return data
    except yaml.YAMLError as e:
        logger.warning(f"Invalid YAML frontmatter in {file_path}: {e}")
        return None


def _infer_category(filename_stem: str) -> str:
    """Infer category from filename prefix like 'engineering-senior-developer'."""
    known = [
        "design", "engineering", "marketing", "product",
        "project-management", "spatial-computing", "specialized",
        "support", "testing",
    ]
    for cat in known:
        if filename_stem.startswith(cat):
            return cat
    # Try first word as category
    parts = filename_stem.split("-")
    if parts:
        return parts[0]
    return "uncategorized"


def _resolve_color(color_str: Optional[str]) -> str:
    """Resolve a color name or hex value to a hex color."""
    if not color_str:
        return "#8b949e"
    if color_str.startswith("#"):
        return color_str
    return _COLOR_MAP.get(color_str.lower(), "#8b949e")


def _parse_agent_file(
    file_path: Path,
    enabled: bool,
    scope: str,
    category: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Parse a single agent .md file into an agent dict."""
    fm = _parse_frontmatter(file_path)
    if fm is None:
        # Minimal entry for files without valid frontmatter
        fm = {}

    stem = file_path.stem
    agent_id = stem

    name = fm.get("name", stem.replace("-", " ").replace("_", " ").title())
    description = fm.get("description", "").replace("\\n", " ").strip()
    color = _resolve_color(fm.get("color"))

    if not category:
        category = _infer_category(stem)

    return {
        "id": agent_id,
        "name": name,
        "description": description,
        "color": color,
        "category": category,
        "enabled": enabled,
        "scope": scope,
        "file_path": str(file_path),
        "filename": file_path.name,
        "base_dir": str(base_dir or file_path.parent),
    }


def _scan_agent_dir(base_dir: Path, scope: str) -> List[Dict[str, Any]]:
    """Scan an agents directory for enabled and disabled agents."""
    agents = []
    if not base_dir.exists():
        return agents

    # Enabled agents: .md files directly in base_dir
    for f in sorted(base_dir.glob("*.md")):
        if f.name.lower() in ("readme.md", "readme.txt"):
            continue
        agent = _parse_agent_file(
            f, enabled=True, scope=scope, base_dir=base_dir
        )
        if agent:
            agents.append(agent)

    # Disabled agents: .md files in _disabled/ subdirectories
    disabled_dir = base_dir / "_disabled"
    if disabled_dir.exists():
        for f in sorted(disabled_dir.rglob("*.md")):
            if f.name.lower() in ("readme.md", "readme.txt"):
                continue
            # Category = parent dir name if it's a subdirectory of _disabled
            if f.parent != disabled_dir:
                category = f.parent.name
            else:
                category = _infer_category(f.stem)
            agent = _parse_agent_file(
                f, enabled=False, scope=scope,
                category=category, base_dir=base_dir,
            )
            if agent:
                agents.append(agent)

    return agents


def discover_agents(project_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Discover all agents from global and project directories.

    Returns a list of agent dicts with id, name, description, color,
    category, enabled, scope, file_path, filename, base_dir.
    """
    agents = []

    # 1. Global agents: ~/.claude/agents/
    agents += _scan_agent_dir(_GLOBAL_AGENTS_DIR, scope="global")

    # 2. Project agents: <project>/.claude/agents/
    if project_path:
        project_dir = Path(project_path) / ".claude" / "agents"
        agents += _scan_agent_dir(project_dir, scope="project")

    return agents


def discover_categories(agents: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build categories dict from discovered agents."""
    cats: Dict[str, Dict[str, Any]] = {}
    for agent in agents:
        cat_key = agent["category"]
        if cat_key not in cats:
            meta = _CATEGORY_META.get(cat_key, {
                "name": cat_key.replace("-", " ").replace("_", " ").title(),
                "color": "#8b949e",
                "icon": "circle",
            })
            cats[cat_key] = meta
    return cats


def find_agent_by_id(
    agent_id: str, project_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Find a specific agent by ID across all directories."""
    for agent in discover_agents(project_path):
        if agent["id"] == agent_id:
            return agent
    return None


def toggle_agent(agent_id: str, enabled: bool, project_path: Optional[str] = None) -> Dict[str, Any]:
    """Toggle an agent between enabled and disabled state.

    - Enable: move from _disabled/<category>/ to agents/ root
    - Disable: move from agents/ root to _disabled/<category>/

    Returns the updated agent dict.
    """
    agent = find_agent_by_id(agent_id, project_path)
    if not agent:
        raise FileNotFoundError(f"Agent '{agent_id}' not found")

    src = Path(agent["file_path"])
    base = Path(agent["base_dir"])

    if enabled and not agent["enabled"]:
        # Move from _disabled/<category>/ to agents/ root
        dest = base / agent["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        agent["file_path"] = str(dest)
        agent["enabled"] = True
    elif not enabled and agent["enabled"]:
        # Move from agents/ root to _disabled/<category>/
        category = agent["category"]
        category_dir = base / "_disabled" / category
        category_dir.mkdir(parents=True, exist_ok=True)
        dest = category_dir / agent["filename"]
        shutil.move(str(src), str(dest))
        agent["file_path"] = str(dest)
        agent["enabled"] = False

    return agent


# Backward-compatible: AVAILABLE_AGENTS as lazy discovery result
AVAILABLE_AGENTS = discover_agents()


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

def get_agents_by_category(agents: Optional[List[Dict[str, Any]]] = None):
    """Group agents by category."""
    if agents is None:
        agents = discover_agents()
    result = {}
    for agent in agents:
        cat = agent["category"]
        if cat not in result:
            result[cat] = []
        result[cat].append(agent)
    return result


def get_agent_by_id(agent_id: str):
    """Get agent by ID."""
    return find_agent_by_id(agent_id)
