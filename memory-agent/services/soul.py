"""Soul Service — Synthesis engine for persistent personality and learning.

Provides 4 core functions:
  1. generate_soul_brief  — Session start: returns synthesized context string
  2. capture_soul_fragment — Stop hook: lightweight regex extraction of high-signal content
  3. run_soul_integration  — Session end: merge fragments into persistent soul_state
  4. enrich_with_soul      — memory_ask: add soul context to search results

All functions are designed with strict time budgets:
  - generate_soul_brief:  < 1s   (pure DB read)
  - capture_soul_fragment: < 200ms (regex only, no LLM)
  - run_soul_integration:  < 5s   (heuristics first, LLM only if 20+ fragments)
  - enrich_with_soul:      < 200ms (DB read only)
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fragment extraction patterns (regex, no LLM)
# ---------------------------------------------------------------------------

SOUL_PATTERNS: Dict[str, List[re.Pattern]] = {
    "decision_made": [
        re.compile(
            r"(?:let's|we'll|going to|chose to|decided to)\s+(?:use|go with|implement|try)\s+(.+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:using|choosing|picked)\s+(\S+)\s+(?:because|since|for)",
            re.IGNORECASE,
        ),
    ],
    "preference_expressed": [
        re.compile(
            r"(?:I prefer|you should always|always use|never use|don't use|I like to)\s+(.+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:remember to|make sure to|don't forget to)\s+(.+)",
            re.IGNORECASE,
        ),
    ],
    "error_resolved": [
        re.compile(
            r"(?:fixed|resolved|solved|the issue was|root cause)\s*:?\s*(.+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:the (?:fix|solution) (?:was|is))\s+(.+)",
            re.IGNORECASE,
        ),
    ],
    "pattern_used": [
        re.compile(
            r"(?:same (?:approach|pattern|method) as)\s+(.+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:like we did (?:for|in|with))\s+(.+)",
            re.IGNORECASE,
        ),
    ],
    "correction_received": [
        re.compile(
            r"(?:no,?\s+(?:actually|that's wrong|not like that)|(?:don't|stop)\s+(?:do|doing)\s+that)\s*[,:]?\s*(.+)",
            re.IGNORECASE,
        ),
    ],
}

# Max content length per fragment
MAX_FRAGMENT_LENGTH = 300


class SoulService:
    """Central synthesis engine for persistent personality and learning."""

    def __init__(self, db):
        """
        Args:
            db: DatabaseService instance with soul table methods
        """
        self.db = db

    # ------------------------------------------------------------------
    # 1. generate_soul_brief — Called at session start
    # ------------------------------------------------------------------

    async def generate_soul_brief(self, project_path: str) -> str:
        """Generate a soul brief for session context injection.

        Reads soul_state for this project and returns a 200-400 word brief.
        Pure DB read, no LLM — budget: < 1s.

        Returns:
            Formatted brief string, or empty string if no soul state exists.
        """
        state = await self.db.get_soul_state(project_path)
        if not state or not state.get("soul_brief"):
            # No soul state yet — return minimal placeholder
            return ""

        # Parse JSON fields safely
        user_model = _safe_json_loads(state.get("user_model", "{}"), {})
        project_understanding = _safe_json_loads(state.get("project_understanding", "{}"), {})
        success_journal = _safe_json_loads(state.get("success_journal", "[]"), [])
        blind_spots = _safe_json_loads(state.get("blind_spots", "[]"), [])
        tool_preferences = _safe_json_loads(state.get("tool_preferences", "{}"), {})
        integration_count = state.get("integration_count", 0)

        # Build brief from template
        parts = []
        parts.append("[SOUL CONTEXT — Claude Memory]")

        # Project header
        project_name = project_understanding.get("name", project_path.split("/")[-1].split("\\")[-1])
        parts.append(f"Project: {project_name}")
        parts.append(f"Sessions integrated: {integration_count}")
        if state.get("last_integrated_at"):
            parts.append(f"Last integration: {state['last_integrated_at']}")

        # User model section
        if user_model:
            parts.append("\nYOU KNOW THIS USER:")
            preferences = user_model.get("preferences", [])
            for pref in preferences[-5:]:  # Last 5 preferences
                parts.append(f"- {pref}")
            dislikes = user_model.get("dislikes", [])
            for dislike in dislikes[-3:]:
                parts.append(f"- Dislikes: {dislike}")
            work_style = user_model.get("work_style", [])
            for ws in work_style[-3:]:
                parts.append(f"- {ws}")

        # Project understanding
        if project_understanding:
            parts.append("\nPROJECT UNDERSTANDING:")
            tech = project_understanding.get("tech_choices", [])
            for t in tech[-4:]:
                parts.append(f"- {t}")
            arch_decisions = project_understanding.get("architecture_decisions", [])
            for ad in arch_decisions[-3:]:
                parts.append(f"- {ad}")
            focus = project_understanding.get("recent_focus", "")
            if focus:
                parts.append(f"- Recent focus: {focus}")

        # Success journal (recent wins)
        if success_journal:
            parts.append("\nRECENT LEARNINGS:")
            for entry in success_journal[-4:]:
                parts.append(f"- {entry}")

        # Blind spots
        if blind_spots:
            parts.append("\nACTIVE BLIND SPOTS:")
            for bs in blind_spots[-3:]:
                parts.append(f"- {bs}")

        # Tool preferences
        if tool_preferences:
            favored = tool_preferences.get("favored", [])
            if favored:
                parts.append(f"\nTOOL PREFERENCES: {', '.join(favored[-5:])}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 2. capture_soul_fragment — Called on every Stop hook
    # ------------------------------------------------------------------

    async def capture_soul_fragment(
        self, session_id: str, fragment_type: str,
        content: str, project_path: str = ""
    ) -> Optional[int]:
        """Capture a soul fragment from a response.

        Lightweight — just stores in soul_fragments staging table.
        Budget: < 200ms.

        Args:
            session_id: Current session ID
            fragment_type: One of: decision_made, error_resolved, preference_expressed,
                          pattern_used, correction_received
            content: The extracted fragment content
            project_path: Project path

        Returns:
            Fragment ID if stored, None on error.
        """
        # Truncate content
        if len(content) > MAX_FRAGMENT_LENGTH:
            content = content[:MAX_FRAGMENT_LENGTH] + "..."

        return await self.db.insert_soul_fragment(
            session_id=session_id,
            project_path=project_path,
            fragment_type=fragment_type,
            content=content.strip(),
        )

    def extract_fragments(self, text: str) -> List[Dict[str, str]]:
        """Extract soul fragments from text using regex patterns.

        Pure regex, no LLM. Budget: < 100ms.

        Args:
            text: Text to scan for high-signal content.

        Returns:
            List of {"fragment_type": ..., "content": ...} dicts.
        """
        fragments = []
        seen_content = set()

        for fragment_type, patterns in SOUL_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    captured = match.group(1).strip() if match.lastindex else match.group(0).strip()
                    # Skip very short or very long matches
                    if len(captured) < 10 or len(captured) > MAX_FRAGMENT_LENGTH:
                        continue
                    # Deduplicate within this extraction
                    key = captured[:50].lower()
                    if key in seen_content:
                        continue
                    seen_content.add(key)

                    fragments.append({
                        "fragment_type": fragment_type,
                        "content": captured[:MAX_FRAGMENT_LENGTH],
                    })

        return fragments

    # ------------------------------------------------------------------
    # 3. run_soul_integration — Called at session end
    # ------------------------------------------------------------------

    async def run_soul_integration(
        self, session_id: str, project_path: str
    ) -> Dict[str, Any]:
        """Integrate soul fragments from a session into persistent soul_state.

        Groups fragments by type, counts patterns, merges into soul_state
        with recency weighting. Uses heuristics (no LLM).
        Budget: < 5s.

        Returns:
            Dict with integration stats.
        """
        # Get all unintegrated fragments for this project
        fragments = await self.db.get_unintegrated_fragments(project_path)
        if not fragments:
            return {"integrated": 0, "message": "No fragments to integrate"}

        # Load existing soul state (or create new)
        state = await self.db.get_soul_state(project_path) or {}
        user_model = _safe_json_loads(state.get("user_model", "{}"), {})
        project_understanding = _safe_json_loads(state.get("project_understanding", "{}"), {})
        success_journal = _safe_json_loads(state.get("success_journal", "[]"), [])
        blind_spots = _safe_json_loads(state.get("blind_spots", "[]"), [])
        tool_preferences = _safe_json_loads(state.get("tool_preferences", "{}"), {})
        integration_count = state.get("integration_count", 0) or 0

        # Group fragments by type
        by_type: Dict[str, List[str]] = {}
        for frag in fragments:
            ft = frag["fragment_type"]
            by_type.setdefault(ft, []).append(frag["content"])

        # --- Merge decisions into project_understanding ---
        decisions = by_type.get("decision_made", [])
        if decisions:
            arch_decisions = project_understanding.get("architecture_decisions", [])
            for d in decisions:
                entry = f"[Session {integration_count + 1}] {d}"
                arch_decisions.append(entry)
            # Keep last 20 decisions
            project_understanding["architecture_decisions"] = arch_decisions[-20:]

        # --- Merge preferences into user_model ---
        preferences = by_type.get("preference_expressed", [])
        if preferences:
            user_prefs = user_model.get("preferences", [])
            for p in preferences:
                # Avoid near-duplicates
                if not any(_is_similar(p, existing) for existing in user_prefs):
                    user_prefs.append(p)
            user_model["preferences"] = user_prefs[-15:]  # Keep last 15

        # --- Merge error resolutions into success_journal ---
        errors = by_type.get("error_resolved", [])
        if errors:
            for e in errors:
                entry = f"Fixed: {e}"
                success_journal.append(entry)
            success_journal = success_journal[-15:]  # Keep last 15

        # --- Merge patterns into success_journal ---
        patterns = by_type.get("pattern_used", [])
        if patterns:
            for p in patterns:
                entry = f"Pattern: {p}"
                success_journal.append(entry)
            success_journal = success_journal[-15:]

        # --- Merge corrections into blind_spots ---
        corrections = by_type.get("correction_received", [])
        if corrections:
            for c in corrections:
                entry = f"[Session {integration_count + 1}] {c}"
                blind_spots.append(entry)
            blind_spots = blind_spots[-10:]  # Keep last 10

        # Build soul brief text
        brief = await self._build_brief_text(
            project_path, user_model, project_understanding,
            success_journal, blind_spots, tool_preferences,
            integration_count + 1
        )

        # Update soul state
        now = datetime.now().isoformat()
        updates = {
            "soul_brief": brief,
            "user_model": json.dumps(user_model),
            "project_understanding": json.dumps(project_understanding),
            "success_journal": json.dumps(success_journal),
            "blind_spots": json.dumps(blind_spots),
            "tool_preferences": json.dumps(tool_preferences),
            "last_integrated_at": now,
            "integration_count": integration_count + 1,
        }

        success = await self.db.upsert_soul_state(project_path, updates)

        # Mark fragments as integrated
        integrated_count = 0
        if success:
            # Mark all fragments from this session AND any older unintegrated ones
            for frag in fragments:
                sid = frag.get("session_id", session_id)
                await self.db.mark_fragments_integrated(sid)
            integrated_count = len(fragments)

        return {
            "integrated": integrated_count,
            "by_type": {k: len(v) for k, v in by_type.items()},
            "integration_number": integration_count + 1,
            "success": success,
        }

    async def _build_brief_text(
        self, project_path: str, user_model: dict,
        project_understanding: dict, success_journal: list,
        blind_spots: list, tool_preferences: dict,
        integration_count: int
    ) -> str:
        """Build the soul brief text from state components."""
        parts = []
        project_name = project_understanding.get(
            "name",
            project_path.replace("\\", "/").rstrip("/").split("/")[-1]
        )

        parts.append(f"Project: {project_name} | Sessions: {integration_count}")

        if user_model.get("preferences"):
            prefs = user_model["preferences"][-3:]
            parts.append("User: " + "; ".join(prefs))

        if project_understanding.get("architecture_decisions"):
            recent = project_understanding["architecture_decisions"][-2:]
            parts.append("Decisions: " + "; ".join(recent))

        if success_journal:
            recent = success_journal[-2:]
            parts.append("Learnings: " + "; ".join(recent))

        if blind_spots:
            recent = blind_spots[-2:]
            parts.append("Watch out: " + "; ".join(recent))

        return " | ".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # 4. enrich_with_soul — Called in memory_ask
    # ------------------------------------------------------------------

    async def enrich_with_soul(
        self, search_results: Dict[str, Any], project_path: str
    ) -> Dict[str, Any]:
        """Add soul context to search results.

        Pure DB read, budget: < 200ms.

        Args:
            search_results: Existing search results dict from memory_ask
            project_path: Project to fetch soul for

        Returns:
            search_results dict with added 'soul_context' key.
        """
        if not project_path:
            return search_results

        state = await self.db.get_soul_state(project_path)
        if not state:
            return search_results

        soul_context = {}

        # Include the brief
        if state.get("soul_brief"):
            soul_context["brief"] = state["soul_brief"]

        # Parse and include relevant fields based on query
        user_model = _safe_json_loads(state.get("user_model", "{}"), {})
        if user_model.get("preferences"):
            soul_context["user_preferences"] = user_model["preferences"][-5:]

        blind_spots = _safe_json_loads(state.get("blind_spots", "[]"), [])
        if blind_spots:
            soul_context["blind_spots"] = blind_spots[-3:]

        success_journal = _safe_json_loads(state.get("success_journal", "[]"), [])
        if success_journal:
            soul_context["recent_learnings"] = success_journal[-3:]

        soul_context["integration_count"] = state.get("integration_count", 0)

        search_results["soul_context"] = soul_context
        return search_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_loads(value: Any, default: Any) -> Any:
    """Safely parse JSON string, returning default on failure."""
    if isinstance(value, (dict, list)):
        return value
    if not value or not isinstance(value, str):
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _is_similar(a: str, b: str, threshold: float = 0.7) -> bool:
    """Quick similarity check between two strings using word overlap."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b)
    total = max(len(words_a), len(words_b))
    return (overlap / total) >= threshold
