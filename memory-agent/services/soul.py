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
import hashlib
import logging
import time
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

# ---------------------------------------------------------------------------
# Session tag regex (used by fingerprint, freshness, and chain helpers)
# ---------------------------------------------------------------------------

_SESSION_TAG_RE = re.compile(r"^\[Session (\d+)\]\s*")

# ---------------------------------------------------------------------------
# Content fingerprinting for deduplication
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "that", "this", "was", "are",
    "be", "has", "had", "have", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "so",
    "if", "then", "than", "too", "very", "just", "about", "up", "out",
})


def _content_fingerprint(text: str) -> str:
    """Generate a stable content fingerprint for deduplication.

    Lowercase, strip session tag, strip punctuation, remove stop words,
    sort remaining words, MD5 hash truncated to 12 chars.
    """
    cleaned = _SESSION_TAG_RE.sub("", text)
    cleaned = re.sub(r"[^\w\s]", "", cleaned.lower())
    words = sorted(w for w in cleaned.split() if w not in _STOP_WORDS)
    return hashlib.md5(" ".join(words).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Freshness decay helpers
# ---------------------------------------------------------------------------


def _parse_session_number(entry: str) -> Optional[int]:
    """Extract session number from a [Session N] tagged entry."""
    m = _SESSION_TAG_RE.match(entry)
    return int(m.group(1)) if m else None


def _freshness_score(entry: str, current_session: int) -> float:
    """Calculate freshness score. Recent entries score higher.

    Untagged entries get floor score 0.3 — they naturally sink to the
    bottom and are eventually trimmed by list limits.
    """
    session_num = _parse_session_number(entry)
    if session_num is None:
        return 0.3
    age = current_session - session_num
    return max(0.3, 1.0 - (age * 0.05))


def _sort_by_freshness(entries: list, current_session: int) -> list:
    """Sort entries by freshness score, most recent first."""
    return sorted(
        entries,
        key=lambda e: _freshness_score(e, current_session),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Decision chain helpers
# ---------------------------------------------------------------------------


def _strip_session_tag(entry: str) -> str:
    """Remove [Session N] prefix from an entry."""
    return _SESSION_TAG_RE.sub("", entry)


def _build_decision_chains(decisions: list, current_session: int) -> list:
    """Group decisions into evolution chains by content similarity.

    Algorithm:
      1. Fingerprint each decision (strip session tag first)
      2. Group by exact fingerprint match
      3. Fuzzy-merge chains with _is_similar(threshold=0.5) on representatives
      4. Sort chains by freshness score

    Returns list of chain dicts with keys:
      topic, entries, sessions, span, freshness
    """
    if not decisions:
        return []

    # Step 1: Build items with metadata
    items = []
    for d in decisions:
        stripped = _strip_session_tag(d)
        fp = _content_fingerprint(stripped)
        session_num = _parse_session_number(d) or 0
        items.append({
            "entry": d,
            "stripped": stripped,
            "fingerprint": fp,
            "session": session_num,
        })

    # Step 2: Group by exact fingerprint
    groups: Dict[str, list] = {}
    for item in items:
        groups.setdefault(item["fingerprint"], []).append(item)

    # Step 3: Build initial chains from fingerprint groups
    chains = []
    for fp, group_items in groups.items():
        chains.append({
            "topic": group_items[0]["stripped"],
            "entries": [g["entry"] for g in group_items],
            "sessions": sorted(set(g["session"] for g in group_items)),
            "representative": group_items[0]["stripped"],
        })

    # Step 4: Fuzzy-merge chains whose representatives are similar
    merged = []
    used: set = set()
    for i, chain in enumerate(chains):
        if i in used:
            continue
        current = {
            "topic": chain["topic"],
            "entries": list(chain["entries"]),
            "sessions": list(chain["sessions"]),
            "representative": chain["representative"],
        }
        for j in range(i + 1, len(chains)):
            if j in used:
                continue
            # Compare with stop words stripped to avoid false
            # positives on structural words like "use", "for"
            rep_a = " ".join(
                w for w in chain["representative"].lower().split()
                if w not in _STOP_WORDS
            )
            rep_b = " ".join(
                w for w in chains[j]["representative"].lower().split()
                if w not in _STOP_WORDS
            )
            if _is_similar(rep_a, rep_b, threshold=0.5):
                current["entries"].extend(chains[j]["entries"])
                current["sessions"].extend(chains[j]["sessions"])
                used.add(j)
        current["sessions"] = sorted(set(current["sessions"]))
        current["span"] = (
            (max(current["sessions"]) - min(current["sessions"]))
            if current["sessions"]
            else 0
        )
        most_recent = max(current["sessions"]) if current["sessions"] else 0
        current["freshness"] = max(
            0.3, 1.0 - ((current_session - most_recent) * 0.05)
        )
        merged.append(current)

    # Step 5: Sort by freshness descending
    merged.sort(key=lambda c: c["freshness"], reverse=True)
    return merged


class SoulService:
    """Central synthesis engine for persistent personality and learning."""

    _CACHE_TTL = 300  # 5 minutes

    def __init__(self, db):
        """
        Args:
            db: DatabaseService instance with soul table methods
        """
        self.db = db
        self._state_cache: Dict[str, tuple] = {}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_get(self, project_path: str) -> Optional[Dict[str, Any]]:
        """Return cached soul state if within TTL, else None."""
        if project_path in self._state_cache:
            state, ts = self._state_cache[project_path]
            if time.time() - ts < self._CACHE_TTL:
                logger.debug("Soul state cache hit for %s", project_path)
                return state
            del self._state_cache[project_path]
        return None

    def _cache_set(self, project_path: str, state: Dict[str, Any]):
        """Store soul state in cache with current timestamp."""
        self._state_cache[project_path] = (state, time.time())

    def _cache_invalidate(self, project_path: str):
        """Remove cached soul state after mutation."""
        self._state_cache.pop(project_path, None)

    # ------------------------------------------------------------------
    # 1. generate_soul_brief — Called at session start
    # ------------------------------------------------------------------

    async def generate_soul_brief(self, project_path: str) -> str:
        """Generate a soul brief for session context injection.

        Reads soul_state for this project and returns a 200-400 word brief.
        Uses in-memory cache, no LLM — budget: < 1s.

        Returns:
            Formatted brief string, or empty string if no soul state exists.
        """
        # Try cache first
        state = self._cache_get(project_path)
        if state is None:
            state = await self.db.get_soul_state(project_path)
            if state:
                self._cache_set(project_path, state)

        if not state or not state.get("soul_brief"):
            return ""

        # Parse JSON fields safely
        user_model = _safe_json_loads(state.get("user_model", "{}"), {})
        project_understanding = _safe_json_loads(
            state.get("project_understanding", "{}"), {}
        )
        success_journal = _safe_json_loads(
            state.get("success_journal", "[]"), []
        )
        blind_spots = _safe_json_loads(state.get("blind_spots", "[]"), [])
        tool_preferences = _safe_json_loads(
            state.get("tool_preferences", "{}"), {}
        )
        integration_count = state.get("integration_count", 0)

        # Build brief from template
        parts = []
        parts.append("[SOUL CONTEXT — Claude Memory]")

        # Project header
        project_name = project_understanding.get(
            "name", project_path.split("/")[-1].split("\\")[-1]
        )
        parts.append(f"Project: {project_name}")
        parts.append(f"Sessions integrated: {integration_count}")
        if state.get("last_integrated_at"):
            parts.append(f"Last integration: {state['last_integrated_at']}")

        # User model section — freshness-sorted preferences
        if user_model:
            parts.append("\nYOU KNOW THIS USER:")
            preferences = user_model.get("preferences", [])
            sorted_prefs = _sort_by_freshness(preferences, integration_count)
            for pref in sorted_prefs[:5]:
                parts.append(f"- {pref}")
            dislikes = user_model.get("dislikes", [])
            for dislike in dislikes[-3:]:
                parts.append(f"- Dislikes: {dislike}")
            work_style = user_model.get("work_style", [])
            for ws in work_style[-3:]:
                parts.append(f"- {ws}")

        # Project understanding — decisions as chains
        if project_understanding:
            parts.append("\nPROJECT UNDERSTANDING:")
            tech = project_understanding.get("tech_choices", [])
            for t in tech[-4:]:
                parts.append(f"- {t}")
            arch_decisions = project_understanding.get(
                "architecture_decisions", []
            )
            if arch_decisions:
                chains = _build_decision_chains(
                    arch_decisions, integration_count
                )
                for chain in chains[:3]:
                    topics = [
                        _strip_session_tag(e) for e in chain["entries"]
                    ]
                    unique = list(dict.fromkeys(topics))
                    summary = " -> ".join(unique[-3:])
                    n_d = len(chain["entries"])
                    n_s = len(chain["sessions"])
                    parts.append(
                        f"- {summary} "
                        f"({n_d} decision{'s' if n_d != 1 else ''}, "
                        f"{n_s} session{'s' if n_s != 1 else ''})"
                    )
            focus = project_understanding.get("recent_focus", "")
            if focus:
                parts.append(f"- Recent focus: {focus}")

        # Success journal — freshness-sorted
        if success_journal:
            parts.append("\nRECENT LEARNINGS:")
            sorted_journal = _sort_by_freshness(
                success_journal, integration_count
            )
            for entry in sorted_journal[:4]:
                parts.append(f"- {entry}")

        # Blind spots — freshness-sorted
        if blind_spots:
            parts.append("\nACTIVE BLIND SPOTS:")
            sorted_spots = _sort_by_freshness(blind_spots, integration_count)
            for bs in sorted_spots[:3]:
                parts.append(f"- {bs}")

        # Tool preferences
        if tool_preferences:
            favored = tool_preferences.get("favored", [])
            if favored:
                parts.append(
                    f"\nTOOL PREFERENCES: {', '.join(favored[-5:])}"
                )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 2. capture_soul_fragment — Called on every Stop hook
    # ------------------------------------------------------------------

    async def capture_soul_fragment(
        self,
        session_id: str,
        fragment_type: str,
        content: str,
        project_path: str = "",
    ) -> Optional[int]:
        """Capture a soul fragment from a response.

        Lightweight — just stores in soul_fragments staging table.
        Budget: < 200ms.

        Args:
            session_id: Current session ID
            fragment_type: One of: decision_made, error_resolved,
                          preference_expressed, pattern_used,
                          correction_received
            content: The extracted fragment content
            project_path: Project path

        Returns:
            Fragment ID if stored, None on error.
        """
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
                    captured = (
                        match.group(1).strip()
                        if match.lastindex
                        else match.group(0).strip()
                    )
                    if (
                        len(captured) < 10
                        or len(captured) > MAX_FRAGMENT_LENGTH
                    ):
                        continue
                    key = captured[:50].lower()
                    if key in seen_content:
                        continue
                    seen_content.add(key)

                    fragments.append(
                        {
                            "fragment_type": fragment_type,
                            "content": captured[:MAX_FRAGMENT_LENGTH],
                        }
                    )

        return fragments

    # ------------------------------------------------------------------
    # 3. run_soul_integration — Called at session end
    # ------------------------------------------------------------------

    async def run_soul_integration(
        self, session_id: str, project_path: str
    ) -> Dict[str, Any]:
        """Integrate soul fragments from a session into persistent soul_state.

        Groups fragments by type, deduplicates via content fingerprinting,
        tags with session numbers for freshness tracking, merges into
        soul_state. Uses heuristics (no LLM). Budget: < 5s.

        Returns:
            Dict with integration stats.
        """
        fragments = await self.db.get_unintegrated_fragments(project_path)
        if not fragments:
            return {"integrated": 0, "message": "No fragments to integrate"}

        state = await self.db.get_soul_state(project_path) or {}
        user_model = _safe_json_loads(state.get("user_model", "{}"), {})
        project_understanding = _safe_json_loads(
            state.get("project_understanding", "{}"), {}
        )
        success_journal = _safe_json_loads(
            state.get("success_journal", "[]"), []
        )
        blind_spots = _safe_json_loads(state.get("blind_spots", "[]"), [])
        tool_preferences = _safe_json_loads(
            state.get("tool_preferences", "{}"), {}
        )
        integration_count = state.get("integration_count", 0) or 0
        session_num = integration_count + 1

        by_type: Dict[str, List[str]] = {}
        for frag in fragments:
            ft = frag["fragment_type"]
            by_type.setdefault(ft, []).append(frag["content"])

        # --- Merge decisions (fingerprint dedup, session-tagged) ---
        decisions = by_type.get("decision_made", [])
        if decisions:
            arch_decisions = project_understanding.get(
                "architecture_decisions", []
            )
            existing_fps = {
                _content_fingerprint(e): True for e in arch_decisions
            }
            for d in decisions:
                fp = _content_fingerprint(d)
                if fp not in existing_fps:
                    arch_decisions.append(f"[Session {session_num}] {d}")
                    existing_fps[fp] = True
            project_understanding["architecture_decisions"] = (
                arch_decisions[-20:]
            )

        # --- Merge preferences (fingerprint + fuzzy fallback, session-tagged) ---
        preferences = by_type.get("preference_expressed", [])
        if preferences:
            user_prefs = user_model.get("preferences", [])
            existing_fps = {
                _content_fingerprint(e): True for e in user_prefs
            }
            for p in preferences:
                fp = _content_fingerprint(p)
                if fp not in existing_fps and not any(
                    _is_similar(p, _strip_session_tag(existing))
                    for existing in user_prefs
                ):
                    user_prefs.append(f"[Session {session_num}] {p}")
                    existing_fps[fp] = True
            user_model["preferences"] = user_prefs[-15:]

        # --- Merge error resolutions (fingerprint dedup, session-tagged) ---
        errors = by_type.get("error_resolved", [])
        if errors:
            existing_fps = {
                _content_fingerprint(e): True for e in success_journal
            }
            for e in errors:
                fp = _content_fingerprint(e)
                if fp not in existing_fps:
                    success_journal.append(
                        f"[Session {session_num}] Fixed: {e}"
                    )
                    existing_fps[fp] = True
            success_journal = success_journal[-15:]

        # --- Merge patterns (fingerprint dedup, session-tagged) ---
        patterns = by_type.get("pattern_used", [])
        if patterns:
            existing_fps = {
                _content_fingerprint(e): True for e in success_journal
            }
            for p in patterns:
                fp = _content_fingerprint(p)
                if fp not in existing_fps:
                    success_journal.append(
                        f"[Session {session_num}] Pattern: {p}"
                    )
                    existing_fps[fp] = True
            success_journal = success_journal[-15:]

        # --- Merge corrections (fingerprint dedup, session-tagged) ---
        corrections = by_type.get("correction_received", [])
        if corrections:
            existing_fps = {
                _content_fingerprint(e): True for e in blind_spots
            }
            for c in corrections:
                fp = _content_fingerprint(c)
                if fp not in existing_fps:
                    blind_spots.append(f"[Session {session_num}] {c}")
                    existing_fps[fp] = True
            blind_spots = blind_spots[-10:]

        brief = await self._build_brief_text(
            project_path,
            user_model,
            project_understanding,
            success_journal,
            blind_spots,
            tool_preferences,
            session_num,
        )

        now = datetime.now().isoformat()
        updates = {
            "soul_brief": brief,
            "user_model": json.dumps(user_model),
            "project_understanding": json.dumps(project_understanding),
            "success_journal": json.dumps(success_journal),
            "blind_spots": json.dumps(blind_spots),
            "tool_preferences": json.dumps(tool_preferences),
            "last_integrated_at": now,
            "integration_count": session_num,
        }

        success = await self.db.upsert_soul_state(project_path, updates)

        # Invalidate cache after mutation
        self._cache_invalidate(project_path)

        integrated_count = 0
        if success:
            for frag in fragments:
                sid = frag.get("session_id", session_id)
                await self.db.mark_fragments_integrated(sid)
            integrated_count = len(fragments)

        return {
            "integrated": integrated_count,
            "by_type": {k: len(v) for k, v in by_type.items()},
            "integration_number": session_num,
            "success": success,
        }

    async def _build_brief_text(
        self,
        project_path: str,
        user_model: dict,
        project_understanding: dict,
        success_journal: list,
        blind_spots: list,
        tool_preferences: dict,
        integration_count: int,
    ) -> str:
        """Build the soul brief text from state components.

        Uses freshness sorting for all lists and renders decisions as
        evolution chains instead of flat entries.
        """
        parts = []
        project_name = project_understanding.get(
            "name",
            project_path.replace("\\", "/").rstrip("/").split("/")[-1],
        )

        parts.append(
            f"Project: {project_name} | Sessions: {integration_count}"
        )

        # Preferences — freshness-sorted
        if user_model.get("preferences"):
            sorted_prefs = _sort_by_freshness(
                user_model["preferences"], integration_count
            )
            prefs = sorted_prefs[:3]
            parts.append("User: " + "; ".join(prefs))

        # Decisions — rendered as chains
        if project_understanding.get("architecture_decisions"):
            chains = _build_decision_chains(
                project_understanding["architecture_decisions"],
                integration_count,
            )
            chain_strs = []
            for chain in chains[:2]:
                topics = [
                    _strip_session_tag(e) for e in chain["entries"]
                ]
                unique = list(dict.fromkeys(topics))
                summary = " -> ".join(unique[-3:])
                n_d = len(chain["entries"])
                n_s = len(chain["sessions"])
                chain_strs.append(
                    f"{summary} "
                    f"({n_d} decision{'s' if n_d != 1 else ''}, "
                    f"{n_s} session{'s' if n_s != 1 else ''})"
                )
            parts.append("Decisions: " + "; ".join(chain_strs))

        # Learnings — freshness-sorted
        if success_journal:
            sorted_journal = _sort_by_freshness(
                success_journal, integration_count
            )
            recent = sorted_journal[:2]
            parts.append("Learnings: " + "; ".join(recent))

        # Blind spots — freshness-sorted
        if blind_spots:
            sorted_spots = _sort_by_freshness(
                blind_spots, integration_count
            )
            recent = sorted_spots[:2]
            parts.append("Watch out: " + "; ".join(recent))

        return " | ".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # 4. enrich_with_soul — Called in memory_ask
    # ------------------------------------------------------------------

    async def enrich_with_soul(
        self, search_results: Dict[str, Any], project_path: str
    ) -> Dict[str, Any]:
        """Add soul context to search results.

        Uses in-memory cache, budget: < 200ms.

        Args:
            search_results: Existing search results dict from memory_ask
            project_path: Project to fetch soul for

        Returns:
            search_results dict with added 'soul_context' key.
        """
        if not project_path:
            return search_results

        # Try cache first
        state = self._cache_get(project_path)
        if state is None:
            state = await self.db.get_soul_state(project_path)
            if state:
                self._cache_set(project_path, state)

        if not state:
            return search_results

        soul_context = {}

        if state.get("soul_brief"):
            soul_context["brief"] = state["soul_brief"]

        user_model = _safe_json_loads(state.get("user_model", "{}"), {})
        if user_model.get("preferences"):
            soul_context["user_preferences"] = user_model["preferences"][-5:]

        blind_spots = _safe_json_loads(state.get("blind_spots", "[]"), [])
        if blind_spots:
            soul_context["blind_spots"] = blind_spots[-3:]

        success_journal = _safe_json_loads(
            state.get("success_journal", "[]"), []
        )
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
