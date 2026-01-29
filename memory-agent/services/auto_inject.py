"""Auto-injection service for mid-task relevance.

Analyzes current context and automatically retrieves relevant memories.
Can be called periodically or triggered by specific events.
"""
import asyncio
import re
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class InjectionContext:
    """Tracks what has been injected to avoid repetition."""
    injected_memory_ids: Set[int] = field(default_factory=set)
    last_query: str = ""
    last_injection_time: float = 0
    injection_count: int = 0


class AutoInjector:
    """Automatically injects relevant context during tasks.

    Features:
    - Analyzes current task/query for keywords
    - Searches memories for relevant context
    - Avoids injecting the same content twice
    - Rate-limits injections to avoid noise
    """

    def __init__(self, db, embeddings):
        self.db = db
        self.embeddings = embeddings
        self._context = InjectionContext()
        self._min_injection_interval = 30  # seconds between injections
        self._max_injections_per_session = 20

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'what', 'which', 'who', 'whom', 'how',
            'when', 'where', 'why', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not',
            'only', 'same', 'so', 'than', 'too', 'very', 'just', 'and',
            'but', 'or', 'if', 'then', 'else', 'for', 'of', 'to', 'in',
            'on', 'at', 'by', 'from', 'with', 'about', 'into', 'through',
            'please', 'help', 'me', 'want', 'need', 'let', 'make', 'get'
        }

        # Extract words
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())

        # Filter and prioritize
        keywords = []
        for word in words:
            if word not in stop_words and len(word) > 2:
                keywords.append(word)

        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords[:10]  # Top 10 keywords

    def _should_inject(self, query: str) -> bool:
        """Determine if we should inject context now."""
        now = time.time()

        # Rate limit
        if now - self._context.last_injection_time < self._min_injection_interval:
            return False

        # Max injections per session
        if self._context.injection_count >= self._max_injections_per_session:
            return False

        # Skip if query is too similar to last
        if query and self._context.last_query:
            # Simple similarity check
            query_words = set(query.lower().split())
            last_words = set(self._context.last_query.lower().split())
            overlap = len(query_words & last_words) / max(len(query_words), 1)
            if overlap > 0.8:
                return False

        return True

    async def get_relevant_context(
        self,
        current_query: str,
        project_path: Optional[str] = None,
        task_type: Optional[str] = None,
        max_results: int = 3
    ) -> Dict[str, Any]:
        """Get relevant context for the current task.

        Args:
            current_query: The current user query or task description
            project_path: Current project path
            task_type: Type of task (debug, implement, refactor, etc.)
            max_results: Maximum context items to return

        Returns:
            Dict with relevant memories, patterns, and suggestions
        """
        if not self._should_inject(current_query):
            return {"injected": False, "reason": "rate_limited"}

        keywords = self._extract_keywords(current_query)
        if not keywords:
            return {"injected": False, "reason": "no_keywords"}

        search_query = " ".join(keywords)
        results = {
            "injected": True,
            "keywords": keywords,
            "memories": [],
            "patterns": [],
            "warnings": []
        }

        # 1. Search for relevant memories
        try:
            from skills.search import semantic_search
            memories = await semantic_search(
                db=self.db,
                embeddings=self.embeddings,
                query=search_query,
                project_path=project_path,
                limit=max_results * 2,  # Get more, filter later
                threshold=0.6  # Higher threshold for relevance
            )

            if memories and memories.get("results"):
                for mem in memories["results"]:
                    mem_id = mem.get("id")
                    if mem_id and mem_id not in self._context.injected_memory_ids:
                        results["memories"].append({
                            "content": mem["content"][:300],
                            "type": mem.get("type"),
                            "relevance": mem.get("similarity", 0)
                        })
                        self._context.injected_memory_ids.add(mem_id)

                        if len(results["memories"]) >= max_results:
                            break
        except Exception:
            pass

        # 2. Search for relevant patterns
        try:
            from skills.search import search_patterns
            patterns = await search_patterns(
                db=self.db,
                embeddings=self.embeddings,
                query=search_query,
                limit=2,
                threshold=0.6
            )

            if patterns and patterns.get("patterns"):
                for pat in patterns["patterns"][:2]:
                    results["patterns"].append({
                        "name": pat.get("name"),
                        "solution": pat.get("solution", "")[:200]
                    })
        except Exception:
            pass

        # 3. Check for relevant errors (warnings)
        if task_type in ["debug", "fix", "error"]:
            try:
                errors = await semantic_search(
                    db=self.db,
                    embeddings=self.embeddings,
                    query=search_query,
                    project_path=project_path,
                    memory_type="error",
                    success_only=True,
                    limit=2,
                    threshold=0.65
                )

                if errors and errors.get("results"):
                    for err in errors["results"]:
                        results["warnings"].append({
                            "past_error": err["content"][:200],
                            "had_solution": err.get("success", False)
                        })
            except Exception:
                pass

        # Update context tracking
        self._context.last_query = current_query
        self._context.last_injection_time = time.time()
        self._context.injection_count += 1

        return results

    def format_injection(self, context: Dict[str, Any]) -> str:
        """Format context for injection into conversation."""
        if not context.get("injected"):
            return ""

        parts = []

        if context.get("memories"):
            parts.append("**Relevant from memory:**")
            for mem in context["memories"]:
                parts.append(f"- [{mem['type']}] {mem['content']}")

        if context.get("patterns"):
            parts.append("\n**Useful patterns:**")
            for pat in context["patterns"]:
                parts.append(f"- **{pat['name']}**: {pat['solution']}")

        if context.get("warnings"):
            parts.append("\n**Past related errors:**")
            for warn in context["warnings"]:
                parts.append(f"- {warn['past_error']}")

        if parts:
            return "\n".join(parts)
        return ""

    def reset_session(self):
        """Reset injection tracking for new session."""
        self._context = InjectionContext()


# Global injector instance
_injector: Optional[AutoInjector] = None


def get_auto_injector(db, embeddings) -> AutoInjector:
    """Get the global auto-injector instance."""
    global _injector
    if _injector is None:
        _injector = AutoInjector(db, embeddings)
    return _injector
