"""Enhanced LLM-based text analysis service using Ollama.

Features:
- Structured extraction of decisions, observations, errors, learnings
- Confidence scores for extracted items
- Caching to avoid re-analyzing the same content
- Pattern deduplication using embeddings
- Fallback to regex when LLM unavailable
"""
import os
import re
import json
import time
import hashlib
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

# Try to import ollama
OLLAMA_AVAILABLE = False
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "llama3.2:3b")
USE_LLM_ANALYSIS = os.getenv("USE_LLM_ANALYSIS", "true").lower() == "true"
CACHE_TTL = int(os.getenv("LLM_CACHE_TTL", "3600"))  # 1 hour default


class AnalysisCache:
    """Simple in-memory cache for analysis results."""

    def __init__(self, max_size: int = 1000, ttl: int = CACHE_TTL):
        self._cache: Dict[str, Tuple[Dict, float]] = {}
        self._max_size = max_size
        self._ttl = ttl

    def _hash_key(self, text: str, analysis_type: str) -> str:
        """Generate a cache key from text and analysis type."""
        content = f"{analysis_type}:{text[:500]}"  # Use first 500 chars for key
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, analysis_type: str) -> Optional[Dict]:
        """Get cached result if available and not expired."""
        key = self._hash_key(text, analysis_type)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return result
            else:
                del self._cache[key]
        return None

    def set(self, text: str, analysis_type: str, result: Dict):
        """Cache an analysis result."""
        # Evict oldest entries if cache is full
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        key = self._hash_key(text, analysis_type)
        self._cache[key] = (result, time.time())

    def clear(self):
        """Clear the cache."""
        self._cache.clear()

    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl": self._ttl
        }


# Regex patterns for fallback detection
DECISION_PATTERNS = [
    r"(?:I(?:'ll| will)|Let(?:'s| me)|Going to|Decided to|Choosing|Using|Will use)\s+(.+?)(?:\.|$)",
    r"(?:better to|should|recommend)\s+(.+?)(?:\.|$)",
    r"(?:instead of|rather than)\s+(.+?),?\s+(?:I'll|we'll|let's)\s+(.+?)(?:\.|$)",
]

OBSERVATION_PATTERNS = [
    r"(?:I (?:notice|see|found|discovered)|Found that|The issue is|Problem is)\s+(.+?)(?:\.|$)",
    r"(?:Looking at|Checking|Examining)\s+.+?,?\s+(.+?)(?:\.|$)",
    r"(?:It (?:appears|seems|looks like))\s+(.+?)(?:\.|$)",
]

ERROR_PATTERNS = [
    r"(?:Error|Exception|Failed|Failure):\s*(.+?)(?:\.|$)",
    r"(?:Bug|Issue|Problem)(?:\s+(?:is|was|with))?\s*:?\s*(.+?)(?:\.|$)",
    r"(?:doesn't work|not working|broken|failing)\s*(?:because|due to)?\s*(.+?)(?:\.|$)",
]

LEARNING_PATTERNS = [
    r"(?:Learned|Discovered|Realized|TIL|Note to self)\s*:?\s*(.+?)(?:\.|$)",
    r"(?:Key (?:insight|learning|takeaway))\s*:?\s*(.+?)(?:\.|$)",
    r"(?:Remember|Important)\s*:?\s*(.+?)(?:\.|$)",
]


class LLMAnalyzer:
    """Enhanced service for LLM-based text analysis.

    Features:
    - Structured extraction with confidence scores
    - Caching to avoid duplicate analysis
    - Fallback to regex patterns when LLM unavailable
    - Health checking and graceful degradation
    """

    def __init__(self):
        self.model = ANALYSIS_MODEL
        self.host = OLLAMA_HOST
        self.use_llm = USE_LLM_ANALYSIS and OLLAMA_AVAILABLE
        self._client = None
        self._cache = AnalysisCache()
        self._degraded_mode = False
        self._last_health_check = 0
        self._health_cache_ttl = 60  # Check health every 60s

    @property
    def client(self):
        """Lazy-load the Ollama client."""
        if self._client is None and OLLAMA_AVAILABLE:
            self._client = ollama.Client(host=self.host)
        return self._client

    async def check_health(self) -> bool:
        """Check if LLM service is available."""
        if not OLLAMA_AVAILABLE:
            return False

        now = time.time()
        if now - self._last_health_check < self._health_cache_ttl:
            return not self._degraded_mode

        try:
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.client.list()),
                timeout=2.0
            )
            self._degraded_mode = False
            self._last_health_check = now
            return True
        except Exception:
            self._degraded_mode = True
            self._last_health_check = now
            return False

    def _extract_with_regex(
        self,
        text: str,
        patterns: List[str],
        max_items: int = 3
    ) -> List[Dict[str, Any]]:
        """Extract items using regex patterns (fallback method)."""
        results = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                # Handle tuple results from groups
                if isinstance(match, tuple):
                    match = " ".join(m for m in match if m)
                if match and len(match) > 10:
                    results.append({
                        "content": match.strip()[:200],
                        "confidence": 0.5,  # Lower confidence for regex
                        "method": "regex"
                    })
                if len(results) >= max_items:
                    break
            if len(results) >= max_items:
                break
        return results

    async def extract_patterns(
        self,
        response_text: str,
        max_decisions: int = 3,
        max_observations: int = 3,
        max_errors: int = 2,
        max_learnings: int = 2
    ) -> Dict[str, Any]:
        """
        Extract all pattern types from Claude's response.

        Args:
            response_text: The text to analyze
            max_decisions: Maximum decisions to extract
            max_observations: Maximum observations to extract
            max_errors: Maximum errors to extract
            max_learnings: Maximum learnings to extract

        Returns:
            Dict with 'decisions', 'observations', 'errors', 'learnings' lists
            Each item has 'content', 'confidence', and 'method' fields
        """
        # Check cache first
        cached = self._cache.get(response_text, "patterns")
        if cached:
            return {**cached, "cached": True}

        # Truncate very long responses
        text = response_text[:4000] if len(response_text) > 4000 else response_text

        # Try LLM analysis first
        if self.use_llm and not self._degraded_mode:
            result = await self._extract_with_llm(
                text, max_decisions, max_observations, max_errors, max_learnings
            )
            if result.get("success"):
                self._cache.set(response_text, "patterns", result)
                return result

        # Fallback to regex
        result = {
            "decisions": self._extract_with_regex(text, DECISION_PATTERNS, max_decisions),
            "observations": self._extract_with_regex(text, OBSERVATION_PATTERNS, max_observations),
            "errors": self._extract_with_regex(text, ERROR_PATTERNS, max_errors),
            "learnings": self._extract_with_regex(text, LEARNING_PATTERNS, max_learnings),
            "success": True,
            "method": "regex",
            "degraded_mode": True
        }

        self._cache.set(response_text, "patterns", result)
        return result

    async def _extract_with_llm(
        self,
        text: str,
        max_decisions: int,
        max_observations: int,
        max_errors: int,
        max_learnings: int
    ) -> Dict[str, Any]:
        """Extract patterns using LLM analysis."""
        prompt = f"""Analyze this AI assistant response and extract structured information.

RESPONSE TO ANALYZE:
---
{text}
---

Extract the following (if present):
1. DECISIONS - Choices made about implementation, architecture, or approach
2. OBSERVATIONS - Things noticed, discovered, or found
3. ERRORS - Bugs, issues, or problems identified
4. LEARNINGS - Insights, lessons learned, or important notes

Return JSON only with this exact structure:
{{
    "decisions": [
        {{"content": "short description", "confidence": 0.0-1.0}}
    ],
    "observations": [
        {{"content": "short description", "confidence": 0.0-1.0}}
    ],
    "errors": [
        {{"content": "short description", "confidence": 0.0-1.0}}
    ],
    "learnings": [
        {{"content": "short description", "confidence": 0.0-1.0}}
    ]
}}

Rules:
- Each content should be a clear, concise phrase (under 150 chars)
- Confidence: 0.9+ for explicit statements, 0.7-0.9 for implicit, 0.5-0.7 for inferred
- Max items: {max_decisions} decisions, {max_observations} observations, {max_errors} errors, {max_learnings} learnings
- Return empty arrays [] if nothing found for a category
- Only include meaningful, actionable items"""

        try:
            loop = asyncio.get_event_loop()

            def _generate():
                return self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        "temperature": 0.1,
                        "num_predict": 800
                    }
                )

            response = await asyncio.wait_for(
                loop.run_in_executor(None, _generate),
                timeout=30.0
            )

            result_text = response.get("response", "{}")

            # Extract JSON from response
            json_start = result_text.find("{")
            json_end = result_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                parsed = json.loads(json_str)

                # Normalize the results
                def normalize_items(items: List, max_count: int) -> List[Dict]:
                    normalized = []
                    for item in (items or [])[:max_count]:
                        if isinstance(item, str):
                            normalized.append({
                                "content": item[:200],
                                "confidence": 0.7,
                                "method": "llm"
                            })
                        elif isinstance(item, dict):
                            normalized.append({
                                "content": str(item.get("content", item))[:200],
                                "confidence": float(item.get("confidence", 0.7)),
                                "method": "llm"
                            })
                    return normalized

                return {
                    "decisions": normalize_items(parsed.get("decisions"), max_decisions),
                    "observations": normalize_items(parsed.get("observations"), max_observations),
                    "errors": normalize_items(parsed.get("errors"), max_errors),
                    "learnings": normalize_items(parsed.get("learnings"), max_learnings),
                    "success": True,
                    "method": "llm"
                }

        except asyncio.TimeoutError:
            self._degraded_mode = True
        except json.JSONDecodeError:
            pass
        except Exception:
            self._degraded_mode = True

        return {"success": False}

    async def extract_decisions_and_observations(
        self,
        response_text: str,
        max_decisions: int = 3,
        max_observations: int = 3
    ) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        Extracts decisions and observations from Claude's response.
        """
        result = await self.extract_patterns(
            response_text,
            max_decisions=max_decisions,
            max_observations=max_observations,
            max_errors=0,
            max_learnings=0
        )

        return {
            "decisions": [d["content"] for d in result.get("decisions", [])],
            "observations": [o["content"] for o in result.get("observations", [])],
            "success": result.get("success", False),
            "fallback": result.get("method") == "regex"
        }

    async def check_statement_against_facts(
        self,
        statement: str,
        facts: List[str]
    ) -> Dict[str, Any]:
        """
        Check if a statement contradicts known facts using LLM.

        Args:
            statement: The statement to check
            facts: List of known facts/anchors

        Returns:
            Dict with contradiction analysis
        """
        if not facts:
            return {"has_contradiction": False, "details": None}

        # Check cache
        cache_key = f"{statement}|{','.join(facts[:5])}"
        cached = self._cache.get(cache_key, "contradiction")
        if cached:
            return {**cached, "cached": True}

        facts_str = "\n".join(f"- {f}" for f in facts[:10])

        # Try LLM first
        if self.use_llm and not self._degraded_mode:
            prompt = f"""Check if this statement contradicts any of the known facts.

KNOWN FACTS:
{facts_str}

STATEMENT TO CHECK:
{statement}

Return JSON only:
{{"has_contradiction": true/false, "conflicting_fact": "the fact it conflicts with or null", "reason": "brief explanation or null", "confidence": 0.0-1.0}}"""

            try:
                loop = asyncio.get_event_loop()

                def _generate():
                    return self.client.generate(
                        model=self.model,
                        prompt=prompt,
                        options={"temperature": 0.1, "num_predict": 200}
                    )

                response = await asyncio.wait_for(
                    loop.run_in_executor(None, _generate),
                    timeout=15.0
                )

                result_text = response.get("response", "{}")
                json_start = result_text.find("{")
                json_end = result_text.rfind("}") + 1

                if json_start >= 0 and json_end > json_start:
                    json_str = result_text[json_start:json_end]
                    result = json.loads(json_str)
                    result["method"] = "llm"
                    self._cache.set(cache_key, "contradiction", result)
                    return result

            except Exception:
                pass

        # Fallback: simple keyword matching
        statement_lower = statement.lower()
        for fact in facts:
            fact_lower = fact.lower()
            # Check for negation patterns
            if "not" in statement_lower and any(
                word in fact_lower for word in statement_lower.split() if len(word) > 3
            ):
                result = {
                    "has_contradiction": True,
                    "conflicting_fact": fact,
                    "reason": "Potential negation detected",
                    "confidence": 0.5,
                    "method": "regex"
                }
                self._cache.set(cache_key, "contradiction", result)
                return result

        result = {"has_contradiction": False, "details": None, "method": "regex"}
        self._cache.set(cache_key, "contradiction", result)
        return result

    async def summarize_session_context(
        self,
        events: List[Dict[str, Any]],
        current_goal: Optional[str] = None
    ) -> str:
        """
        Generate a concise summary of session context.

        Args:
            events: List of timeline events
            current_goal: Current session goal

        Returns:
            Concise summary string
        """
        events_str = "\n".join(
            f"- [{e.get('event_type', '?')}] {e.get('summary', '')}"
            for e in events[:15]
        )

        goal_str = f"Goal: {current_goal}" if current_goal else "No explicit goal set"

        # Try LLM first
        if self.use_llm and not self._degraded_mode:
            prompt = f"""Summarize this session context in 2-3 sentences.

{goal_str}

Recent events:
{events_str}

Write a brief summary focusing on: what's being worked on, key decisions made, current status."""

            try:
                loop = asyncio.get_event_loop()

                def _generate():
                    return self.client.generate(
                        model=self.model,
                        prompt=prompt,
                        options={"temperature": 0.3, "num_predict": 150}
                    )

                response = await asyncio.wait_for(
                    loop.run_in_executor(None, _generate),
                    timeout=15.0
                )
                return response.get("response", "").strip()

            except Exception:
                pass

        # Fallback
        return f"Session with {len(events)} events. {goal_str}"

    async def deduplicate_patterns(
        self,
        patterns: List[Dict[str, Any]],
        embeddings_service,
        threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Remove semantically duplicate patterns using embeddings.

        Args:
            patterns: List of pattern dicts with 'content' field
            embeddings_service: Embedding service for similarity check
            threshold: Similarity threshold for considering duplicates

        Returns:
            Deduplicated list of patterns
        """
        if len(patterns) <= 1:
            return patterns

        unique = []
        embeddings = []

        for pattern in patterns:
            content = pattern.get("content", "")
            if not content:
                continue

            # Generate embedding
            embedding = await embeddings_service.generate_embedding(content)
            if embedding is None:
                unique.append(pattern)
                continue

            # Check similarity with existing patterns
            is_duplicate = False
            for existing_emb in embeddings:
                # Cosine similarity
                import numpy as np
                a = np.array(embedding)
                b = np.array(existing_emb)
                similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

                if similarity >= threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(pattern)
                embeddings.append(embedding)

        return unique

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "model": self.model,
            "host": self.host,
            "use_llm": self.use_llm,
            "ollama_available": OLLAMA_AVAILABLE,
            "degraded_mode": self._degraded_mode,
            "cache": self._cache.stats()
        }

    def clear_cache(self):
        """Clear the analysis cache."""
        self._cache.clear()


# Global instance
_analyzer: Optional[LLMAnalyzer] = None


def get_analyzer() -> LLMAnalyzer:
    """Get the global LLM analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = LLMAnalyzer()
    return _analyzer
