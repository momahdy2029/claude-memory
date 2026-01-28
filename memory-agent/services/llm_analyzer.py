"""LLM-based text analysis service using Ollama."""
import os
import json
import ollama
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "llama3.2:3b")  # Small, fast model for analysis


class LLMAnalyzer:
    """Service for LLM-based text analysis using Ollama."""

    def __init__(self):
        self.model = ANALYSIS_MODEL
        self.client = ollama.Client(host=OLLAMA_HOST)

    async def extract_decisions_and_observations(
        self,
        response_text: str,
        max_decisions: int = 3,
        max_observations: int = 3
    ) -> Dict[str, Any]:
        """
        Extract decisions and observations from Claude's response using LLM.

        This is more accurate than regex-based detection because it understands
        context and implicit decisions.

        Args:
            response_text: The text to analyze
            max_decisions: Maximum number of decisions to extract
            max_observations: Maximum number of observations to extract

        Returns:
            Dict with 'decisions' and 'observations' lists
        """
        # Truncate very long responses
        text = response_text[:3000] if len(response_text) > 3000 else response_text

        prompt = f"""Analyze this AI assistant response and extract:

1. DECISIONS - Explicit or implicit choices made (e.g., "use X instead of Y", architecture choices, approach selections)
2. OBSERVATIONS - Things noticed or discovered (e.g., "found that X", "the issue is Y", bugs found)

Response to analyze:
---
{text}
---

Return JSON only, no explanation:
{{"decisions": ["decision 1", "decision 2"], "observations": ["observation 1"]}}

Rules:
- Each item should be a short phrase (under 100 chars)
- Max {max_decisions} decisions, {max_observations} observations
- If none found, return empty lists
- Only include clear, actionable items"""

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistency
                    "num_predict": 500   # Limit output length
                }
            )

            result_text = response.get("response", "{}")

            # Try to extract JSON from the response
            # Handle cases where model adds extra text
            json_start = result_text.find("{")
            json_end = result_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                result = json.loads(json_str)
                return {
                    "decisions": result.get("decisions", [])[:max_decisions],
                    "observations": result.get("observations", [])[:max_observations],
                    "success": True
                }

        except json.JSONDecodeError:
            pass
        except Exception as e:
            pass

        # Fallback: return empty
        return {
            "decisions": [],
            "observations": [],
            "success": False,
            "fallback": True
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

        facts_str = "\n".join(f"- {f}" for f in facts[:10])

        prompt = f"""Check if this statement contradicts any of the known facts.

KNOWN FACTS:
{facts_str}

STATEMENT TO CHECK:
{statement}

Return JSON only:
{{"has_contradiction": true/false, "conflicting_fact": "the fact it conflicts with or null", "reason": "brief explanation or null"}}"""

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "num_predict": 200
                }
            )

            result_text = response.get("response", "{}")

            json_start = result_text.find("{")
            json_end = result_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                return json.loads(json_str)

        except:
            pass

        return {"has_contradiction": False, "details": None, "error": True}

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

        prompt = f"""Summarize this session context in 2-3 sentences.

{goal_str}

Recent events:
{events_str}

Write a brief summary focusing on: what's being worked on, key decisions made, current status."""

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 150
                }
            )
            return response.get("response", "").strip()
        except:
            return f"Session with {len(events)} events. {goal_str}"
