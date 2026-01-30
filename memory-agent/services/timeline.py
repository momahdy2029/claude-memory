"""Timeline service for session event tracking and management."""
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Flag to enable/disable LLM-based analysis
USE_LLM_ANALYSIS = os.getenv("USE_LLM_ANALYSIS", "true").lower() == "true"

# Session gap threshold - 4 hours
SESSION_GAP_SECONDS = int(os.getenv("SESSION_GAP_HOURS", "4")) * 60 * 60

# Checkpoint thresholds
CHECKPOINT_EVENT_THRESHOLD = int(os.getenv("CHECKPOINT_EVENT_THRESHOLD", "25"))
CHECKPOINT_TIME_THRESHOLD_MINUTES = int(os.getenv("CHECKPOINT_TIME_MINUTES", "15"))


class TimelineService:
    """Service for managing session timelines and event tracking."""

    def __init__(self, db_service, embedding_service):
        self.db = db_service
        self.embeddings = embedding_service

    # ============================================================
    # SESSION MANAGEMENT
    # ============================================================

    async def get_or_create_session(
        self,
        project_path: str
    ) -> tuple[str, bool, Optional[Dict[str, Any]]]:
        """
        Get current session or create new one if gap exceeded.

        Returns:
            tuple: (session_id, is_new_session, previous_session_state)
        """
        last_state = await self.db.get_latest_session_for_project(project_path)

        if last_state:
            last_activity = self._parse_datetime(last_state["last_activity_at"])
            gap = (datetime.now() - last_activity).total_seconds()

            if gap < SESSION_GAP_SECONDS:
                # Continue existing session
                return last_state["session_id"], False, None
            else:
                # Gap exceeded - create handoff checkpoint for old session
                await self._create_handoff_checkpoint(last_state)

        # Create new session
        new_session_id = str(uuid.uuid4())
        await self.db.get_or_create_session_state(new_session_id, project_path)

        return new_session_id, True, last_state

    async def _create_handoff_checkpoint(self, session_state: Dict[str, Any]):
        """Create a handoff checkpoint when session times out."""
        summary = f"Session paused after inactivity."
        if session_state.get("current_goal"):
            summary += f" Last goal: {session_state['current_goal']}"

        await self.db.store_checkpoint(
            session_id=session_state["session_id"],
            summary=summary,
            current_goal=session_state.get("current_goal"),
            entities=session_state.get("entity_registry"),
            pending_items=session_state.get("pending_questions")
        )

    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse ISO datetime string."""
        if not dt_str:
            return datetime.now()
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now()

    # ============================================================
    # EVENT LOGGING
    # ============================================================

    async def log_events_batch(
        self,
        session_id: str,
        events: List[Dict[str, Any]],
        project_path: Optional[str] = None,
        parent_event_id: Optional[int] = None,
        root_event_id: Optional[int] = None,
        generate_embeddings: bool = True
    ) -> List[int]:
        """
        Log multiple timeline events in a single batch operation.

        This is more efficient than calling log_event() multiple times because:
        1. Single database transaction for all events
        2. Batch embedding generation (if supported)
        3. Single checkpoint check at the end

        Args:
            session_id: The session ID
            events: List of event dicts, each containing:
                - event_type: Type of event (required)
                - summary: Brief description (required)
                - details: Full context (optional)
                - entities: Entity references (optional)
                - status: Event status (optional, default "completed")
                - outcome: Result or error message (optional)
                - confidence: Confidence level 0-1 (optional)
                - is_anchor: Whether this is a verified fact (optional)
            project_path: Project path for all events (optional)
            parent_event_id: ID of parent event for all events (optional)
            root_event_id: ID of root user request for all events (optional)
            generate_embeddings: Whether to generate embeddings (default True)

        Returns:
            List of event IDs in the same order as input events
        """
        if not events:
            return []

        event_ids = []

        # Generate embeddings in batch if enabled
        embeddings_list = []
        if generate_embeddings and self.embeddings:
            embed_texts = []
            for event in events:
                summary = event.get("summary", "")
                details = event.get("details", "")
                embed_text = summary
                if details:
                    embed_text += f"\n{details[:500]}"
                embed_texts.append(embed_text)

            # Try batch embedding if available, otherwise fall back to sequential
            try:
                if hasattr(self.embeddings, 'generate_embeddings_batch'):
                    embeddings_list = await self.embeddings.generate_embeddings_batch(embed_texts)
                else:
                    # Fall back to sequential embedding generation
                    for text in embed_texts:
                        emb = await self.embeddings.generate_embedding(text)
                        embeddings_list.append(emb)
            except Exception:
                # If embedding fails, continue without embeddings
                embeddings_list = [None] * len(events)
        else:
            embeddings_list = [None] * len(events)

        # Store all events (database service should handle transaction)
        for i, event in enumerate(events):
            event_id = await self.db.store_timeline_event(
                session_id=session_id,
                event_type=event.get("event_type", "observation"),
                summary=event.get("summary", "")[:200],
                details=event.get("details"),
                embedding=embeddings_list[i] if i < len(embeddings_list) else None,
                project_path=project_path,
                parent_event_id=parent_event_id,
                root_event_id=root_event_id,
                entities=event.get("entities"),
                status=event.get("status", "completed"),
                outcome=event.get("outcome"),
                confidence=event.get("confidence"),
                is_anchor=event.get("is_anchor", False)
            )
            event_ids.append(event_id)

        # Single checkpoint check after all events (not per-event)
        await self._maybe_create_auto_checkpoint(session_id, project_path)

        return event_ids

    async def log_event(
        self,
        session_id: str,
        event_type: str,
        summary: str,
        details: Optional[str] = None,
        project_path: Optional[str] = None,
        parent_event_id: Optional[int] = None,
        root_event_id: Optional[int] = None,
        entities: Optional[Dict[str, List[str]]] = None,
        status: str = "completed",
        outcome: Optional[str] = None,
        confidence: Optional[float] = None,
        is_anchor: bool = False,
        generate_embedding: bool = True
    ) -> int:
        """
        Log a timeline event.

        Args:
            session_id: The session ID
            event_type: Type of event (user_request, decision, action, observation, error, checkpoint)
            summary: Brief description (<200 chars)
            details: Full context (optional)
            project_path: Project path (optional)
            parent_event_id: ID of parent event (causal chain)
            root_event_id: ID of root user request
            entities: Dict of entity references {"files": [], "functions": [], etc.}
            status: Event status (pending, in_progress, completed, failed)
            outcome: Result or error message
            confidence: Confidence level 0-1
            is_anchor: Whether this is a verified fact
            generate_embedding: Whether to generate embedding for semantic search

        Returns:
            Event ID
        """
        embedding = None
        if generate_embedding and self.embeddings:
            # Generate embedding from summary + details for semantic search
            embed_text = summary
            if details:
                embed_text += f"\n{details[:500]}"  # Limit details for embedding
            embedding = await self.embeddings.generate_embedding(embed_text)

        event_id = await self.db.store_timeline_event(
            session_id=session_id,
            event_type=event_type,
            summary=summary,
            details=details,
            embedding=embedding,
            project_path=project_path,
            parent_event_id=parent_event_id,
            root_event_id=root_event_id,
            entities=entities,
            status=status,
            outcome=outcome,
            confidence=confidence,
            is_anchor=is_anchor
        )

        # Check if checkpoint is needed
        await self._maybe_create_auto_checkpoint(session_id, project_path)

        return event_id

    async def get_events(
        self,
        session_id: str,
        limit: int = 20,
        event_type: Optional[str] = None,
        since_event_id: Optional[int] = None,
        anchors_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get timeline events for a session."""
        return await self.db.get_timeline_events(
            session_id=session_id,
            limit=limit,
            event_type=event_type,
            since_event_id=since_event_id,
            anchors_only=anchors_only
        )

    async def search_events(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Semantic search across timeline events."""
        if not self.embeddings:
            return []

        embedding = await self.embeddings.generate_embedding(query)
        return await self.db.search_timeline_events(
            embedding=embedding,
            session_id=session_id,
            limit=limit,
            threshold=threshold
        )

    # ============================================================
    # AUTO-DETECTION (Parse responses for decisions/observations)
    # ============================================================

    # Decision patterns
    DECISION_PATTERNS = [
        r"I'll use (.+?) instead of",
        r"Let's go with (.+)",
        r"The best approach is (.+)",
        r"I've decided to (.+)",
        r"I'm going to (.+)",
        r"We should (.+)",
        r"I recommend (.+)",
    ]

    # Observation patterns
    OBSERVATION_PATTERNS = [
        r"I notice that (.+)",
        r"Found: (.+)",
        r"The issue is (.+)",
        r"Looking at .+?, I see (.+)",
        r"The problem is (.+)",
        r"This shows that (.+)",
        r"It appears that (.+)",
        # File structure discoveries
        r"The (?:file|directory|folder) (?:structure|layout) shows (.+)",
        r"There (?:is|are) (\d+ (?:files?|directories|folders).+)",
        r"The codebase (?:has|contains|includes) (.+)",
        # Error encounters
        r"(?:An? )?[Ee]rror (?:occurred|happened): (.+)",
        r"(?:The )?(?:test|build|command) failed (?:because|with) (.+)",
        r"There's an? (?:issue|bug|error) (?:in|with) (.+)",
        # Configuration findings
        r"The config(?:uration)? (?:shows|indicates|has) (.+)",
        r"(?:This|The) (?:setting|option|flag) (?:is set to|controls|enables) (.+)",
        r"The (?:database|server|API) is (?:configured to|set up for|running) (.+)",
        # Pattern discoveries
        r"(?:The )?code (?:follows|uses) (?:the )?(.+) pattern",
        r"(?:This|The) (?:function|method|class) (?:implements|handles|manages) (.+)",
    ]

    def detect_decisions(self, text: str) -> List[str]:
        """Detect decisions from text."""
        decisions = []
        for pattern in self.DECISION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            decisions.extend(matches)
        return decisions

    def detect_observations(self, text: str) -> List[str]:
        """Detect observations from text."""
        observations = []
        for pattern in self.OBSERVATION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            observations.extend(matches)
        return observations

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entity references from text."""
        entities = {
            "files": [],
            "functions": [],
            "variables": [],
            "urls": []
        }

        # File patterns (common extensions)
        file_pattern = r'[\w\-./\\]+\.(py|js|ts|tsx|jsx|json|md|yaml|yml|toml|sql|html|css|scss)'
        entities["files"] = list(set(re.findall(file_pattern, text)))

        # Function/method patterns
        func_pattern = r'(?:function|def|async def|class)\s+(\w+)'
        entities["functions"] = list(set(re.findall(func_pattern, text)))

        # Variable assignment patterns
        var_pattern = r'(\w+)\s*[=:]\s*[^=]'
        potential_vars = re.findall(var_pattern, text)
        # Filter common keywords
        keywords = {'if', 'else', 'for', 'while', 'return', 'class', 'def', 'async', 'await', 'import', 'from'}
        entities["variables"] = [v for v in potential_vars if v.lower() not in keywords][:10]

        # URL patterns
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        entities["urls"] = list(set(re.findall(url_pattern, text)))

        # Clean up empty lists
        return {k: v for k, v in entities.items() if v}

    async def auto_log_from_response(
        self,
        session_id: str,
        response_text: str,
        project_path: Optional[str] = None,
        parent_event_id: Optional[int] = None,
        root_event_id: Optional[int] = None
    ) -> List[int]:
        """
        Auto-detect and log decisions/observations from a response.

        Uses LLM-based analysis when available, falls back to regex.

        Args:
            session_id: The session ID
            response_text: Claude's response text to analyze
            project_path: Project path (optional)
            parent_event_id: Parent event ID for causal chain (optional)
            root_event_id: Root user request event ID (optional)

        Returns list of created event IDs.
        """
        event_ids = []
        decisions = []
        observations = []

        # Try LLM-based analysis first (more accurate)
        if USE_LLM_ANALYSIS:
            try:
                from services.llm_analyzer import LLMAnalyzer
                analyzer = LLMAnalyzer()
                result = await analyzer.extract_decisions_and_observations(
                    response_text,
                    max_decisions=3,
                    max_observations=3
                )
                if result.get("success"):
                    decisions = result.get("decisions", [])
                    observations = result.get("observations", [])
            except Exception as e:
                # LLM not available, fall back to regex
                pass

        # Fall back to regex-based detection if LLM didn't find anything
        if not decisions:
            decisions = self.detect_decisions(response_text)
        if not observations:
            observations = self.detect_observations(response_text)

        # Log decisions (higher confidence if from LLM)
        confidence_boost = 0.1 if USE_LLM_ANALYSIS else 0.0
        for decision in decisions[:3]:  # Limit to top 3
            event_id = await self.log_event(
                session_id=session_id,
                event_type="decision",
                summary=decision[:200],
                project_path=project_path,
                parent_event_id=parent_event_id,
                root_event_id=root_event_id,
                confidence=0.7 + confidence_boost,  # Higher if LLM-detected
                generate_embedding=True
            )
            event_ids.append(event_id)

        # Log observations
        for observation in observations[:3]:  # Limit to top 3
            event_id = await self.log_event(
                session_id=session_id,
                event_type="observation",
                summary=observation[:200],
                project_path=project_path,
                parent_event_id=parent_event_id,
                root_event_id=root_event_id,
                confidence=0.6 + confidence_boost,
                generate_embedding=True
            )
            event_ids.append(event_id)

        return event_ids

    # ============================================================
    # AUTO-CHECKPOINT
    # ============================================================

    async def _maybe_create_auto_checkpoint(
        self,
        session_id: str,
        project_path: Optional[str] = None
    ):
        """Check if automatic checkpoint is needed and create if so."""
        state = await self.db.get_or_create_session_state(session_id, project_path)

        events_since = state.get("events_since_checkpoint", 0)

        if events_since >= CHECKPOINT_EVENT_THRESHOLD:
            await self.create_checkpoint(
                session_id=session_id,
                auto_generated=True
            )

    async def create_checkpoint(
        self,
        session_id: str,
        summary: Optional[str] = None,
        auto_generated: bool = False
    ) -> int:
        """Create a checkpoint for the session."""
        state = await self.db.get_or_create_session_state(session_id)

        # Get recent events for summary
        recent_events = await self.db.get_timeline_events(
            session_id=session_id,
            limit=state.get("events_since_checkpoint", 25)
        )

        # Build summary if not provided
        if not summary:
            summary = self._generate_checkpoint_summary(recent_events, state, auto_generated)

        # Extract key facts (anchors and high-confidence decisions)
        key_facts = [
            e["summary"] for e in recent_events
            if e.get("is_anchor") or (e.get("event_type") == "decision" and e.get("confidence", 0) >= 0.8)
        ]

        # Extract decisions
        decisions = [
            e["summary"] for e in recent_events
            if e.get("event_type") == "decision"
        ]

        # Get last event ID
        event_id = recent_events[0]["id"] if recent_events else None

        checkpoint_id = await self.db.store_checkpoint(
            session_id=session_id,
            summary=summary,
            event_id=event_id,
            key_facts=key_facts[:10],  # Limit to top 10
            decisions=decisions[:10],
            entities=state.get("entity_registry"),
            current_goal=state.get("current_goal"),
            pending_items=state.get("pending_questions"),
            event_count=len(recent_events)
        )

        return checkpoint_id

    def _generate_checkpoint_summary(
        self,
        events: List[Dict[str, Any]],
        state: Dict[str, Any],
        auto_generated: bool
    ) -> str:
        """Generate a summary for auto-checkpoint."""
        parts = []

        if auto_generated:
            parts.append(f"Auto-checkpoint ({len(events)} events)")

        if state.get("current_goal"):
            parts.append(f"Goal: {state['current_goal']}")

        # Count event types
        type_counts = {}
        for e in events:
            t = e.get("event_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        if type_counts:
            type_summary = ", ".join(f"{c} {t}s" for t, c in type_counts.items())
            parts.append(f"Activity: {type_summary}")

        return ". ".join(parts) if parts else "Checkpoint created"

    # ============================================================
    # CONTEXT LOADING (for session resume)
    # ============================================================

    async def load_session_context(
        self,
        session_id: str,
        include_checkpoint: bool = True,
        include_recent_events: int = 10
    ) -> Dict[str, Any]:
        """
        Load full context for a session.

        Returns dict with state, recent events, and latest checkpoint.
        """
        state = await self.db.get_or_create_session_state(session_id)

        context = {
            "session_id": session_id,
            "state": state,
            "recent_events": [],
            "checkpoint": None
        }

        if include_recent_events > 0:
            context["recent_events"] = await self.db.get_timeline_events(
                session_id=session_id,
                limit=include_recent_events
            )

        if include_checkpoint:
            context["checkpoint"] = await self.db.get_latest_checkpoint(session_id)

        return context
