"""Natural language interface for memory system.

Allows users to interact with memory using natural commands like:
- "remember this: ..."
- "what did I learn about X?"
- "forget about Y"
- "show me past errors"
"""
import re
from typing import Dict, Any, Optional, Tuple


# Intent patterns - ordered from most specific to least specific
# More specific patterns (list_errors, list_decisions, etc.) must come before generic "search"
INTENT_PATTERNS = {
    "list_errors": [
        r"(?:show|list|get)\s+(?:me\s+)?(?:past\s+)?errors?",
        r"what\s+errors?\s+(?:have\s+I\s+)?(?:had|seen|encountered)",
        r"(?:past|recent)\s+errors?",
    ],
    "list_decisions": [
        r"(?:show|list|get)\s+(?:me\s+)?(?:past\s+)?decisions?",
        r"what\s+(?:have\s+I\s+)?decided",
        r"(?:past|recent)\s+decisions?",
    ],
    "list_patterns": [
        r"(?:show|list|get)\s+(?:me\s+)?patterns?",
        r"what\s+patterns?\s+(?:do\s+I\s+)?(?:have|know)",
        r"useful\s+patterns?",
    ],
    "stats": [
        r"(?:memory\s+)?stats?",
        r"how\s+(?:many|much)\s+(?:do\s+I\s+)?remember",
        r"memory\s+(?:status|info|summary)",
    ],
    "project_context": [
        r"(?:what|tell\s+me)\s+about\s+(?:this\s+)?project",
        r"project\s+(?:info|context|summary)",
        r"current\s+project",
    ],
    "store": [
        r"remember\s+(?:this|that)?[:\s]*(.+)",
        r"save\s+(?:this|that)?[:\s]*(.+)",
        r"store\s+(?:this|that)?[:\s]*(.+)",
        r"note\s+(?:this|that)?[:\s]*(.+)",
        r"keep\s+(?:in\s+mind)?[:\s]*(.+)",
    ],
    "forget": [
        r"forget\s+(?:about\s+)?(.+)",
        r"delete\s+(?:memory\s+about\s+)?(.+)",
        r"remove\s+(?:memory\s+about\s+)?(.+)",
        r"clear\s+(?:memory\s+about\s+)?(.+)",
    ],
    "search": [
        r"what\s+(?:did\s+I|do\s+I|have\s+I)\s+(?:learn|know|remember)\s+about\s+(.+)",
        r"show\s+me\s+(?:memories?\s+about\s+)?(.+)",
        r"find\s+(?:memories?\s+about\s+)?(.+)",
        r"search\s+(?:for\s+)?(.+)",
        r"recall\s+(.+)",
        r"what\s+about\s+(.+)",
    ],
}


def parse_intent(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse natural language to determine intent and extract content.

    Returns:
        Tuple of (intent, extracted_content)
    """
    text = text.strip().lower()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract captured group if any
                content = match.group(1).strip() if match.groups() else None
                return intent, content

    return None, None


async def process_natural_command(
    db,
    embeddings,
    command: str,
    project_path: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Process a natural language memory command.

    Args:
        db: Database service
        embeddings: Embeddings service
        command: Natural language command
        project_path: Current project path
        session_id: Current session ID

    Returns:
        Result of the processed command
    """
    intent, content = parse_intent(command)

    if not intent:
        return {
            "success": False,
            "understood": False,
            "message": "I didn't understand that memory command. Try:\n"
                      "- 'remember this: [content]'\n"
                      "- 'what did I learn about [topic]?'\n"
                      "- 'show me past errors'\n"
                      "- 'memory stats'"
        }

    result = {"success": True, "understood": True, "intent": intent}

    if intent == "store":
        if not content:
            return {"success": False, "message": "What should I remember?"}

        from skills.store import store_memory
        store_result = await store_memory(
            db=db,
            embeddings=embeddings,
            content=content,
            memory_type="chunk",
            project_path=project_path,
            session_id=session_id,
            importance=6,
            tags=["natural-language", "user-stored"]
        )
        result["memory_id"] = store_result.get("memory_id")
        result["message"] = f"Got it! I'll remember that. (Memory #{result['memory_id']})"

    elif intent == "search":
        if not content:
            return {"success": False, "message": "What should I search for?"}

        from skills.search import semantic_search
        search_result = await semantic_search(
            db=db,
            embeddings=embeddings,
            query=content,
            project_path=project_path,
            limit=5
        )

        results = search_result.get("results", [])
        if results:
            result["results"] = results
            result["message"] = f"Found {len(results)} related memories:\n"
            for i, r in enumerate(results[:3], 1):
                result["message"] += f"\n{i}. {r['content'][:150]}..."
        else:
            result["message"] = f"No memories found about '{content}'"

    elif intent == "forget":
        if not content:
            return {"success": False, "message": "What should I forget?"}

        # Search and mark for deletion (soft delete via archive)
        from skills.search import semantic_search
        search_result = await semantic_search(
            db=db,
            embeddings=embeddings,
            query=content,
            project_path=project_path,
            limit=3,
            threshold=0.7  # High threshold to be sure
        )

        results = search_result.get("results", [])
        if results:
            # Archive the most relevant match
            from services.cleanup import get_cleanup_service
            cleanup = get_cleanup_service(db, embeddings)
            top_result = results[0]
            await cleanup._archive_memory(
                top_result["id"],
                "user_requested",
                f"User asked to forget: {content}"
            )
            result["archived_id"] = top_result["id"]
            result["message"] = f"Archived memory about: {top_result['content'][:100]}..."
        else:
            result["message"] = f"No memories found matching '{content}'"

    elif intent == "list_errors":
        from skills.search import semantic_search
        errors = await semantic_search(
            db=db,
            embeddings=embeddings,
            query="error bug problem exception",
            project_path=project_path,
            memory_type="error",
            limit=5
        )

        results = errors.get("results", [])
        if results:
            result["results"] = results
            result["message"] = f"Found {len(results)} past errors:\n"
            for i, r in enumerate(results[:5], 1):
                status = "Fixed" if r.get("success") else "Unresolved"
                result["message"] += f"\n{i}. [{status}] {r['content'][:100]}..."
        else:
            result["message"] = "No past errors recorded"

    elif intent == "list_decisions":
        from skills.search import semantic_search
        decisions = await semantic_search(
            db=db,
            embeddings=embeddings,
            query="decided chose selected approach",
            project_path=project_path,
            memory_type="decision",
            limit=5
        )

        results = decisions.get("results", [])
        if results:
            result["results"] = results
            result["message"] = f"Found {len(results)} past decisions:\n"
            for i, r in enumerate(results[:5], 1):
                result["message"] += f"\n{i}. {r['content'][:100]}..."
        else:
            result["message"] = "No past decisions recorded"

    elif intent == "list_patterns":
        from skills.search import search_patterns
        patterns = await search_patterns(
            db=db,
            embeddings=embeddings,
            query="pattern solution approach",
            limit=5
        )

        results = patterns.get("patterns", [])
        if results:
            result["results"] = results
            result["message"] = f"Found {len(results)} patterns:\n"
            for i, p in enumerate(results[:5], 1):
                result["message"] += f"\n{i}. **{p['name']}**: {p['solution'][:80]}..."
        else:
            result["message"] = "No patterns recorded"

    elif intent == "stats":
        stats = await db.get_stats()
        result["stats"] = stats
        result["message"] = (
            f"Memory Stats:\n"
            f"- Total memories: {stats.get('total_memories', 0)}\n"
            f"- Patterns: {stats.get('total_patterns', 0)}\n"
            f"- Projects: {stats.get('total_projects', 0)}"
        )

    elif intent == "project_context":
        from skills.search import get_project_context
        context = await get_project_context(
            db=db,
            embeddings=embeddings,
            project_path=project_path
        )

        if context.get("project"):
            proj = context["project"]
            result["project"] = proj
            result["message"] = (
                f"Project: {proj.get('name', 'Unknown')}\n"
                f"Type: {proj.get('project_type', 'Unknown')}\n"
                f"Tech Stack: {', '.join(proj.get('tech_stack', []))}"
            )
        else:
            result["message"] = "No project info stored for this path"

    return result
