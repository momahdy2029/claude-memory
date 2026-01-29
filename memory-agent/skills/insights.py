"""Skills for cross-session learning and insight management.

These skills allow Claude to:
- Run aggregation to detect patterns
- Retrieve insights for current context
- Get CLAUDE.md improvement suggestions
- Record feedback on insight usefulness
"""
from typing import Dict, Any, Optional, List
from services.insights import get_insights_service


async def run_aggregation(
    db,
    embeddings,
    days_back: int = 30
) -> Dict[str, Any]:
    """Run cross-session learning aggregation.

    Analyzes memories across sessions to identify:
    - Recurring error patterns
    - Repeated decision patterns
    - User correction patterns (Claude blind spots)
    - High-value frequently accessed memories

    Args:
        db: Database service
        embeddings: Embeddings service
        days_back: Number of days to analyze

    Returns:
        Summary of generated insights
    """
    insights_service = get_insights_service(db, embeddings)
    results = await insights_service.run_aggregation(days_back)

    return {
        "success": True,
        "message": f"Generated {results['total_insights']} insights",
        "results": results
    }


async def get_insights(
    db,
    embeddings,
    insight_type: Optional[str] = None,
    project_path: Optional[str] = None,
    min_confidence: float = 0.5,
    limit: int = 10
) -> Dict[str, Any]:
    """Retrieve cross-session learning insights.

    Args:
        db: Database service
        embeddings: Embeddings service
        insight_type: Filter by type (recurring_error, decision_pattern,
                      correction_pattern, high_value_memory)
        project_path: Filter by project
        min_confidence: Minimum confidence threshold (0-1)
        limit: Maximum results

    Returns:
        List of relevant insights
    """
    insights_service = get_insights_service(db, embeddings)
    insights = await insights_service.get_insights(
        insight_type=insight_type,
        project_path=project_path,
        min_confidence=min_confidence,
        limit=limit
    )

    return {
        "success": True,
        "insights": insights,
        "count": len(insights),
        "filters": {
            "insight_type": insight_type,
            "project_path": project_path,
            "min_confidence": min_confidence
        }
    }


async def suggest_improvements(
    db,
    embeddings,
    min_confidence: float = 0.7
) -> Dict[str, Any]:
    """Get suggestions for CLAUDE.md improvements based on insights.

    Analyzes high-confidence insights that haven't been applied yet
    and generates actionable instructions to add to CLAUDE.md.

    Args:
        db: Database service
        embeddings: Embeddings service
        min_confidence: Minimum confidence for suggestions

    Returns:
        List of suggested CLAUDE.md updates
    """
    insights_service = get_insights_service(db, embeddings)
    suggestions = await insights_service.suggest_claude_md_updates(min_confidence)

    return {
        "success": True,
        "suggestions": suggestions,
        "count": len(suggestions),
        "message": f"Found {len(suggestions)} potential CLAUDE.md improvements"
    }


async def record_insight_feedback(
    db,
    embeddings,
    insight_id: int,
    helpful: bool,
    session_id: Optional[str] = None,
    comment: Optional[str] = None
) -> Dict[str, Any]:
    """Record feedback on whether an insight was helpful.

    This helps improve the accuracy of future insights by
    tracking which patterns are actually useful.

    Args:
        db: Database service
        embeddings: Embeddings service
        insight_id: The insight ID
        helpful: Whether the insight was helpful
        session_id: Current session
        comment: Optional feedback comment

    Returns:
        Confirmation of recorded feedback
    """
    insights_service = get_insights_service(db, embeddings)
    success = await insights_service.record_feedback(
        insight_id=insight_id,
        helpful=helpful,
        session_id=session_id,
        comment=comment
    )

    return {
        "success": success,
        "message": "Feedback recorded" if success else "Failed to record feedback",
        "insight_id": insight_id,
        "helpful": helpful
    }


async def mark_insight_applied(
    db,
    embeddings,
    insight_id: int
) -> Dict[str, Any]:
    """Mark an insight as applied to CLAUDE.md.

    Call this after adding an insight's suggestion to CLAUDE.md
    to prevent it from being suggested again.

    Args:
        db: Database service
        embeddings: Embeddings service
        insight_id: The insight ID

    Returns:
        Confirmation
    """
    insights_service = get_insights_service(db, embeddings)
    success = await insights_service.mark_applied_to_claude_md(insight_id)

    return {
        "success": success,
        "message": "Insight marked as applied" if success else "Insight not found",
        "insight_id": insight_id
    }


async def get_project_insights(
    db,
    embeddings,
    project_path: str,
    include_global: bool = True,
    limit: int = 10
) -> Dict[str, Any]:
    """Get insights relevant to a specific project.

    Retrieves both project-specific insights and global insights
    that may apply to this project's tech stack.

    Args:
        db: Database service
        embeddings: Embeddings service
        project_path: Project path to get insights for
        include_global: Include insights without a specific project
        limit: Maximum results

    Returns:
        Project-relevant insights
    """
    insights_service = get_insights_service(db, embeddings)

    # Get project-specific insights
    project_insights = await insights_service.get_insights(
        project_path=project_path,
        limit=limit
    )

    result = {
        "success": True,
        "project_path": project_path,
        "project_insights": project_insights,
        "project_count": len(project_insights)
    }

    if include_global:
        # Get global insights (no project_path)
        global_insights = await insights_service.get_insights(
            project_path=None,
            limit=limit // 2
        )
        # Filter to only truly global ones
        global_only = [i for i in global_insights if not i.get("project_path")]
        result["global_insights"] = global_only
        result["global_count"] = len(global_only)

    return result
