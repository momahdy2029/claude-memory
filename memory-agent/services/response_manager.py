"""Response size management with progressive degradation.

Ensures MCP tool responses stay within Claude Code's token limits
by applying increasingly aggressive size reduction strategies.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from config import config

logger = logging.getLogger(__name__)

# Fields added by graph enrichment that are safe to strip
GRAPH_ENRICHMENT_FIELDS = frozenset({
    "known_fixes", "rationale", "consequences",
    "contradictions", "causal_chain",
})

# Keys in result dicts that hold lists of memory items
RESULT_LIST_KEYS = (
    "results", "memories", "patterns", "decisions",
    "code_patterns", "relevant_to_query", "matches",
)


def _json_size(data: Any, indent: Optional[int] = 2) -> int:
    """Return the character count of JSON-serialized data."""
    return len(json.dumps(data, indent=indent, default=str))


def _strip_graph_fields(data: Any) -> Any:
    """Recursively remove graph enrichment fields from dicts/lists."""
    if isinstance(data, dict):
        return {
            k: _strip_graph_fields(v)
            for k, v in data.items()
            if k not in GRAPH_ENRICHMENT_FIELDS
        }
    if isinstance(data, list):
        return [_strip_graph_fields(item) for item in data]
    return data


def _truncate_content_fields(data: Any, max_len: int) -> Any:
    """Truncate string values in 'content', 'outcome', 'solution' fields."""
    truncatable = {"content", "outcome", "solution", "description"}
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if k in truncatable and isinstance(v, str) and len(v) > max_len:
                result[k] = v[:max_len] + "..."
            else:
                result[k] = _truncate_content_fields(v, max_len)
        return result
    if isinstance(data, list):
        return [_truncate_content_fields(item, max_len) for item in data]
    return data


def _halve_result_lists(data: Any, min_count: int) -> Any:
    """Reduce list-type result fields to at most half their size (min min_count)."""
    if not isinstance(data, dict):
        return data
    result = {}
    for k, v in data.items():
        if k in RESULT_LIST_KEYS and isinstance(v, list) and len(v) > min_count:
            new_len = max(len(v) // 2, min_count)
            result[k] = v[:new_len]
        else:
            result[k] = _halve_result_lists(v, min_count)
    return result


def fit_response(
    data: Any,
    max_chars: Optional[int] = None,
) -> str:
    """Serialize data to JSON, applying progressive degradation if too large.

    Degradation levels:
        0 - Full response with indent=2
        1 - Compact JSON (no indent)
        2 - Strip graph enrichment fields
        3 - Truncate content fields to CONTENT_TRUNCATE_LENGTH
        4 - Halve result list counts (min MIN_RESULT_COUNT)
        5 - Emergency hard truncation

    Returns:
        JSON string guaranteed to be <= max_chars.
    """
    if max_chars is None:
        max_chars = config.MAX_RESPONSE_CHARS

    level = 0
    working = data

    # Level 0: full pretty-printed JSON
    output = json.dumps(working, indent=2, default=str)
    if len(output) <= max_chars:
        return output

    # Level 1: compact JSON
    level = 1
    output = json.dumps(working, default=str)
    if len(output) <= max_chars:
        return _with_meta(output, working, level, max_chars)

    # Level 2: strip graph enrichment fields
    level = 2
    working = _strip_graph_fields(working)
    output = json.dumps(working, default=str)
    if len(output) <= max_chars:
        return _with_meta(output, working, level, max_chars)

    # Level 3: truncate content fields
    level = 3
    working = _truncate_content_fields(working, config.CONTENT_TRUNCATE_LENGTH)
    output = json.dumps(working, default=str)
    if len(output) <= max_chars:
        return _with_meta(output, working, level, max_chars)

    # Level 4: halve result counts
    level = 4
    working = _halve_result_lists(working, config.MIN_RESULT_COUNT)
    output = json.dumps(working, default=str)
    if len(output) <= max_chars:
        return _with_meta(output, working, level, max_chars)

    # Level 5: emergency hard truncation — return valid JSON
    level = 5
    logger.warning(
        "Response required emergency truncation: %d -> %d chars",
        len(output), max_chars,
    )
    return json.dumps({
        "_response_meta": {
            "degradation_level": level,
            "truncated": True,
            "original_chars": len(output),
            "note": "Response was emergency-truncated. Use specific queries to retrieve full data.",
        }
    })


def _with_meta(
    compact_json: str,
    working_data: Any,
    level: int,
    max_chars: int,
) -> str:
    """Inject _response_meta into the serialized response."""
    if not isinstance(working_data, dict):
        return compact_json

    meta = {
        "degradation_level": level,
        "truncated": False,
        "note": _level_description(level),
    }
    working_data["_response_meta"] = meta
    output = json.dumps(working_data, default=str)

    # If adding meta pushes us over, return without meta
    if len(output) > max_chars:
        del working_data["_response_meta"]
        return compact_json

    return output


def _level_description(level: int) -> str:
    descriptions = {
        1: "Compact JSON (whitespace removed)",
        2: "Graph enrichment fields stripped",
        3: "Content fields truncated",
        4: "Result counts reduced",
        5: "Emergency truncation applied",
    }
    return descriptions.get(level, "Unknown degradation")
