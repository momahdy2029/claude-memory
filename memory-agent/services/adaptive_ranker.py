"""Adaptive Retrieval Ranking - CLaRa-inspired differentiable top-K.

Replaces the static (similarity * 0.7) + (confidence * 0.3) ranking with
multi-signal scoring and temperature-controlled selection.

Features:
- 6 weighted signals (semantic similarity, confidence, recency, access frequency,
  outcome success, context match)
- Query-type detection (error lookup, decision recall, general)
- Temperature control for result diversity
"""
import math
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from config import config

logger = logging.getLogger(__name__)


# Weight profiles for different query types
WEIGHT_PROFILES = {
    'default': {
        'semantic_similarity': 0.35,
        'confidence': 0.15,
        'recency': 0.15,
        'access_frequency': 0.10,
        'outcome_success': 0.15,
        'context_match': 0.10,
    },
    'error_lookup': {
        'semantic_similarity': 0.30,
        'confidence': 0.10,
        'recency': 0.20,
        'access_frequency': 0.05,
        'outcome_success': 0.30,
        'context_match': 0.05,
    },
    'decision_recall': {
        'semantic_similarity': 0.40,
        'confidence': 0.25,
        'recency': 0.10,
        'access_frequency': 0.05,
        'outcome_success': 0.05,
        'context_match': 0.15,
    },
}

# Keywords for query type detection
QUERY_TYPE_KEYWORDS = {
    'error_lookup': [
        'error', 'bug', 'fix', 'crash', 'exception', 'fail', 'broken',
        'issue', 'problem', 'traceback', 'stack trace', 'debug', 'resolve'
    ],
    'decision_recall': [
        'decision', 'architecture', 'chose', 'why', 'approach', 'pattern',
        'convention', 'design', 'strategy', 'trade-off', 'tradeoff'
    ],
}


class AdaptiveRanker:
    """Multi-signal ranking with temperature-controlled result selection.

    Computes a composite score from multiple signals, weighted by query type,
    then applies temperature scaling for diversity control.
    """

    def __init__(self, temperature: Optional[float] = None):
        self.temperature = temperature or config.DEFAULT_SEARCH_TEMPERATURE

    def detect_query_type(self, query_text: str) -> str:
        """Auto-detect query type from keywords.

        Args:
            query_text: The search query

        Returns:
            Query type string: 'error_lookup', 'decision_recall', or 'default'
        """
        if not query_text:
            return 'default'

        query_lower = query_text.lower()

        # Count keyword matches for each type
        scores = {}
        for qtype, keywords in QUERY_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[qtype] = score

        if not scores:
            return 'default'

        # Return type with most keyword matches
        return max(scores, key=scores.get)

    def get_weights(self, query_type: str = 'default') -> Dict[str, float]:
        """Get weight profile for a query type.

        Args:
            query_type: One of 'default', 'error_lookup', 'decision_recall'

        Returns:
            Dict of signal_name -> weight
        """
        return WEIGHT_PROFILES.get(query_type, WEIGHT_PROFILES['default']).copy()

    def compute_signal_scores(self, result: dict) -> Dict[str, float]:
        """Compute individual signal scores for a search result.

        All signals normalized to 0-1 range.

        Args:
            result: Search result dict

        Returns:
            Dict of signal_name -> score (0-1)
        """
        signals = {}

        # 1. Semantic similarity (already 0-1 from FAISS/cosine)
        signals['semantic_similarity'] = result.get('similarity', 0.0)

        # 2. Confidence (already 0-1)
        signals['confidence'] = result.get('confidence', 0.5) or 0.5

        # 3. Recency (exponential decay, half-life = 30 days)
        created_at = result.get('created_at', '')
        signals['recency'] = self._recency_score(created_at)

        # 4. Access frequency (log scale, normalized)
        access_count = result.get('access_count', 0) or 0
        signals['access_frequency'] = min(1.0, math.log(1 + access_count) / math.log(51))

        # 5. Outcome success
        outcome_status = result.get('outcome_status', 'pending')
        success = result.get('success')
        signals['outcome_success'] = self._outcome_score(outcome_status, success)

        # 6. Context match (pre-computed context_adjustment from search)
        context_adj = result.get('context_adjustment', 0.0)
        # Normalize from [-0.2, 0.2] range to [0, 1]
        signals['context_match'] = min(1.0, max(0.0, (context_adj + 0.2) / 0.4))

        return signals

    def _recency_score(self, created_at: str) -> float:
        """Calculate recency score with exponential decay."""
        if not created_at:
            return 0.5

        try:
            created_dt = datetime.fromisoformat(
                created_at.replace('Z', '+00:00')
            ).replace(tzinfo=None)
            age_days = (datetime.now() - created_dt).total_seconds() / 86400.0
            # Half-life of 30 days
            return math.pow(0.5, age_days / 30.0)
        except (ValueError, TypeError, AttributeError):
            return 0.5

    def _outcome_score(self, outcome_status: Optional[str], success: Optional[bool]) -> float:
        """Calculate outcome-based score."""
        if outcome_status == 'success':
            return 1.0
        if success:
            return 0.9
        if outcome_status == 'partial':
            return 0.6
        if outcome_status == 'failed':
            return 0.1
        if outcome_status == 'pending':
            return 0.5
        return 0.5

    def compute_composite_score(
        self,
        result: dict,
        query_type: str = 'default',
        weights_override: Optional[Dict[str, float]] = None
    ) -> float:
        """Compute weighted composite score for a result.

        Args:
            result: Search result dict
            query_type: Query type for weight selection
            weights_override: Optional custom weights

        Returns:
            Composite score (0-1 range before temperature scaling)
        """
        signals = self.compute_signal_scores(result)
        weights = weights_override or self.get_weights(query_type)

        score = sum(
            signals.get(signal, 0.0) * weight
            for signal, weight in weights.items()
        )

        # Apply decay multiplier if present
        decay = result.get('_decay_multiplier', 1.0)
        score *= decay

        # Apply outcome boost
        outcome_boost = result.get('outcome_boost', 1.0)
        score *= outcome_boost

        return score

    def apply_temperature(self, score: float) -> float:
        """Apply temperature scaling to a score.

        - temperature=0.3: sharp ranking (best match dominates)
        - temperature=1.0: balanced
        - temperature=2.0: diverse (flatter distribution)

        Formula: score^(1/temperature)

        Args:
            score: Raw composite score

        Returns:
            Temperature-adjusted score
        """
        if score <= 0:
            return 0.0
        if self.temperature <= 0:
            return score

        return math.pow(score, 1.0 / self.temperature)

    def rank_results(
        self,
        results: List[dict],
        query_text: str = '',
        query_type: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> List[dict]:
        """Rank search results using multi-signal adaptive scoring.

        Args:
            results: List of search result dicts
            query_text: Original query (for type detection)
            query_type: Override query type (skips detection)
            temperature: Override temperature

        Returns:
            Results sorted by adaptive score (descending), with scores attached
        """
        if not results:
            return results

        # Detect or use provided query type
        detected_type = query_type or self.detect_query_type(query_text)
        temp = temperature if temperature is not None else self.temperature

        # Score and sort
        for result in results:
            raw_score = self.compute_composite_score(result, detected_type)
            result['_adaptive_score'] = raw_score
            result['_adaptive_score_tempered'] = self.apply_temperature(raw_score) if temp != 1.0 else raw_score
            result['_query_type'] = detected_type

        results.sort(
            key=lambda x: x.get('_adaptive_score_tempered', 0),
            reverse=True
        )

        return results
