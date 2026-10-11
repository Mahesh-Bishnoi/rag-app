"""Heuristic rerank: token overlap + exact-phrase bonus over RRF order.

Pure and deterministic. A single algorithm does not earn a port (clean-code
one-vs-two rule); the RerankPort arrives with the second algorithm
(cross-encoder), already recorded as the upgrade path in ADR-0014.
"""

from rag_lab.adapters.retrieval import tokenize
from rag_lab.core.models import RetrievalResult

_OVERLAP_WEIGHT = 1.0
_PHRASE_BONUS = 0.3
_RRF_PRIOR_WEIGHT = 0.1


def rerank_candidates(
    query: str, candidates: list[RetrievalResult], limit: int
) -> list[RetrievalResult]:
    """Reorder fused candidates best-first, truncated to limit. No IO."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    query_tokens = set(tokenize(query))
    phrase = query.strip().lower()

    def score(position: int, result: RetrievalResult) -> float:
        content_tokens = set(tokenize(result.chunk.content))
        overlap = len(query_tokens & content_tokens) / len(query_tokens) if query_tokens else 0.0
        exact = _PHRASE_BONUS if phrase and phrase in result.chunk.content.lower() else 0.0
        prior = 1.0 / (position + 1)
        return overlap * _OVERLAP_WEIGHT + exact + prior * _RRF_PRIOR_WEIGHT

    ordered = sorted(enumerate(candidates), key=lambda pair: (-score(*pair), pair[0]))
    return [result for _, result in ordered[:limit]]
