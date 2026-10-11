"""Pure retrieval math: limits, cosine, BM25, reciprocal-rank fusion. No IO."""

import math
from collections import Counter

RRF_K = 60


def validate_limit(limit: int) -> None:
    """Shared guard: search and rerank limits must be positive."""
    if limit <= 0:
        raise ValueError("limit must be positive")


def cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity in [0, 1] for non-negative embeddings. Loud on dim drift."""
    if len(left) != len(right):
        raise ValueError(f"vector dim mismatch {len(left)} != {len(right)}")
    if not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def bm25_scores(query: list[str], docs: list[list[str]]) -> list[float]:
    """Plain BM25 (k1=1.2, b=0.75) for the in-memory fake. Pure."""
    if not docs:
        return []
    doc_count = len(docs)
    doc_freq = Counter(term for doc in docs for term in set(doc))
    avg_len = sum(len(doc) for doc in docs) / doc_count
    scores: list[float] = []
    for doc in docs:
        term_freq = Counter(doc)
        doc_len = len(doc)
        total = 0.0
        for term in query:
            count = term_freq.get(term, 0)
            if count == 0:
                continue
            idf = math.log(1.0 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            norm = 1.2 * (1.0 - 0.75 + 0.75 * (doc_len / avg_len if avg_len else 1.0))
            total += idf * (count * 2.2) / (count + norm)
        scores.append(total)
    return scores


def rrf_fuse(ranked: list[list[str]], limit: int) -> list[tuple[str, int]]:
    """Fuse ranked id lists with reciprocal-rank fusion.

    Returns (chunk_id, 1-based fused rank) best-first; ties break on id so
    fusion is deterministic.
    """
    validate_limit(limit)
    fused: dict[str, float] = {}
    for ids in ranked:
        for rank, chunk_id in enumerate(ids, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [(chunk_id, rank) for rank, (chunk_id, _) in enumerate(ordered[:limit], start=1)]
