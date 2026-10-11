"""Heuristic eval metrics (ADR-0015). Pure functions, no IO."""

from rag_lab.adapters.retrieval import tokenize


def hit_at_k(ranked_ids: list[str], expected_id: str, k: int) -> bool:
    """Whether the expected id lands in the top k."""
    if k <= 0:
        raise ValueError("k must be positive")
    return expected_id in ranked_ids[:k]


def recall_at_k(ranked_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Fraction of expected ids retrieved in the top k."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not expected_ids:
        return 1.0
    return len(expected_ids & set(ranked_ids[:k])) / len(expected_ids)


def reciprocal_rank(ranked_ids: list[str], expected_id: str) -> float:
    """1/rank of the expected id, or 0.0 when absent."""
    try:
        return 1.0 / (ranked_ids.index(expected_id) + 1)
    except ValueError:
        return 0.0


def mean_reciprocal_rank(ranked_lists: list[list[str]], expected_ids: list[str]) -> float:
    """Mean reciprocal rank over parallel ranked lists and expectations."""
    if len(ranked_lists) != len(expected_ids):
        raise ValueError("ranked lists and expected ids must align")
    if not ranked_lists:
        return 1.0
    pairs = zip(ranked_lists, expected_ids, strict=True)
    return sum(reciprocal_rank(ranked, expected) for ranked, expected in pairs) / len(ranked_lists)


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer (case-insensitive)."""
    if not expected_keywords:
        return 1.0
    lowered = answer.lower()
    return sum(1 for keyword in expected_keywords if keyword.lower() in lowered) / len(
        expected_keywords
    )


def groundedness(answer: str, cited_contents: list[str]) -> float:
    """Fraction of substantive answer tokens appearing in the cited passages.

    Substantive means alphanumeric tokens of length >= 4; glue words are
    excluded so short answers are not punished for grammar.
    """
    answer_tokens = {token for token in tokenize(answer) if len(token) >= 4}
    if not answer_tokens:
        return 1.0
    cited_tokens: set[str] = set()
    for content in cited_contents:
        cited_tokens.update(token for token in tokenize(content) if len(token) >= 4)
    return len(answer_tokens & cited_tokens) / len(answer_tokens)
