"""Golden-set eval runner (ADR-0015). Hermetic: fakes only, no network.

Per item the harness seeds an InMemoryRetrieval with the relevant passage
plus two distractors and queries with the relevant embedding — a simulated
perfect embedder. That isolates the fusion + rerank seam under test; it
says nothing about real embedder quality (the live tier covers real paths).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from rag_lab.adapters.rerank import rerank_candidates
from rag_lab.adapters.retrieval import InMemoryRetrieval
from rag_lab.core.models import Chunk
from rag_lab.eval.metrics import groundedness, hit_at_k, keyword_coverage, reciprocal_rank

GOLDEN = Path(__file__).resolve().parents[3] / "eval" / "golden.jsonl"
RELEVANT_VECTOR = [1.0, 0.0, 0.0]
DISTRACTOR_VECTORS = ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])

# Strictest gates that stay green (observed 2026-10-11: all 1.0 except
# mean groundedness 0.90 — threshold pinned just below with reason).
MIN_RETRIEVAL_HIT_AT_1 = 1.0
MIN_RERANK_TOP1_RATE = 1.0
MIN_MEAN_KEYWORD_COVERAGE = 1.0
MIN_MEAN_GROUNDEDNESS = 0.8


@dataclass(frozen=True)
class EvalThresholds:
    """Injectable pass gates; defaults mirror the MIN_* constants above."""

    retrieval_hit_at_1: float = MIN_RETRIEVAL_HIT_AT_1
    rerank_top1_rate: float = MIN_RERANK_TOP1_RATE
    mean_keyword_coverage: float = MIN_MEAN_KEYWORD_COVERAGE
    mean_groundedness: float = MIN_MEAN_GROUNDEDNESS


@dataclass
class ItemResult:
    item_id: str
    retrieved_hit: bool
    reciprocal_rank: float
    rerank_top1: bool
    keyword_coverage: float
    groundedness: float


@dataclass
class SuiteSummary:
    results: list[ItemResult] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.results)

    @property
    def retrieval_hit_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.retrieved_hit) / len(self.results)

    @property
    def rerank_top1_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.rerank_top1) / len(self.results)

    @property
    def mean_keyword_coverage(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.keyword_coverage for r in self.results) / len(self.results)

    @property
    def mean_groundedness(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.groundedness for r in self.results) / len(self.results)

    def passes(self, thresholds: EvalThresholds | None = None) -> bool:
        gates = thresholds or EvalThresholds()
        return (
            self.retrieval_hit_rate >= gates.retrieval_hit_at_1
            and self.rerank_top1_rate >= gates.rerank_top1_rate
            and self.mean_keyword_coverage >= gates.mean_keyword_coverage
            and self.mean_groundedness >= gates.mean_groundedness
        )


def _require_text(item: dict[str, object], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str):
        raise ValueError("golden item needs string id/query/relevant_content/relevant_title/answer")
    return value


def _parse_eval_item(
    item: dict[str, object],
) -> tuple[str, str, str, str, str, list[dict[str, object]], list[str]]:
    """Validate the golden dict once at the boundary; inner code trusts the types."""
    item_id = _require_text(item, "id")
    query = _require_text(item, "query")
    relevant_content = _require_text(item, "relevant_content")
    relevant_title = _require_text(item, "relevant_title")
    answer = _require_text(item, "answer")
    distractors = item.get("distractors")
    if not isinstance(distractors, list) or len(distractors) != 2:
        raise ValueError(f"golden item {item_id} needs exactly 2 distractors")
    for index, distractor in enumerate(distractors):
        if not isinstance(distractor, dict):
            raise ValueError(f"golden item {item_id} distractor {index} must be an object")
        if not isinstance(distractor.get("content"), str) or not isinstance(
            distractor.get("title"), str
        ):
            raise ValueError(f"golden item {item_id} distractor {index} needs string content/title")
    raw_keywords = item.get("expected_keywords")
    if not isinstance(raw_keywords, list):
        raise ValueError(f"golden item {item_id} needs an expected_keywords list")
    typed_distractors = cast("list[dict[str, object]]", distractors)
    return (
        item_id,
        query,
        relevant_content,
        relevant_title,
        answer,
        typed_distractors,
        [str(keyword) for keyword in raw_keywords],
    )


def _build_eval_chunks(
    item_id: str, relevant_content: str, relevant_title: str, distractors: list[dict[str, object]]
) -> tuple[list[Chunk], str]:
    """Seed chunks: relevant passage plus two distractors with fixed vectors."""
    relevant_id = f"a-{item_id}"
    chunks = [
        Chunk(
            id=relevant_id,
            content=relevant_content,
            title=relevant_title,
            embedding=list(RELEVANT_VECTOR),
        )
    ]
    for index, distractor in enumerate(distractors):
        chunks.append(
            Chunk(
                id=f"{'m' if index == 0 else 'z'}-{item_id}",
                content=str(distractor["content"]),
                title=str(distractor["title"]),
                embedding=list(DISTRACTOR_VECTORS[index]),
            )
        )
    return chunks, relevant_id


def evaluate_item(item: dict[str, object]) -> ItemResult:
    """Run one golden item through search → rerank → metric checks.

    Deterministic and hermetic (in-memory fake only), not pure: it builds
    a seeded InMemoryRetrieval per item.
    """
    item_id, query, relevant_content, relevant_title, answer, distractors, keywords = (
        _parse_eval_item(item)
    )
    chunks, relevant_id = _build_eval_chunks(item_id, relevant_content, relevant_title, distractors)
    retriever = InMemoryRetrieval(chunks)
    results = retriever.search(list(RELEVANT_VECTOR), query, 3)
    ranked_ids = [result.chunk.id or "" for result in results]
    reranked = rerank_candidates(query, results, 1)
    top1 = reranked[0].chunk.id or "" if reranked else ""
    return ItemResult(
        item_id=item_id,
        retrieved_hit=hit_at_k(ranked_ids, relevant_id, 1),
        reciprocal_rank=reciprocal_rank(ranked_ids, relevant_id),
        rerank_top1=top1 == relevant_id,
        keyword_coverage=keyword_coverage(answer, keywords),
        groundedness=groundedness(answer, [relevant_content]),
    )


def run_suite(golden_path: Path = GOLDEN) -> SuiteSummary:
    """Load the golden file and evaluate every item. No IO beyond the file."""
    items = [
        json.loads(line)
        for line in golden_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items:
        raise ValueError(f"golden set is empty: {golden_path}")
    return SuiteSummary(results=[evaluate_item(item) for item in items])
