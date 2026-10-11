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
        return sum(1 for r in self.results if r.retrieved_hit) / len(self.results)

    @property
    def rerank_top1_rate(self) -> float:
        return sum(1 for r in self.results if r.rerank_top1) / len(self.results)

    @property
    def mean_keyword_coverage(self) -> float:
        return sum(r.keyword_coverage for r in self.results) / len(self.results)

    @property
    def mean_groundedness(self) -> float:
        return sum(r.groundedness for r in self.results) / len(self.results)

    def passes(self) -> bool:
        return (
            self.retrieval_hit_rate >= MIN_RETRIEVAL_HIT_AT_1
            and self.rerank_top1_rate >= MIN_RERANK_TOP1_RATE
            and self.mean_keyword_coverage >= MIN_MEAN_KEYWORD_COVERAGE
            and self.mean_groundedness >= MIN_MEAN_GROUNDEDNESS
        )


def evaluate_item(item: dict[str, object]) -> ItemResult:
    """Run one golden item through search → rerank → metric checks. Pure."""
    item_id = str(item["id"])
    query = str(item["query"])
    relevant_id = f"a-{item_id}"
    distractors = item["distractors"]
    if not isinstance(distractors, list) or len(distractors) != 2:
        raise ValueError(f"golden item {item_id} needs exactly 2 distractors")
    chunks = [
        Chunk(
            id=relevant_id,
            content=str(item["relevant_content"]),
            title=str(item["relevant_title"]),
            embedding=list(RELEVANT_VECTOR),
        )
    ]
    for index, distractor in enumerate(distractors):
        if not isinstance(distractor, dict):
            raise ValueError(f"golden item {item_id} distractor {index} must be an object")
        chunks.append(
            Chunk(
                id=f"{'m' if index == 0 else 'z'}-{item_id}",
                content=str(distractor["content"]),
                title=str(distractor["title"]),
                embedding=list(DISTRACTOR_VECTORS[index]),
            )
        )
    retriever = InMemoryRetrieval(chunks)
    results = retriever.search(list(RELEVANT_VECTOR), query, 3)
    ranked_ids = [result.chunk.id or "" for result in results]
    reranked = rerank_candidates(query, results, 1)
    top1 = reranked[0].chunk.id or "" if reranked else ""
    answer = str(item["answer"])
    raw_keywords = item["expected_keywords"]
    if not isinstance(raw_keywords, list):
        raise ValueError(f"golden item {item_id} needs an expected_keywords list")
    keywords = [str(keyword) for keyword in raw_keywords]
    return ItemResult(
        item_id=item_id,
        retrieved_hit=hit_at_k(ranked_ids, relevant_id, 1),
        reciprocal_rank=reciprocal_rank(ranked_ids, relevant_id),
        rerank_top1=top1 == relevant_id,
        keyword_coverage=keyword_coverage(answer, keywords),
        groundedness=groundedness(answer, [str(item["relevant_content"])]),
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


def main() -> int:
    summary = run_suite()
    print(f"eval: {summary.size} items from eval/golden.jsonl")
    for result in summary.results:
        print(
            f"  {result.item_id}: retrieval_hit={result.retrieved_hit} "
            f"rr={result.reciprocal_rank:.2f} rerank_top1={result.rerank_top1} "
            f"keywords={result.keyword_coverage:.2f} grounded={result.groundedness:.2f}"
        )
    print(
        f"summary: hit@1={summary.retrieval_hit_rate:.2f} "
        f"rerank_top1={summary.rerank_top1_rate:.2f} "
        f"keywords={summary.mean_keyword_coverage:.2f} "
        f"grounded={summary.mean_groundedness:.2f} "
        f"-> {'PASS' if summary.passes() else 'FAIL'}"
    )
    return 0 if summary.passes() else 1


if __name__ == "__main__":
    raise SystemExit(main())
