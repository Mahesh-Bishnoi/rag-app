"""Eval CLI: console entrypoint over the hermetic golden suite (ADR-0015).

Imperative shell: formatting + exit code live here; harness logic stays in
:mod:`rag_lab.eval.run` so tests import behavior without console IO.
"""

from rag_lab.eval.run import run_suite


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
