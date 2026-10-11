"""Eval metrics: pure-function specs plus the golden-suite contract."""

from pathlib import Path

import pytest

from rag_lab.eval.cli import main
from rag_lab.eval.metrics import (
    groundedness,
    hit_at_k,
    keyword_coverage,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)
from rag_lab.eval.run import GOLDEN, run_suite

GOLDEN_SIZE = 12


def test_hit_at_k_finds_top_k_only() -> None:
    assert hit_at_k(["a", "b"], "a", 1)
    assert not hit_at_k(["a", "b"], "b", 1)
    assert hit_at_k(["a", "b"], "b", 2)


def test_hit_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        hit_at_k(["a"], "a", 0)


def test_recall_at_k_counts_expected_share() -> None:
    assert recall_at_k(["a", "c"], {"a", "b"}, 2) == pytest.approx(0.5)
    assert recall_at_k(["a"], set(), 1) == 1.0
    with pytest.raises(ValueError, match="k must be positive"):
        recall_at_k(["a"], {"a"}, 0)


def test_reciprocal_rank_scores_position_or_zero() -> None:
    assert reciprocal_rank(["a", "b"], "b") == pytest.approx(0.5)
    assert reciprocal_rank(["a"], "missing") == 0.0


def test_mean_reciprocal_rank_averages_aligned_lists() -> None:
    assert mean_reciprocal_rank([["a"], ["b", "a"]], ["a", "a"]) == pytest.approx(0.75)
    assert mean_reciprocal_rank([], []) == 1.0
    with pytest.raises(ValueError, match="must align"):
        mean_reciprocal_rank([["a"]], ["a", "b"])


def test_keyword_coverage_is_case_insensitive() -> None:
    assert keyword_coverage("Use HNSW here", ["hnsw", "missing"]) == pytest.approx(0.5)
    assert keyword_coverage("anything", []) == 1.0


def test_groundedness_ignores_glue_words() -> None:
    assert groundedness("the cat sits", ["the cat sits on the mat"]) == 1.0
    assert groundedness("quantum entanglement", ["weather report"]) == 0.0
    assert groundedness("a an the", ["nothing shared"]) == 1.0


def test_golden_suite_passes_thresholds() -> None:
    assert GOLDEN.exists()
    summary = run_suite(GOLDEN)
    assert summary.size == GOLDEN_SIZE
    assert summary.passes()
    assert summary.retrieval_hit_rate == pytest.approx(1.0)
    assert summary.rerank_top1_rate == pytest.approx(1.0)


def test_suite_rejects_empty_golden_file(tmp_path: Path) -> None:
    empty = tmp_path / "golden.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="golden set is empty"):
        run_suite(empty)


def test_run_main_reports_pass(capsys: pytest.CaptureFixture[str]) -> None:
    assert main() == 0
    out = capsys.readouterr().out
    assert "12 items" in out
    assert "PASS" in out
