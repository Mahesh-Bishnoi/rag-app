"""Prompt templates: rendered output is byte-identical to the inline contract."""

import pytest

from rag_lab.adapters.generation import grounded_prompt, prompt_to_messages
from rag_lab.prompts import load_prompt


def test_grounded_prompt_with_context_renders_exactly() -> None:
    assert grounded_prompt("why?", ["first passage", "second passage"]) == (
        "Answer the question using only the passages below. "
        "Cite the passages you use with their [n] markers.\n\n"
        "Passages:\n[1] first passage\n\n[2] second passage\n\nQuestion: why?\nAnswer:"
    )


def test_grounded_prompt_no_context_renders_exactly() -> None:
    assert grounded_prompt("why?", []) == (
        "Answer the question directly. No knowledge-base passages "
        "were retrieved.\n\nQuestion: why?\nAnswer:"
    )


def test_system_envelope_uses_template() -> None:
    assert prompt_to_messages("q") == [
        {"role": "system", "content": "You answer questions with citations."},
        {"role": "user", "content": "q"},
    ]


def test_braces_in_query_survive_rendering() -> None:
    assert "what is {weird}?" in grounded_prompt("what is {weird}?", [])
    assert "{passages}" not in grounded_prompt("q", ["a {b} passage"])


def test_unknown_template_raises_loudly() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("nope.prompt")
