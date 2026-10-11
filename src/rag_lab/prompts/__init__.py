"""Grounded-answer prompt templates as plain .prompt files.

Placeholders are literal {query} / {passages} tokens filled by the caller
with str.replace — no template logic, no conditional syntax. Numbering of
passages ([1..n]) stays in Python so templates never branch.
"""

from functools import cache
from importlib import resources


@cache
def load_prompt(name: str) -> str:
    """Read a template by filename. Raises FileNotFoundError for unknown names."""
    return resources.files("rag_lab.prompts").joinpath(name).read_text(encoding="utf-8")
