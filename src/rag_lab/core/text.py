"""Shared text helpers. Tokenizer lives here so adapters and eval share one."""

import re

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text_value: str) -> list[str]:
    """Lowercase alphanumeric tokens. Shared by keyword scoring and rerank."""
    return _TOKEN.findall(text_value.lower())
