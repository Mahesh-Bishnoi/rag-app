"""Blob CORS for browser uploads: rule shape, emulator-only guard, live round-trip."""

import os

import pytest

from rag_lab.adapters.storage import AzuriteBlobStorage, cors_rule_matches, desired_cors_rule
from rag_lab.core.config import Settings

LIVE = os.environ.get("RAGLAB_LIVE_DB") == "1"
needs_live = pytest.mark.skipif(not LIVE, reason="needs live services (RAGLAB_LIVE_DB=1)")

ORIGINS = ["http://localhost:8000", "http://127.0.0.1:8000"]

# Never an emulator account, so ensure_cors must return before any IO.
PROD_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=https;AccountName=prod;"
    "AccountKey=aGk=;"
    "BlobEndpoint=https://prod.blob.core.windows.net;"
)


def test_desired_cors_rule_covers_browser_put() -> None:
    rule = desired_cors_rule(ORIGINS)
    assert sorted(str(rule.allowed_origins).split(",")) == sorted(ORIGINS)
    assert "PUT" in str(rule.allowed_methods).split(",")
    assert cors_rule_matches(rule, ORIGINS)


def test_cors_rule_mismatch_on_different_origins() -> None:
    assert not cors_rule_matches(desired_cors_rule(ORIGINS), ["http://localhost:9000"])


def test_ensure_cors_skips_non_emulator_without_io() -> None:
    store = AzuriteBlobStorage(PROD_CONNECTION_STRING, "kb-prod")
    assert store.ensure_cors(ORIGINS) is False


@needs_live
def test_ensure_cors_round_trip_is_idempotent() -> None:
    settings = Settings()
    assert settings.azurite_connection_string, "AZURITE_CONNECTION_STRING not configured"
    store = AzuriteBlobStorage(settings.azurite_connection_string, "kb-test-cors")
    assert store.ensure_cors(ORIGINS) is True
    assert store.ensure_cors(ORIGINS) is True
