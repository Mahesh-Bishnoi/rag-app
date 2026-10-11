"""Blob storage behind a port. v1 adapter is Azurite (Azure Blob REST)."""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from azure.core.exceptions import AzureError, ResourceExistsError, ResourceNotFoundError

if TYPE_CHECKING:
    from azure.storage.blob import CorsRule

logger = logging.getLogger(__name__)

UPLOAD_URL_TTL_SECONDS = 900
# Only this account ever gets CORS rules written: it is the well-known local
# emulator credential, never a real subscription. Real Azure keeps
# operator-owned CORS; the app must not overwrite it.
DEV_EMULATOR_ACCOUNT = "devstoreaccount1"
# Browser uploads PUT blobs directly, so the rule must cover the PUT method
# plus reads the UI may do while debugging. "*" headers covers Content-Type
# and the x-ms-blob-type the UI sends (ADR-0017).
BLOB_CORS_METHODS = ["PUT", "GET", "HEAD", "OPTIONS"]
BLOB_CORS_MAX_AGE_SECONDS = 3600


def blob_name(blob_url: str) -> str:
    """Object key from a stored canonical blob URL (never a SAS URL)."""
    return blob_url.rsplit("/", 1)[-1].split("?", 1)[0]


@dataclass(frozen=True)
class BlobStat:
    size: int
    content_type: str


@dataclass
class _StoredBlob:
    data: bytes
    content_type: str


@runtime_checkable
class StoragePort(Protocol):
    def put(self, name: str, data: bytes, content_type: str) -> str: ...
    def ping(self) -> bool: ...
    def blob_url(self, name: str) -> str: ...
    def issue_upload_url(
        self, name: str, content_type: str, expires_seconds: int = UPLOAD_URL_TTL_SECONDS
    ) -> tuple[str, int]: ...
    def stat(self, name: str) -> BlobStat | None: ...
    def download(self, name: str) -> bytes | None: ...
    def ensure_cors(self, origins: list[str]) -> bool: ...


class InMemoryStorage:
    """Test fake sharing the StoragePort interface."""

    def __init__(self) -> None:
        self.blobs: dict[str, _StoredBlob] = {}

    def put(self, name: str, data: bytes, content_type: str) -> str:
        self.blobs[name] = _StoredBlob(data=data, content_type=content_type)
        return self.blob_url(name)

    def ping(self) -> bool:
        return True

    def blob_url(self, name: str) -> str:
        return f"memory://{name}"

    def issue_upload_url(
        self, name: str, content_type: str, expires_seconds: int = UPLOAD_URL_TTL_SECONDS
    ) -> tuple[str, int]:
        del content_type
        token = uuid.uuid4().hex
        return (f"memory://{name}?token={token}&expires_in={expires_seconds}", expires_seconds)

    def stat(self, name: str) -> BlobStat | None:
        stored = self.blobs.get(name)
        if stored is None:
            return None
        return BlobStat(size=len(stored.data), content_type=stored.content_type)

    def download(self, name: str) -> bytes | None:
        stored = self.blobs.get(name)
        return stored.data if stored is not None else None

    def ensure_cors(self, origins: list[str]) -> bool:
        """In-memory fake has no CORS surface; always reports not-applied."""
        del origins
        return False


def _parse_connection_string(connection_string: str) -> tuple[str, str]:
    parts = dict(part.split("=", 1) for part in connection_string.split(";") if "=" in part)
    return (parts.get("AccountName", ""), parts.get("AccountKey", ""))


def desired_cors_rule(origins: list[str]) -> "CorsRule":
    """The one CORS rule browser uploads need. Pure: no IO, safe to test."""
    from azure.storage.blob import CorsRule

    return CorsRule(
        allowed_origins=origins,
        allowed_methods=BLOB_CORS_METHODS,
        allowed_headers=["*"],
        exposed_headers=["*"],
        max_age_in_seconds=BLOB_CORS_MAX_AGE_SECONDS,
    )


def _split_csv_or_list(value: str | list[str]) -> list[str]:
    """SDK quirk: CorsRule fields come back comma-joined; accept both forms."""
    if isinstance(value, str):
        return [part for part in value.split(",") if part]
    return list(value)


def cors_rule_matches(rule: "CorsRule", origins: list[str]) -> bool:
    """True when a stored rule already grants the UI origins what they need."""
    methods = _split_csv_or_list(rule.allowed_methods)
    return (
        sorted(_split_csv_or_list(rule.allowed_origins)) == sorted(origins)
        and all(method in methods for method in BLOB_CORS_METHODS)
        and rule.max_age_in_seconds == BLOB_CORS_MAX_AGE_SECONDS
    )


class AzuriteBlobStorage:
    # Fail fast by default: clients are built per request against localhost,
    # where backoff buys nothing and /health must report down immediately.
    def __init__(self, connection_string: str, container: str, retry_total: int = 0) -> None:
        from azure.storage.blob import ContainerClient

        self._connection_string = connection_string
        self._container = container
        self._client = ContainerClient.from_connection_string(
            connection_string, container_name=container, retry_total=retry_total
        )

    def put(self, name: str, data: bytes, content_type: str) -> str:
        from azure.storage.blob import ContentSettings

        self._ensure_container()
        self._client.upload_blob(
            name,
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
        return self.blob_url(name)

    def ping(self) -> bool:
        try:
            return self._client.exists()
        except AzureError as e:
            logger.debug("blob ping failed: %s", e)
            return False

    def blob_url(self, name: str) -> str:
        return f"{self._client.url}/{name}"

    def issue_upload_url(
        self, name: str, content_type: str, expires_seconds: int = UPLOAD_URL_TTL_SECONDS
    ) -> tuple[str, int]:
        from azure.storage.blob import BlobSasPermissions, generate_blob_sas

        del content_type
        self._ensure_container()
        account_name, account_key = _parse_connection_string(self._connection_string)
        expiry = datetime.now(UTC) + timedelta(seconds=expires_seconds)
        sas = generate_blob_sas(
            account_name=account_name,
            container_name=self._container,
            blob_name=name,
            account_key=account_key,
            permission=BlobSasPermissions(write=True, create=True),
            expiry=expiry,
        )
        blob_client = self._client.get_blob_client(name)
        return (f"{blob_client.url}?{sas}", expires_seconds)

    def stat(self, name: str) -> BlobStat | None:
        try:
            props = self._client.get_blob_client(name).get_blob_properties()
        except ResourceNotFoundError:
            return None
        content_type = props.content_settings.content_type or "application/octet-stream"
        return BlobStat(size=props.size, content_type=content_type)

    def download(self, name: str) -> bytes | None:
        try:
            return self._client.get_blob_client(name).download_blob().readall()
        except ResourceNotFoundError:
            return None

    def _ensure_container(self) -> None:
        try:
            self._client.create_container()
        except ResourceExistsError:
            logger.debug("blob container already exists")

    def ensure_cors(self, origins: list[str]) -> bool:
        """Write the browser-upload CORS rule to the local emulator.

        Returns True when the rule is present afterwards, False when skipped
        (non-emulator account — real Azure keeps operator-owned CORS).
        Raises on emulator IO failures; the caller decides fail-open/closed.
        """
        from azure.storage.blob import BlobServiceClient

        account, _ = _parse_connection_string(self._connection_string)
        if account != DEV_EMULATOR_ACCOUNT:
            logger.debug("skipping blob CORS ensure for non-emulator account")
            return False
        service = BlobServiceClient.from_connection_string(self._connection_string, retry_total=0)
        existing = list(service.get_service_properties().get("cors") or [])
        if any(cors_rule_matches(rule, origins) for rule in existing):
            return True
        service.set_service_properties(cors=[*existing, desired_cors_rule(origins)])
        return True
