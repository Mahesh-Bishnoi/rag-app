"""Blob storage behind a port. v1 adapter is Azurite (Azure Blob REST)."""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from azure.core.exceptions import AzureError, ResourceExistsError, ResourceNotFoundError

logger = logging.getLogger(__name__)

UPLOAD_URL_TTL_SECONDS = 900


@dataclass(frozen=True)
class BlobStat:
    size: int
    content_type: str


@dataclass
class _StoredBlob:
    data: bytes
    content_type: str


class StoragePort(Protocol):
    def put(self, name: str, data: bytes, content_type: str) -> str: ...
    def ping(self) -> bool: ...
    def blob_url(self, name: str) -> str: ...
    def issue_upload_url(
        self, name: str, content_type: str, expires_seconds: int = UPLOAD_URL_TTL_SECONDS
    ) -> tuple[str, int]: ...
    def stat(self, name: str) -> BlobStat | None: ...
    def download(self, name: str) -> bytes | None: ...


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


def _parse_connection_string(connection_string: str) -> tuple[str, str]:
    parts = dict(part.split("=", 1) for part in connection_string.split(";") if "=" in part)
    return (parts.get("AccountName", ""), parts.get("AccountKey", ""))


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
