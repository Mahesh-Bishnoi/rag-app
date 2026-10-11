"""Blob storage behind a port. v1 adapter is Azurite (Azure Blob REST)."""

import logging
from typing import Protocol

from azure.core.exceptions import AzureError, ResourceExistsError

logger = logging.getLogger(__name__)


class StoragePort(Protocol):
    def put(self, name: str, data: bytes, content_type: str) -> str: ...
    def ping(self) -> bool: ...


class InMemoryStorage:
    """Test fake sharing the StoragePort interface."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def put(self, name: str, data: bytes, content_type: str) -> str:
        self.blobs[name] = data
        return f"memory://{name}"

    def ping(self) -> bool:
        return True


class AzuriteBlobStorage:
    # Fail fast by default: clients are built per request against localhost,
    # where backoff buys nothing and /health must report down immediately.
    def __init__(self, connection_string: str, container: str, retry_total: int = 0) -> None:
        from azure.storage.blob import ContainerClient

        self._client = ContainerClient.from_connection_string(
            connection_string, container_name=container, retry_total=retry_total
        )

    def put(self, name: str, data: bytes, content_type: str) -> str:
        from azure.storage.blob import ContentSettings

        try:
            self._client.create_container()
        except ResourceExistsError:
            logger.debug("blob container already exists")
        self._client.upload_blob(
            name,
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
        return f"{self._client.url}/{name}"

    def ping(self) -> bool:
        try:
            return self._client.exists()
        except AzureError as e:
            logger.debug("blob ping failed: %s", e)
            return False
