"""Worker entrypoint: poll for uploaded documents and ingest them."""

import logging
import time

from rag_lab.adapters.db import get_engine
from rag_lab.adapters.documents import PostgresDocumentStore
from rag_lab.adapters.embeddings import OpenAICompatibleEmbedding
from rag_lab.adapters.storage import AzuriteBlobStorage, InMemoryStorage, StoragePort
from rag_lab.core.config import Settings
from rag_lab.ingest.loaders import LoaderPort, PdfLoader, TextLoader
from rag_lab.worker.pipeline import drain

logger = logging.getLogger(__name__)


def main() -> None:  # pragma: no cover - infinite poll loop, proved live
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    storage: StoragePort
    if settings.azurite_connection_string:
        storage = AzuriteBlobStorage(settings.azurite_connection_string, settings.blob_container)
    else:
        storage = InMemoryStorage()
    doc_store = PostgresDocumentStore(get_engine(settings.database_url))
    embedder = OpenAICompatibleEmbedding(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
    )
    loaders: list[LoaderPort] = [TextLoader(), PdfLoader()]
    logger.info("ingest worker polling every %ss", settings.ingest_poll_seconds)
    while True:
        try:
            processed = drain(doc_store, storage, embedder, loaders)
        except Exception:
            logger.exception("ingest drain failed")
        else:
            if processed:
                logger.info("ingested %d document(s)", processed)
        time.sleep(settings.ingest_poll_seconds)


if __name__ == "__main__":  # pragma: no cover - process entrypoint, not importable logic
    main()
