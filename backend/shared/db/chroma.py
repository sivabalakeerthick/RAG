"""
ChromaDB PersistentClient — data stored on disk, shared via mounted volume.

Both document-service and rag-service mount the same Docker volume at
CHROMA_PERSIST_PATH, so they read/write the same persistent vector store.
No separate ChromaDB container is required.
"""
import chromadb
from functools import lru_cache

from shared.config import settings

COLLECTION_NAME = "cognidoc_chunks"


@lru_cache
def get_chroma_client() -> chromadb.PersistentClient:
    """
    Returns a singleton PersistentClient that stores data at CHROMA_PERSIST_PATH.
    Data survives container restarts because the path is a Docker volume mount.
    """
    return chromadb.PersistentClient(path=settings.CHROMA_PERSIST_PATH)


def get_collection() -> chromadb.Collection:
    """Get or create the shared chunks collection (cosine similarity space)."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
