"""
LangChain Chroma vector store indexing helpers for the document service.
Uses langchain_chroma.Chroma with the persistent client.

Every vector carries a `document_id` metadata tag, so all document-scoped
operations (purge / re-tag / count) work off that tag rather than the
chroma_id list tracked in PostgreSQL. That way vectors orphaned by an
interrupted request are still cleaned up and the two stores stay in sync.
"""
from functools import lru_cache
from langchain_chroma import Chroma
from shared.db.chroma import COLLECTION_NAME, get_chroma_client
from services.document_service.core.embedder import get_embeddings_model


@lru_cache
def get_vector_store() -> Chroma:
    """Returns singleton LangChain Chroma vector store instance."""
    return Chroma(
        client=get_chroma_client(),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings_model(),
    )


def add_to_chroma(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """Add documents with precomputed embeddings to Chroma via LangChain store."""
    store = get_vector_store()
    store._collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def delete_from_chroma(ids: list[str]) -> None:
    """Delete documents by ID from Chroma via LangChain store."""
    if not ids:
        return
    store = get_vector_store()
    store.delete(ids=ids)


def count_all_vectors() -> int:
    """Total number of vectors in the collection, across all documents."""
    return get_vector_store()._collection.count()


def get_document_vector_ids(document_id: str) -> list[str]:
    """Every vector ID tagged with this document_id, tracked in PG or not."""
    store = get_vector_store()
    result = store._collection.get(where={"document_id": document_id}, include=[])
    return list(result.get("ids") or [])


def delete_document_vectors(document_id: str) -> int:
    """
    Purge all vectors belonging to a document — including orphans left behind by
    a previously interrupted upload/reindex. Returns the number deleted.
    """
    ids = get_document_vector_ids(document_id)
    if ids:
        get_vector_store()._collection.delete(ids=ids)
    return len(ids)


def count_document_vectors(document_id: str) -> int:
    """Number of vectors currently stored for a document."""
    return len(get_document_vector_ids(document_id))


def update_document_vector_metadata(document_id: str, updates: dict) -> int:
    """
    Merge `updates` into the metadata of every vector of a document, so edits to
    name/category on the document row stay reflected in the vector store.
    """
    store = get_vector_store()
    result = store._collection.get(where={"document_id": document_id}, include=["metadatas"])
    ids = list(result.get("ids") or [])
    if not ids:
        return 0

    metadatas = list(result.get("metadatas") or [])
    merged = [{**(md or {}), **updates} for md in metadatas]
    store._collection.update(ids=ids, metadatas=merged)
    return len(ids)


def get_chunks_from_chroma(ids: list[str]) -> list[dict]:
    """
    Look up the stored vector for each chunk so the inspector drawer can report
    whether the chunk is actually embedded, and at what dimensionality.

    Note: there is no similarity "score" for a stored chunk — a score only
    exists relative to a query — so the drawer reports embedding state instead.
    """
    if not ids:
        return []

    store = get_vector_store()
    result = store._collection.get(ids=ids, include=["embeddings"])
    found_ids = list(result.get("ids") or [])
    embeddings = result.get("embeddings")
    if embeddings is None:
        embeddings = []

    return [
        {"id": id_, "dim": len(emb) if emb is not None else 0}
        for id_, emb in zip(found_ids, embeddings)
    ]
