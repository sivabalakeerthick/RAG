"""
Hybrid retrieval using LangChain:
  - Dense: LangChain Chroma vector store retriever
  - Sparse: LangChain BM25Retriever over PostgreSQL chunks
  - Fusion: Reciprocal Rank Fusion (RRF) combining LangChain Documents
"""
import asyncio
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LCDocument
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import settings
from shared.db.chroma import COLLECTION_NAME, get_chroma_client
from shared.db.postgres import Chunk
from services.document_service.core.embedder import get_embeddings_model


def _get_dense_retriever():
    """Build LangChain Chroma retriever."""
    store = Chroma(
        client=get_chroma_client(),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings_model(),
    )
    return store.as_retriever(search_kwargs={"k": settings.DENSE_TOP_K})


async def _get_sparse_retriever(db: AsyncSession) -> BM25Retriever | None:
    """Build LangChain BM25Retriever from PostgreSQL chunks."""
    result = await db.execute(select(Chunk).order_by(Chunk.document_id, Chunk.chunk_index))
    all_chunks = result.scalars().all()

    if not all_chunks:
        return None

    docs = [
        LCDocument(
            page_content=c.text,
            metadata={
                "document_id": str(c.document_id),
                "chunk_index": c.chunk_index,
                "chroma_id": c.chroma_id,
            },
        )
        for c in all_chunks
    ]
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = settings.SPARSE_TOP_K
    return retriever


def _reciprocal_rank_fusion(
    dense_docs: list[LCDocument],
    sparse_docs: list[LCDocument],
    k: int = 60,
) -> list[dict]:
    """
    Merge dense and sparse LangChain documents using Reciprocal Rank Fusion.
    RRF score = Σ 1 / (k + rank + 1)
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, doc in enumerate(dense_docs):
        doc_id = doc.metadata.get("document_id", "")
        chunk_idx = doc.metadata.get("chunk_index", rank)
        filename = doc.metadata.get("filename", "Unknown")
        # ChromaDB stores filename as part of the uploaded metadata
        if not filename or filename == "Unknown":
            filename = doc.metadata.get("filename", "Unknown Document")
        key = f"{doc_id}:{chunk_idx}"

        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = {
            "text": doc.page_content,
            "document_id": doc_id,
            "chunk_index": chunk_idx,
            "filename": filename,
        }

    for rank, doc in enumerate(sparse_docs):
        doc_id = doc.metadata.get("document_id", "")
        chunk_idx = doc.metadata.get("chunk_index", rank)
        filename = doc.metadata.get("filename", "Unknown")
        key = f"{doc_id}:{chunk_idx}"

        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in doc_map:
            doc_map[key] = {
                "text": doc.page_content,
                "document_id": doc_id,
                "chunk_index": chunk_idx,
                "filename": filename,
            }

    sorted_keys = sorted(scores, key=lambda k_: scores[k_], reverse=True)
    results = []
    for key in sorted_keys[: settings.RERANK_TOP_K]:
        item = doc_map[key].copy()
        item["rrf_score"] = scores[key]
        results.append(item)

    return results


async def hybrid_retrieve(query: str, db: AsyncSession) -> list[dict]:
    """
    Execute dense + sparse retrieval concurrently and fuse results with RRF.
    """
    dense_retriever = _get_dense_retriever()
    sparse_retriever = await _get_sparse_retriever(db)

    dense_task = dense_retriever.ainvoke(query)
    if sparse_retriever:
        sparse_task = sparse_retriever.ainvoke(query)
        dense_docs, sparse_docs = await asyncio.gather(dense_task, sparse_task)
    else:
        dense_docs = await dense_task
        sparse_docs = []

    return _reciprocal_rank_fusion(dense_docs, sparse_docs)
