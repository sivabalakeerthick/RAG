"""
High-Performance Hybrid Retrieval Engine:
  - Stage 1 Hierarchical Document Routing (narrows search space across large libraries)
  - Dense: LangChain Chroma vector store retriever with metadata pre-filtering
  - Sparse: PostgreSQL native GIN-indexed Full-Text Search with BM25 fallback
  - Fusion: Reciprocal Rank Fusion (RRF) combining dense and sparse hits
"""
import asyncio
import uuid
import logging
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LCDocument
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import settings
from shared.db.chroma import (
    COLLECTION_NAME,
    get_chroma_client,
    get_document_collection,
)
from shared.db.postgres import Chunk, Document
from services.document_service.core.embedder import (
    get_embeddings_model,
    embed_query,
)

logger = logging.getLogger(__name__)


async def _hierarchical_stage1_doc_ids(
    query: str,
    category: str | None = None,
    top_docs: int = 3,
) -> list[str] | None:
    """
    Stage 1 Hierarchical Retrieval:
    When the library has multiple documents, search document-level summaries
    first to restrict chunk search strictly to the top relevant documents.
    """
    try:
        doc_col = get_document_collection()
        total_docs = doc_col.count()
        if total_docs <= 3:
            # Small corpus — searching all documents directly is faster and covers everything
            return None

        query_vector = await embed_query(query)
        where_filter = {"category": category} if category else None

        results = doc_col.query(
            query_embeddings=[query_vector],
            n_results=min(top_docs, total_docs),
            where=where_filter,
        )
        ids = (results.get("ids") or [[]])[0]
        if ids:
            logger.info("Hierarchical Stage 1 narrowed search to %d documents: %s", len(ids), ids)
            return [str(d) for d in ids]
    except Exception as exc:
        logger.debug("Hierarchical Stage 1 skipped: %s", exc)

    return None


def _get_dense_retriever(
    category: str | None = None,
    document_ids: list[str] | None = None,
):
    """Build LangChain Chroma retriever with metadata pre-filtering."""
    store = Chroma(
        client=get_chroma_client(),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings_model(),
    )

    filters = []
    if document_ids:
        if len(document_ids) == 1:
            filters.append({"document_id": document_ids[0]})
        else:
            filters.append({"document_id": {"$in": document_ids}})
    if category:
        filters.append({"category": category})

    where_clause = None
    if len(filters) == 1:
        where_clause = filters[0]
    elif len(filters) > 1:
        where_clause = {"$and": filters}

    search_kwargs = {"k": settings.DENSE_TOP_K}
    if where_clause:
        search_kwargs["filter"] = where_clause

    return store.as_retriever(search_kwargs=search_kwargs)


async def _postgres_fts_search(
    query: str,
    db: AsyncSession,
    category: str | None = None,
    document_ids: list[str] | None = None,
    top_k: int = settings.SPARSE_TOP_K,
) -> list[LCDocument]:
    """
    Native PostgreSQL Full-Text Search using GIN index on disk:
    Executes directly in PostgreSQL in sub-milliseconds without scanning in Python RAM.
    """
    try:
        ts_query = func.plainto_tsquery("english", query)
        ts_vector = func.to_tsvector("english", Chunk.text)

        stmt = (
            select(Chunk, Document.name)
            .join(Document, Chunk.document_id == Document.id)
            .where(ts_vector.op("@@")(ts_query))
        )
        if category:
            stmt = stmt.where(Document.category == category)
        if document_ids:
            doc_uuids = []
            for d in document_ids:
                try:
                    doc_uuids.append(uuid.UUID(str(d)))
                except (ValueError, TypeError):
                    pass
            if doc_uuids:
                stmt = stmt.where(Chunk.document_id.in_(doc_uuids))

        stmt = stmt.order_by(func.ts_rank_cd(ts_vector, ts_query).desc()).limit(top_k)
        result = await db.execute(stmt)
        rows = result.all()

        if rows:
            return [
                LCDocument(
                    page_content=c.text,
                    metadata={
                        "document_id": str(c.document_id),
                        "chunk_index": c.chunk_index,
                        "chroma_id": c.chroma_id,
                        "filename": doc_name or "Unknown Document",
                    },
                )
                for c, doc_name in rows
            ]
    except Exception as exc:
        logger.debug("PostgreSQL FTS fallback: %s", exc)

    return []


class _CachedBM25:
    """In-memory cache for BM25Retriever used as fallback for edge queries."""

    def __init__(self):
        self.retriever: BM25Retriever | None = None
        self.chunk_count: int = -1

    async def get(self, db: AsyncSession) -> BM25Retriever | None:
        count_result = await db.execute(select(func.count(Chunk.id)))
        current_count = count_result.scalar() or 0

        if current_count == 0:
            self.retriever = None
            self.chunk_count = 0
            return None

        if self.retriever is not None and self.chunk_count == current_count:
            return self.retriever

        result = await db.execute(
            select(Chunk, Document.name)
            .join(Document, Chunk.document_id == Document.id)
            .order_by(Chunk.document_id, Chunk.chunk_index)
        )
        rows = result.all()

        if not rows:
            self.retriever = None
            self.chunk_count = 0
            return None

        docs = [
            LCDocument(
                page_content=c.text,
                metadata={
                    "document_id": str(c.document_id),
                    "chunk_index": c.chunk_index,
                    "chroma_id": c.chroma_id,
                    "filename": doc_name or "Unknown Document",
                },
            )
            for c, doc_name in rows
        ]
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = settings.SPARSE_TOP_K
        self.retriever = retriever
        self.chunk_count = current_count
        return self.retriever


_bm25_cache = _CachedBM25()


async def _get_sparse_retriever(db: AsyncSession) -> BM25Retriever | None:
    """Build or retrieve cached LangChain BM25Retriever from PostgreSQL chunks."""
    return await _bm25_cache.get(db)


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


async def hybrid_retrieve(
    query: str,
    db: AsyncSession,
    category: str | None = None,
    document_id: str | None = None,
) -> list[dict]:
    """
    High-efficiency hybrid retrieval:
      1. Stage 1 Hierarchical Routing: narrows to top documents if not specified.
      2. Metadata Pre-Filtering: applies category/document filters to both stores.
      3. Native PostgreSQL GIN FTS: executes sparse search in-database.
      4. Reciprocal Rank Fusion: combines dense and sparse hits fairly.
    """
    # 1. Hierarchical stage: determine target document IDs
    target_doc_ids = [document_id] if document_id else None
    if not target_doc_ids:
        target_doc_ids = await _hierarchical_stage1_doc_ids(query, category=category)

    # 2. Dense retrieval with pre-filters
    dense_retriever = _get_dense_retriever(category=category, document_ids=target_doc_ids)
    dense_task = dense_retriever.ainvoke(query)

    # 3. Native PostgreSQL Full-Text Search with GIN index
    fts_docs = await _postgres_fts_search(
        query, db, category=category, document_ids=target_doc_ids
    )

    if fts_docs:
        dense_docs = await dense_task
        sparse_docs = fts_docs
    else:
        # Fallback to BM25 retriever if FTS returned zero hits
        sparse_retriever = await _get_sparse_retriever(db)
        if sparse_retriever:
            sparse_task = sparse_retriever.ainvoke(query)
            dense_docs, sparse_docs = await asyncio.gather(dense_task, sparse_task)
        else:
            dense_docs = await dense_task
            sparse_docs = []

    return _reciprocal_rank_fusion(dense_docs, sparse_docs)
