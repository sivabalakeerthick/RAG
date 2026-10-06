"""
Standalone Agent Tools for CogniDoc Agentic RAG.

These tools interact directly with the shared ChromaDB collection and
PostgreSQL database — zero imports from services.rag_service.

Tools:
  1. search_knowledge_base   – Dense vector search + Jaccard chunk dedup
  2. query_database_metadata  – Read-only SQL analytics on documents/chunks
  3. get_adjacent_chunks       – Context window stitching from PostgreSQL
"""
import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from sqlalchemy import func, select

from shared.db.chroma import get_collection
from shared.db.postgres import AsyncSessionLocal, Chunk, Document
from shared.security_guard import sanitize_context_text
from services.document_service.core.embedder import embed_query

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval-Time Chunk Deduplication (Jaccard Similarity)
# ══════════════════════════════════════════════════════════════════════════════

def _deduplicate_chunks(chunks: list[dict], threshold: float = 0.70) -> list[dict]:
    """
    Filter out chunks that share >= 70% word overlap with an already-selected
    chunk.  Prevents overlapping sliding-window chunks and boilerplate headers
    from consuming LLM context tokens.
    """
    unique: list[dict] = []
    seen_word_sets: list[set[str]] = []
    for chunk in chunks:
        words = set(chunk.get("text", "").lower().split())
        if not words:
            continue
        is_duplicate = any(
            len(words & seen) / max(len(words | seen), 1) >= threshold
            for seen in seen_word_sets
        )
        if not is_duplicate:
            unique.append(chunk)
            seen_word_sets.append(words)
    return unique


# ══════════════════════════════════════════════════════════════════════════════
# Tool 1: Semantic Vector Search (ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════

@tool
async def search_knowledge_base(
    query: str,
    category: Optional[str] = None,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """
    Search document text chunks by semantic similarity in ChromaDB.
    Applies Jaccard lexical deduplication to remove overlapping chunk windows.

    Use this when searching for policies, guidelines, procedures, rules,
    code documentation, or any textual facts stored in the knowledge base.

    Args:
        query: The search query describing what information is needed.
        category: Optional document category filter (e.g. 'HR', 'Engineering').
        top_k: Number of top chunks to return after deduplication (default 6).

    Returns:
        List of chunk dicts with keys: text, document_id, filename, chunk_index, category.
    """
    col = get_collection()
    query_vector = await embed_query(query)
    where_filter = {"category": category} if category else None

    # Fetch candidates from ChromaDB
    results = col.query(
        query_embeddings=[query_vector],
        n_results=min(top_k * 3, 20),
        where=where_filter,
    )

    docs: list[dict] = []
    if results and results.get("documents"):
        for i, text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            distance = results["distances"][0][i] if results.get("distances") else 1.0
            docs.append({
                "text": sanitize_context_text(text),
                "document_id": meta.get("document_id"),
                "filename": meta.get("filename", "Unknown Document"),
                "chunk_index": meta.get("chunk_index", i),
                "category": meta.get("category"),
                "similarity": round(1.0 - float(distance), 4),
            })

    # Exact keyword enrichment from PostgreSQL chunks for domain term recall
    try:
        async with AsyncSessionLocal() as session:
            keywords = [w.strip() for w in query.split() if len(w.strip()) > 3][:3]
            if keywords:
                from sqlalchemy import or_
                conditions = [Chunk.text.ilike(f"%{kw}%") for kw in keywords]
                stmt = (
                    select(Chunk, Document.name, Document.category)
                    .join(Document, Chunk.document_id == Document.id)
                    .where(or_(*conditions))
                    .limit(top_k)
                )
                if category:
                    stmt = stmt.where(Document.category == category)
                res = await session.execute(stmt)
                for c, doc_name, doc_cat in res.all():
                    docs.append({
                        "text": sanitize_context_text(c.text),
                        "document_id": str(c.document_id),
                        "filename": doc_name or "Unknown Document",
                        "chunk_index": c.chunk_index,
                        "category": doc_cat,
                        "similarity": 0.85,
                    })
    except Exception as exc:
        logger.warning("[tool:search_knowledge_base] pg keyword search fallback error: %s", exc)

    deduped = _deduplicate_chunks(docs, threshold=0.70)
    result = deduped[:top_k]

    logger.info(
        "[tool:search_knowledge_base] query=%r category=%s -> %d raw, %d deduped, %d returned",
        query[:60], category, len(docs), len(deduped), len(result),
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Tool 2: Structured Metadata Queries (PostgreSQL)
# ══════════════════════════════════════════════════════════════════════════════

@tool
async def query_database_metadata(
    action: str,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Query structured metadata from the PostgreSQL database.

    Actions available:
      - 'count_documents': Returns total document count and counts grouped by category.
      - 'list_documents': Returns document filenames, categories, and chunk counts.
      - 'document_stats': Returns overall chunk and document metrics.

    Use this for analytical, quantitative, counting, or overview questions that
    semantic text search cannot answer (e.g. "how many documents do we have",
    "list all HR documents", "what categories exist").

    Args:
        action: One of 'count_documents', 'list_documents', 'document_stats'.
        category: Optional category filter for list_documents.

    Returns:
        Dict with the query results.
    """
    async with AsyncSessionLocal() as session:
        if action == "count_documents":
            stmt = (
                select(Document.category, func.count(Document.id))
                .group_by(Document.category)
            )
            res = await session.execute(stmt)
            counts = {cat: count for cat, count in res.all()}
            total = sum(counts.values())
            logger.info("[tool:query_database_metadata] count_documents -> %d total", total)
            return {"total_documents": total, "by_category": counts}

        elif action == "list_documents":
            stmt = select(Document.name, Document.category, Document.chunks_count, Document.status)
            if category:
                stmt = stmt.where(Document.category == category)
            stmt = stmt.order_by(Document.created_at.desc())
            res = await session.execute(stmt)
            docs = [
                {"name": r[0], "category": r[1], "chunks": r[2], "status": r[3]}
                for r in res.all()
            ]
            logger.info("[tool:query_database_metadata] list_documents -> %d docs", len(docs))
            return {"documents": docs}

        elif action == "document_stats":
            doc_cnt = (await session.execute(select(func.count(Document.id)))).scalar() or 0
            chunk_cnt = (await session.execute(select(func.count(Chunk.id)))).scalar() or 0
            logger.info(
                "[tool:query_database_metadata] document_stats -> %d docs, %d chunks",
                doc_cnt, chunk_cnt,
            )
            return {"total_documents": doc_cnt, "total_chunks": chunk_cnt}

        return {
            "error": (
                f"Invalid action '{action}'. "
                "Valid actions: count_documents, list_documents, document_stats."
            )
        }


# ══════════════════════════════════════════════════════════════════════════════
# Tool 3: Adjacent Chunk Retrieval (PostgreSQL)
# ══════════════════════════════════════════════════════════════════════════════

@tool
async def get_adjacent_chunks(
    document_id: str,
    chunk_index: int,
    window: int = 1,
) -> List[Dict[str, Any]]:
    """
    Fetch chunks immediately before and/or after a given chunk index from PostgreSQL.

    Use this when a retrieved chunk appears truncated mid-sentence, ends with
    an incomplete list, references 'the following section', 'continued below',
    or 'see next page', or when you need surrounding context to fully answer
    the user's question.

    Args:
        document_id: The UUID of the document containing the chunk.
        chunk_index: The index of the chunk whose neighbors are needed.
        window: Number of chunks to fetch on each side (default 1 = prev + next).

    Returns:
        List of chunk dicts ordered by chunk_index, each with: chunk_index, text, filename.
    """
    try:
        doc_uuid = uuid.UUID(str(document_id))
    except (ValueError, TypeError):
        return [{"error": f"Invalid document_id format: '{document_id}'"}]

    async with AsyncSessionLocal() as session:
        start = max(0, chunk_index - window)
        end = chunk_index + window
        target_indices = list(range(start, end + 1))

        stmt = (
            select(Chunk, Document.name)
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Chunk.document_id == doc_uuid,
                Chunk.chunk_index.in_(target_indices),
            )
            .order_by(Chunk.chunk_index)
        )
        res = await session.execute(stmt)
        chunks = [
            {
                "chunk_index": c.chunk_index,
                "text": sanitize_context_text(c.text),
                "filename": doc_name,
                "document_id": str(c.document_id),
            }
            for c, doc_name in res.all()
        ]

        logger.info(
            "[tool:get_adjacent_chunks] doc=%s center=%d window=%d -> %d chunks",
            document_id[:8], chunk_index, window, len(chunks),
        )
        return chunks
