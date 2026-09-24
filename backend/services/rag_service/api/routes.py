"""
RAG Service route: POST /api/chat/query

Pipeline:
  1. Embed query (Gemini gemini-embedding-2)
  2. Dense retrieval (ChromaDB, top-20)
  3. Sparse retrieval (BM25 over all chunks in PostgreSQL, top-20)
  4. RRF fusion → top-10
  5. Rerank with Gemini → top-5
  6. Generate answer with Gemini Flash Lite
  7. Log to PostgreSQL
  8. Async: call Judge Service
  9. Return ChatMessage
"""
import time
import uuid

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import settings
from shared.db.postgres import QueryLog, get_db
from shared.models.chat import (
    ChatMessage,
    FeedbackRequest,
    JudgeEvalRequest,
    QueryRequest,
    SourceChunk,
)
from services.rag_service.core.retriever import hybrid_retrieve
from services.rag_service.core.reranker import rerank_chunks
from services.rag_service.core.generator import generate_answer

router = APIRouter()


@router.post("/chat/query", response_model=ChatMessage, tags=["chat"])
async def query(
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start_ms = time.time() * 1000

    # 1-4: Hybrid retrieval + RRF fusion
    ranked_chunks = await hybrid_retrieve(req.query, db)

    if not ranked_chunks:
        return ChatMessage(
            role="ai",
            content="I could not find relevant information in the knowledge base for your query.",
            sources=[],
        )

    # 5: Reranking with Gemini
    reranked = await rerank_chunks(req.query, ranked_chunks)
    final_chunks = reranked[: settings.FINAL_TOP_K]

    # 6: Answer generation
    context = "\n\n---\n\n".join(c["text"] for c in final_chunks)
    answer_text = await generate_answer(req.query, context)

    # Build sources for frontend
    sources = [
        SourceChunk(
            id=str(i),
            fileName=c.get("filename", "Unknown"),
            chunkLocation=f"Chunk {c.get('chunk_index', i)}",
            score=f"{c.get('rrf_score', 0):.0%}",
        )
        for i, c in enumerate(final_chunks)
    ]

    latency_ms = (time.time() * 1000) - start_ms

    # 7: Log to PostgreSQL
    log = QueryLog(
        query=req.query,
        answer=answer_text,
        sources=[s.model_dump() for s in sources],
        retrieval_method="hybrid",
        latency_ms=latency_ms,
    )
    db.add(log)
    await db.flush()
    log_id = str(log.id)

    # 8: Async LLM-as-Judge evaluation (non-blocking)
    background_tasks.add_task(
        _call_judge,
        query_log_id=log_id,
        query=req.query,
        answer=answer_text,
        context=context,
    )

    return ChatMessage(
        role="ai",
        content=answer_text,
        sources=sources,
        queryLogId=log_id,
    )


@router.post("/chat/feedback", tags=["chat"])
async def submit_feedback(
    req: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Records user feedback (thumbs_up or thumbs_down) for an answered query.
    Used alongside LLM Judge evaluations to compute Precision, Recall, and F1.
    """
    if req.feedback not in ("thumbs_up", "thumbs_down"):
        raise HTTPException(
            status_code=400,
            detail="Feedback must be 'thumbs_up' or 'thumbs_down'",
        )

    try:
        query_uuid = uuid.UUID(req.query_log_id)
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid query_log_id format. Must be a valid UUID.",
        )

    result = await db.execute(
        update(QueryLog)
        .where(QueryLog.id == query_uuid)
        .values(user_feedback=req.feedback)
    )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Query log entry not found.")

    await db.commit()
    return {"status": "ok", "query_log_id": req.query_log_id, "feedback": req.feedback}


async def _call_judge(query_log_id: str, query: str, answer: str, context: str) -> None:
    """Background task — calls judge service, does not block the main response."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                f"{settings.JUDGE_SERVICE_URL}/api/judge/evaluate",
                json=JudgeEvalRequest(
                    query_log_id=query_log_id,
                    query=query,
                    answer=answer,
                    context=context,
                ).model_dump(),
            )
    except Exception as exc:
        # Judge failure is non-critical — log and continue
        print(f"[Judge] evaluation failed for log {query_log_id}: {exc}")
