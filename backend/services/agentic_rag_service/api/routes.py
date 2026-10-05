"""
Agentic RAG Service API — port 8005.

  POST /api/agent/query   Run the LangGraph workflow for one question
  GET  /api/agent/health  Liveness + live rate-limit / cache telemetry

Answers are logged to the shared `query_logs` table with agent_mode='agentic'
and an intent tag, so the existing /api/chat/feedback endpoint and the metrics
dashboard work against agentic rows with no changes on their side.
"""
import logging
import re
import time

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import settings
from shared.db.postgres import QueryLog, get_db
from shared.models.chat import (
    AgentChatMessage,
    AgentCitation,
    AgentQueryRequest,
    JudgeEvalRequest,
    SourceChunk,
)
from services.agentic_rag_service.cache import get_cache, normalise_key
from services.agentic_rag_service.config import (
    GENERATION_MODEL,
    MAX_RETRIES,
    RETRIEVE_TOP_K,
)
from services.agentic_rag_service.rate_limiter import get_rate_limiter
from services.agentic_rag_service.graph.workflow import run_workflow
from services.document_service.core.embedder import embed_query

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/agent/query", response_model=AgentChatMessage, tags=["agent"])
async def agent_query(
    req: AgentQueryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Execute the agentic graph.

    Budget per call: 0 Gemini generation calls for greetings and cache hits,
    otherwise exactly 1.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    started = time.perf_counter()
    cache = get_cache()
    cache_key = normalise_key(req.query)

    # ── Cache hit (Exact string): 0 API calls ─────────────────────────────────
    cached = cache.get(cache_key)
    if cached is not None:
        payload = cached.model_copy(
            update={
                "cached": True,
                "apiCallsUsed": 0,
                "embeddingCallsUsed": 0,
                "latencyMs": round((time.perf_counter() - started) * 1000, 1),
            }
        )
        logger.info("[agent] exact cache hit | %r", req.query)
        return payload

    # ── Cache hit (Semantic Vector Match): 0 chunk search, 0 generation calls ─
    query_vector = None
    if cache._vectors:
        try:
            query_vector = await embed_query(req.query)
            semantic_cached = cache.get_semantic(query_vector)
            if semantic_cached is not None:
                payload = semantic_cached.model_copy(
                    update={
                        "cached": True,
                        "apiCallsUsed": 0,
                        "embeddingCallsUsed": 1,
                        "latencyMs": round((time.perf_counter() - started) * 1000, 1),
                    }
                )
                logger.info("[agent] semantic cache hit (0 chunks searched) | %r", req.query)
                return payload
        except Exception as sem_exc:
            logger.debug("[agent] semantic cache check skipped: %s", sem_exc)

    # ── Run the graph with metadata filters ───────────────────────────────────
    try:
        state = await run_workflow(
            req.query, category=req.category, document_id=req.document_id
        )
    except Exception as exc:
        logger.exception("[agent] workflow failed")
        raise HTTPException(status_code=500, detail=f"Agentic workflow failed: {exc}")

    latency_ms = (time.perf_counter() - started) * 1000

    intent = state.get("intent", "retrieval")
    raw_answer = state.get("answer", "") or ""
    answer = re.sub(r"\[Source:\s*[^\]]+\]", "", raw_answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r" {2,}", " ", answer)
    citations = [AgentCitation(**c) for c in (state.get("citations") or [])]
    docs = state.get("retrieved_docs") or []

    # `sources` mirrors the baseline SourceChunk shape so the existing Angular
    # citation rendering can consume an agentic answer unchanged.
    #
    # Deliberately empty when grading rejected the retrieved chunks: those
    # documents did not support the answer, and listing them under a
    # "not found" reply would imply the opposite.
    answered = bool(state.get("is_relevant")) and int(state.get("api_calls_used", 0)) > 0
    seen_files = set()
    sources = []
    if answered:
        for i, d in enumerate(docs):
            fname = str(d.get("filename", "Unknown Document"))
            if fname in seen_files:
                continue
            seen_files.add(fname)
            sources.append(
                SourceChunk(
                    id=str(i),
                    fileName=fname,
                    chunkLocation=f"Chunk {d.get('chunk_index', i)}",
                    score=f"{max(0.0, float(d.get('rrf_score', d.get('similarity', 0.0)))):.0%}",
                )
            )

    response = AgentChatMessage(
        role="ai",
        content=answer,
        sources=sources,
        citations=citations,
        agentMode="agentic",
        intent=intent,
        answered=answered,
        grounded=bool(state.get("grounded", False)),
        confidenceScore=float(state.get("confidence_score", 0.0)),
        apiCallsUsed=int(state.get("api_calls_used", 0)),
        embeddingCallsUsed=int(state.get("embedding_calls_used", 0)),
        retryCount=int(state.get("retry_count", 0)),
        cached=False,
        graphPath=list(state.get("path") or []),
        latencyMs=round(latency_ms, 1),
    )

    # ── Log greetings without a query_logs row ────────────────────────────────
    # A greeting is not a retrieval event. It is tagged and stored so the
    # dashboard can count it, but it carries no judge evaluation and is excluded
    # from retrieval metrics by the metrics service.
    log = QueryLog(
        query=req.query,
        answer=answer,
        sources=[s.model_dump() for s in sources],
        retrieval_method="agentic_greeting" if intent == "greeting" else "agentic_hybrid",
        latency_ms=latency_ms,
        agent_mode="agentic",
        intent=intent,
    )
    db.add(log)
    await db.flush()
    response.queryLogId = str(log.id)

    # ── Judge only real retrieval answers ─────────────────────────────────────
    # Sending a canned greeting to the judge would spend quota grading a string
    # that was never generated from documents.
    should_judge = (
        intent == "retrieval"
        and bool(docs)
        and bool(state.get("is_relevant"))
        and int(state.get("api_calls_used", 0)) > 0
    )
    if should_judge:
        context = "\n\n---\n\n".join(str(d.get("text", "")) for d in docs)
        background_tasks.add_task(
            _call_judge,
            query_log_id=str(log.id),
            query=req.query,
            answer=answer,
            context=context,
        )

    # Only cache answers actually grounded in documents. Caching a rate-limit
    # notice or a "not found" would serve a transient failure for 5 minutes.
    if state.get("grounded") and intent == "retrieval" and not state.get("error"):
        cache.set(cache_key, response, vector=query_vector)

    logger.info(
        "[agent] intent=%s llm_calls=%d embed_calls=%d retries=%d grounded=%s %.0fms path=%s",
        intent,
        response.apiCallsUsed,
        response.embeddingCallsUsed,
        response.retryCount,
        response.grounded,
        latency_ms,
        " -> ".join(response.graphPath),
    )

    return response


@router.get("/agent/health", tags=["agent"])
async def agent_health():
    """Liveness plus the live budget telemetry the spec's RPM checks rely on."""
    return {
        "status": "ok",
        "service": "agentic-rag-service",
        "port": 8005,
        "model": GENERATION_MODEL,
        "maxLlmCallsPerQuery": 1,
        "maxRetries": MAX_RETRIES,
        "retrieveTopK": RETRIEVE_TOP_K,
        "rateLimiter": get_rate_limiter().snapshot(),
        "cache": get_cache().stats(),
    }


async def _call_judge(query_log_id: str, query: str, answer: str, context: str) -> None:
    """
    Background task — evaluates the agentic answer via the existing judge
    service. Failure is non-critical and never surfaces to the user.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                f"{settings.JUDGE_SERVICE_URL}/api/judge/evaluate",
                json=JudgeEvalRequest(
                    query_log_id=query_log_id,
                    query=query,
                    answer=answer,
                    context=context[:4000],
                ).model_dump(),
            )
    except Exception as exc:
        logger.warning("[agent] judge evaluation failed for %s: %s", query_log_id, exc)
