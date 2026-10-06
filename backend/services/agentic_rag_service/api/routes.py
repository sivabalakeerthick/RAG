"""
Agentic RAG Service API — port 8005.

  POST /api/agent/query   Run the LangGraph tool-calling workflow for one question
  GET  /api/agent/health  Liveness + live rate-limit / cache telemetry

Answers are logged to PostgreSQL:
  - Multi-turn conversation saved to `chat_sessions` and `session_messages`
  - Quality metrics logged to `query_logs` with agent_mode='agentic'
  - Async evaluation triggered via judge_service (:8003)
"""
from datetime import datetime
import logging
import re
import time
import uuid

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import settings
from shared.db.postgres import ChatSession, QueryLog, SessionMessage, get_db
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
from services.agentic_rag_service.graph.nodes import _extract_text_content
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
    Execute the autonomous tool-calling LangGraph workflow.
    Supports multi-turn sessions with sliding-window history.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    started = time.perf_counter()
    cache = get_cache()
    cache_key = normalise_key(req.query)

    # ── Multi-turn Session Management ────────────────────────────────────────
    session_uuid = None
    history: list[dict[str, str]] = []

    if req.session_id:
        try:
            session_uuid = uuid.UUID(str(req.session_id))
        except (ValueError, TypeError):
            session_uuid = uuid.uuid4()

        # Check or create ChatSession in PostgreSQL
        session_res = await db.execute(select(ChatSession).where(ChatSession.id == session_uuid))
        existing_session = session_res.scalar_one_or_none()
        if not existing_session:
            new_session = ChatSession(id=session_uuid, agent_mode="agentic")
            db.add(new_session)
            await db.flush()

        # Load sliding window history (last 4 messages: 2 turns)
        history_stmt = (
            select(SessionMessage)
            .where(SessionMessage.session_id == session_uuid)
            .order_by(SessionMessage.created_at.desc())
            .limit(4)
        )
        history_rows = (await db.execute(history_stmt)).scalars().all()
        # Order chronologically for the model
        history = [{"role": m.role, "content": m.content} for m in reversed(history_rows)]

    # ── Cache hit (Exact string — only for fresh first-turn queries) ──────────
    if not history:
        cached = cache.get(cache_key)
        if cached is not None:
            payload = cached.model_copy(
                update={
                    "cached": True,
                    "sessionId": str(session_uuid) if session_uuid else None,
                    "apiCallsUsed": 0,
                    "embeddingCallsUsed": 0,
                    "latencyMs": round((time.perf_counter() - started) * 1000, 1),
                }
            )
            logger.info("[agent] exact cache hit | %r", req.query)
            return payload

    # ── Run the Autonomous Tool-Calling Workflow ─────────────────────────────
    try:
        state = await run_workflow(
            req.query,
            category=req.category,
            document_id=req.document_id,
            session_id=str(session_uuid) if session_uuid else None,
            history=history,
        )
    except Exception as exc:
        logger.exception("[agent] workflow execution failed")
        raise HTTPException(status_code=500, detail=f"Agentic workflow failed: {exc}")

    latency_ms = (time.perf_counter() - started) * 1000

    intent = state.get("intent", "retrieval")
    raw_answer = _extract_text_content(state.get("answer", "") or "")
    answer = re.sub(r"\[Source:\s*[^\]]+\]", "", raw_answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r" {2,}", " ", answer)
    citations = [AgentCitation(**c) for c in (state.get("citations") or [])]
    docs = state.get("retrieved_docs") or []
    tool_trace = state.get("tool_calls_trace") or []

    # Map retrieved docs to SourceChunk shape
    answered = bool(state.get("is_relevant")) or (int(state.get("api_calls_used", 0)) > 0 and bool(answer))
    seen_files = set()
    sources = []
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
                score=f"{max(0.0, float(d.get('similarity', 0.8))):.0%}",
            )
        )

    response = AgentChatMessage(
        role="ai",
        content=answer,
        sources=sources,
        citations=citations,
        agentMode="agentic",
        intent=intent,
        sessionId=str(session_uuid) if session_uuid else None,
        answered=answered,
        grounded=bool(state.get("grounded", False)),
        confidenceScore=float(state.get("confidence_score", 0.0)),
        apiCallsUsed=int(state.get("api_calls_used", 0)),
        embeddingCallsUsed=int(state.get("embedding_calls_used", 0)),
        retryCount=int(state.get("iteration_count", 0)),
        cached=False,
        graphPath=list(state.get("path") or []),
        toolCalls=tool_trace,
        latencyMs=round(latency_ms, 1),
    )

    # ── Persist Session Messages to PostgreSQL ───────────────────────────────
    if session_uuid:
        db.add(SessionMessage(session_id=session_uuid, role="user", content=req.query))
        db.add(SessionMessage(
            session_id=session_uuid,
            role="assistant",
            content=answer,
            tool_calls=tool_trace,
        ))

    method = "agentic+rewriter" if state.get("cleaned_query") and state.get("cleaned_query") != req.query else "agentic_tool_react"
    log = QueryLog(
        query=req.query,
        answer=answer,
        sources=[s.model_dump() for s in sources],
        retrieval_method=method,
        latency_ms=latency_ms,
        agent_mode="agentic",
        intent=intent,
    )
    db.add(log)
    await db.commit()
    response.queryLogId = str(log.id)

    # ── Judge real answers asynchronously ────────────────────────────────────
    should_judge = (
        intent == "retrieval"
        and bool(answer)
        and int(state.get("api_calls_used", 0)) > 0
        and not answer.startswith("Security Alert")
    )
    if should_judge:
        context_preview = "\n\n---\n\n".join(str(d.get("text", "")) for d in docs) if docs else str(tool_trace)
        background_tasks.add_task(
            _call_judge,
            query_log_id=str(log.id),
            query=req.query,
            answer=answer,
            context=context_preview,
        )

    # Cache standalone verified answers (only when no session history involved)
    if not history and state.get("grounded") and intent == "retrieval" and not state.get("error"):
        cache.set(cache_key, response)

    logger.info(
        "[agent] session=%s intent=%s llm_calls=%d tools_used=%d grounded=%s %.0fms path=%s",
        str(session_uuid)[:8] if session_uuid else "none",
        intent,
        response.apiCallsUsed,
        len(tool_trace),
        response.grounded,
        latency_ms,
        " -> ".join(response.graphPath),
    )

    return response


@router.get("/agent/health", tags=["agent"])
async def agent_health():
    """Liveness plus live telemetry."""
    return {
        "status": "ok",
        "service": "agentic-rag-service",
        "port": 8005,
        "model": GENERATION_MODEL,
        "paradigm": "ReAct Tool-Calling Agent",
        "tools": ["search_knowledge_base", "query_database_metadata", "get_adjacent_chunks"],
        "maxIterations": 2,
        "rateLimiter": get_rate_limiter().snapshot(),
        "cache": get_cache().stats(),
    }


async def _call_judge(query_log_id: str, query: str, answer: str, context: str) -> None:
    """Background task evaluating answer quality via judge_service (:8003)."""
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
