"""
Metrics Service API route: GET /api/metrics

Aggregates live stats from PostgreSQL:
  - indexedDocuments / processingDocuments: document counts by status
  - vectorChunks:    chunk rows in PostgreSQL — the same source the document
                     table's per-file chunk count comes from, so the card and
                     the table cannot disagree
  - monthlyQueries:  query count in the last 30 days
  - groundingRate:   % of queries judged PASS by the LLM-as-Judge
  - avgLatencyMs:    average RAG latency (ms) in the last 30 days

The ChromaDB vector count is deliberately NOT read here: the persistent Chroma
client is single-process, so it is owned by the document service and exposed via
GET /api/documents/index-health.
"""
from datetime import datetime, timedelta
import time

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.postgres import Chunk, Document, QueryLog, get_db
from shared.models.document import DashboardMetrics

router = APIRouter()

# ── Metrics In-Memory Cache (15s TTL) ─────────────────────────────────────────
_metrics_cache: dict = {
    "timestamp": 0.0,
    "data": None,
}
METRICS_CACHE_TTL = 15.0


@router.get("/metrics", response_model=DashboardMetrics, tags=["metrics"])
async def get_metrics(
    refresh: bool = Query(default=False, description="Bypass cache and recalculate live"),
    db: AsyncSession = Depends(get_db),
):
    now = time.time()
    if not refresh and _metrics_cache["data"] is not None and (now - _metrics_cache["timestamp"] < METRICS_CACHE_TTL):
        return _metrics_cache["data"]

    thirty_days_ago = datetime.utcnow() - timedelta(days=30)

    # Document counts by status
    docs_result = await db.execute(
        select(Document.status, func.count(Document.id)).group_by(Document.status)
    )
    docs_by_status = {status: count for status, count in docs_result.all()}

    # Total vector chunks across all indexed documents
    chunks_result = await db.execute(select(func.count(Chunk.id)))
    total_chunks = chunks_result.scalar() or 0

    # Monthly query count
    monthly_result = await db.execute(
        select(func.count(QueryLog.id)).where(QueryLog.created_at >= thirty_days_ago)
    )
    monthly_queries = monthly_result.scalar() or 0

    # Grounding rate = PASS verdicts / total judged (in last 30 days)
    judged_result = await db.execute(
        select(func.count(QueryLog.id)).where(
            QueryLog.judge_verdict.isnot(None),
            QueryLog.created_at >= thirty_days_ago,
        )
    )
    pass_result = await db.execute(
        select(func.count(QueryLog.id)).where(
            QueryLog.judge_verdict == "PASS",
            QueryLog.created_at >= thirty_days_ago,
        )
    )
    total_judged = judged_result.scalar() or 0
    total_pass = pass_result.scalar() or 0
    grounding_rate = (
        f"{(total_pass / total_judged * 100):.1f}%" if total_judged > 0 else "N/A"
    )

    # Average RAG latency
    latency_result = await db.execute(
        select(func.avg(QueryLog.latency_ms)).where(
            QueryLog.latency_ms.isnot(None),
            QueryLog.created_at >= thirty_days_ago,
        )
    )
    avg_latency = latency_result.scalar() or 0.0

    # ── Confusion Matrix (TP, TN, FP, FN) from Judge + User Feedback ──────────
    eval_result = await db.execute(
        select(
            QueryLog.judge_faithfulness,
            QueryLog.judge_completeness,
            QueryLog.judge_relevance,
            QueryLog.judge_verdict,
            QueryLog.user_feedback,
        ).where(QueryLog.created_at >= thirty_days_ago)
    )
    eval_rows = eval_result.all()

    pos_feedback = sum(1 for r in eval_rows if r.user_feedback == "thumbs_up")
    neg_feedback = sum(1 for r in eval_rows if r.user_feedback == "thumbs_down")
    total_feedback = pos_feedback + neg_feedback

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for r in eval_rows:
        # Determine Predicted label from LLM-as-a-Judge verdict
        predicted = None
        if r.judge_verdict is not None:
            predicted = 1 if r.judge_verdict.upper() == "PASS" else 0
        elif r.judge_faithfulness is not None:
            predicted = 1 if float(r.judge_faithfulness) >= 7.0 else 0

        # Determine Actual (Ground Truth) label:
        # 1. User feedback takes primary precedence as human ground truth:
        #    - thumbs_up = Positive (verified accurate)
        #    - thumbs_down = Negative (inaccurate or hallucinated)
        # 2. If user hasn't voted, multi-criteria factual consensus acts as ground truth:
        #    - grounded (faithfulness >= 7) and relevant (relevance >= 7) = Positive
        #    - ungrounded or off-topic = Negative
        actual = None
        if r.user_feedback == "thumbs_up":
            actual = 1
        elif r.user_feedback == "thumbs_down":
            actual = 0
        elif r.judge_faithfulness is not None and r.judge_relevance is not None:
            actual = 1 if (float(r.judge_faithfulness) >= 7.0 and float(r.judge_relevance) >= 7.0) else 0

        # If both predicted and actual exist, categorize into confusion matrix
        if predicted is not None and actual is not None:
            if actual == 1 and predicted == 1:
                tp += 1
            elif actual == 0 and predicted == 1:
                fp += 1
            elif actual == 1 and predicted == 0:
                fn += 1
            elif actual == 0 and predicted == 0:
                tn += 1

    # Standard metrics calculation from confusion matrix:
    # Precision = TP / (TP + FP)
    # Recall    = TP / (TP + FN)
    # F1 Score  = 2 * (Precision * Recall) / (Precision + Recall)
    total_predicted_positive = tp + fp
    total_actual_positive = tp + fn

    if total_predicted_positive > 0:
        precision_val = tp / total_predicted_positive
        precision_str = f"{(precision_val * 100):.1f}%"
    else:
        precision_val = None
        precision_str = "N/A"

    if total_actual_positive > 0:
        recall_val = tp / total_actual_positive
        recall_str = f"{(recall_val * 100):.1f}%"
    else:
        recall_val = None
        recall_str = "N/A"

    if precision_val is not None and recall_val is not None and (precision_val + recall_val) > 0:
        f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)
        f1_str = f"{(f1_val * 100):.1f}%"
    else:
        f1_str = "N/A"

    metrics_obj = DashboardMetrics(
        indexedDocuments=docs_by_status.get("Indexed", 0),
        processingDocuments=docs_by_status.get("Processing", 0),
        vectorChunks=total_chunks,
        monthlyQueries=monthly_queries,
        groundingRate=grounding_rate,
        avgLatencyMs=round(avg_latency, 1),
        precision=precision_str,
        recall=recall_str,
        f1Score=f1_str,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        userFeedbackCount=total_feedback,
        positiveFeedbackCount=pos_feedback,
        negativeFeedbackCount=neg_feedback,
    )
    _metrics_cache["timestamp"] = time.time()
    _metrics_cache["data"] = metrics_obj
    return metrics_obj
