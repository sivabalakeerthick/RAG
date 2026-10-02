"""
LangGraph state for the agentic RAG workflow.

Every node receives the whole state and returns a partial dict; LangGraph merges
the returned keys over the existing state. Keys are therefore never popped, only
overwritten.
"""
from typing import Any, Dict, List

from typing_extensions import TypedDict


class AgenticRAGState(TypedDict, total=False):
    """
    total=False so a node may return a partial update without LangGraph or a
    type checker demanding every key.
    """

    # ── Input ────────────────────────────────────────────────────────────────
    question: str
    # Whitespace-normalised question used for retrieval and caching.
    cleaned_query: str

    # ── Routing ──────────────────────────────────────────────────────────────
    # 'greeting' | 'retrieval' | 'out_of_scope'
    intent: str

    # ── Retrieval ────────────────────────────────────────────────────────────
    # Each dict: text, document_id, chunk_index, filename, category,
    #            similarity, bm25_score
    retrieved_docs: List[Dict[str, Any]]
    is_relevant: bool
    retry_count: int
    # Set by the grader to request exactly one broadened retry pass. Routing
    # reads this rather than inferring intent from retry_count, which cannot
    # distinguish "retry requested" from "retry already spent".
    should_retry: bool

    # ── Generation ───────────────────────────────────────────────────────────
    answer: str
    citations: List[Dict[str, Any]]
    confidence_score: float
    grounded: bool

    # ── Budget accounting ────────────────────────────────────────────────────
    # Gemini *generation* calls consumed. Must never exceed 1.
    api_calls_used: int
    # Gemini *embedding* calls consumed (separate endpoint, tracked separately).
    embedding_calls_used: int

    # ── Diagnostics ──────────────────────────────────────────────────────────
    # Ordered list of node names visited, returned to the UI/logs so the graph
    # path taken for a given answer is auditable.
    path: List[str]
    error: str


def initial_state(question: str) -> AgenticRAGState:
    """Build a fully-populated starting state so no node reads a missing key."""
    cleaned = " ".join(question.split())
    return AgenticRAGState(
        question=question,
        cleaned_query=cleaned,
        intent="retrieval",
        retrieved_docs=[],
        is_relevant=False,
        retry_count=0,
        should_retry=False,
        answer="",
        citations=[],
        confidence_score=0.0,
        grounded=False,
        api_calls_used=0,
        embedding_calls_used=0,
        path=[],
        error="",
    )
