"""
LangGraph state for the agentic RAG workflow.

Every node receives the state and returns a partial dict; LangGraph merges
the returned keys over the existing state.
"""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage


class AgenticRAGState(TypedDict, total=False):
    """
    total=False so a node may return a partial update without LangGraph or a
    type checker demanding every key.
    """

    # ── Input & Metadata Filters ─────────────────────────────────────────────
    question: str
    cleaned_query: str
    category: Optional[str]
    document_id: Optional[str]
    session_id: Optional[str]
    query_vector: Optional[List[float]]

    # ── Conversation Messages & Loop Tracking ────────────────────────────────
    # Maintains the conversation history, agent thoughts, tool calls, and observations
    messages: List[BaseMessage]
    iteration_count: int         # Loop guard (max 2 tool execution iterations)
    tool_calls_trace: List[Dict[str, Any]]  # Auditable list of tool executions

    # ── Routing & Intent ─────────────────────────────────────────────────────
    # 'greeting' | 'retrieval' | 'blocked'
    intent: str

    # ── Retrieval Candidates ─────────────────────────────────────────────────
    # Each dict: text, document_id, chunk_index, filename, category, similarity
    retrieved_docs: List[Dict[str, Any]]
    is_relevant: bool
    retry_count: int
    should_retry: bool

    # ── Generation & Output ──────────────────────────────────────────────────
    answer: str
    citations: List[Dict[str, Any]]
    confidence_score: float
    grounded: bool

    # ── Budget accounting ────────────────────────────────────────────────────
    api_calls_used: int          # Gemini *generation* calls consumed
    embedding_calls_used: int    # Gemini *embedding* calls consumed

    # ── Diagnostics ──────────────────────────────────────────────────────────
    path: List[str]              # Visited nodes for UI path trace
    error: str


def initial_state(
    question: str,
    category: Optional[str] = None,
    document_id: Optional[str] = None,
    session_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    rewritten_query: Optional[str] = None,
    initial_api_calls: int = 0,
) -> AgenticRAGState:
    """Build a fully-populated starting state with prior conversation history."""
    cleaned = (rewritten_query.strip() if rewritten_query else " ".join(question.split()))
    messages: List[BaseMessage] = []

    # Populate sliding-window conversation history if provided
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "human"):
                messages.append(HumanMessage(content=content))
            else:
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=content))

    # Add the current user question. If reformulated for multi-turn co-reference, include resolved context.
    user_prompt = question.strip()
    if rewritten_query and rewritten_query.strip().lower() != question.strip().lower():
        user_prompt = f"{question.strip()} (Search context: {rewritten_query.strip()})"
    messages.append(HumanMessage(content=user_prompt))

    return AgenticRAGState(
        question=question,
        cleaned_query=cleaned,
        category=category,
        document_id=document_id,
        session_id=session_id,
        query_vector=None,
        messages=messages,
        iteration_count=0,
        tool_calls_trace=[],
        intent="retrieval",
        retrieved_docs=[],
        is_relevant=False,
        retry_count=0,
        should_retry=False,
        answer="",
        citations=[],
        confidence_score=0.0,
        grounded=False,
        api_calls_used=initial_api_calls,
        embedding_calls_used=0,
        path=[],
        error="",
    )
