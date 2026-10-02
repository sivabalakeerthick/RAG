"""Shared Pydantic API models for chat — matches Angular ChatMessage interface."""
from pydantic import BaseModel


# ── API Models ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class SourceChunk(BaseModel):
    """Source citation returned alongside an AI answer."""
    id: str
    fileName: str
    chunkLocation: str
    score: str


class ChatMessage(BaseModel):
    """AI response returned to the frontend — mirrors Angular ChatMessage."""
    role: str = "ai"
    content: str
    sources: list[SourceChunk] = []
    queryLogId: str | None = None
    userFeedback: str | None = None


class FeedbackRequest(BaseModel):
    query_log_id: str
    feedback: str  # "thumbs_up" | "thumbs_down"


# ── Agentic RAG Models (port 8005) ───────────────────────────────────────────
# Kept separate from ChatMessage so the baseline /api/chat/query response shape
# on port 8002 is byte-for-byte unchanged.

class AgentQueryRequest(BaseModel):
    query: str


class AgentCitation(BaseModel):
    """A source the answer actually cited, verified locally against retrieval."""
    fileName: str
    chunkLocation: str
    similarity: float = 0.0
    bm25Score: float = 0.0


class AgentChatMessage(BaseModel):
    """Response from the LangGraph agentic pipeline."""
    role: str = "ai"
    content: str
    sources: list[SourceChunk] = []
    citations: list[AgentCitation] = []
    queryLogId: str | None = None
    userFeedback: str | None = None

    # ── Agentic telemetry ────────────────────────────────────────────────────
    agentMode: str = "agentic"
    intent: str = "retrieval"          # "greeting" | "retrieval"
    # True only when the model actually produced a document-backed answer.
    # False for greetings, rate-limit notices and "Information not found",
    # so the UI can distinguish "verified answer" from "declined to answer"
    # instead of inferring it from apiCallsUsed (which a cache hit resets to 0).
    answered: bool = False
    grounded: bool = False
    confidenceScore: float = 0.0
    # Gemini generation calls spent on this answer. Must never exceed 1.
    apiCallsUsed: int = 0
    embeddingCallsUsed: int = 0
    retryCount: int = 0
    cached: bool = False
    # Ordered node names visited, so the route taken is auditable from the UI.
    graphPath: list[str] = []
    latencyMs: float = 0.0


# ── Judge Models ─────────────────────────────────────────────────────────────

class JudgeEvalRequest(BaseModel):
    query_log_id: str
    query: str
    answer: str
    context: str  # concatenated chunk texts


class JudgeScore(BaseModel):
    faithfulness: float    # 0–10
    relevance: float       # 0–10
    completeness: float    # 0–10
    verdict: str           # "PASS" | "FAIL"
    reasoning: str
