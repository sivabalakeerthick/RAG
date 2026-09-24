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
