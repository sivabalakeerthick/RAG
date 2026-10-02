"""
Service-specific configuration for the agentic RAG service.

Everything here is local to port 8005. Shared settings (API keys, DB URL,
Chroma path) still come from `shared.config.settings` so there is exactly one
source of truth for credentials.
"""
from shared.config import settings

# ── Gemini free-tier budget ───────────────────────────────────────────────────
# Free tier allows 15 requests/minute. We cap at 12 to leave headroom for the
# baseline rag_service and judge_service sharing the same key.
RATE_LIMIT_MAX_CALLS = 12
RATE_LIMIT_WINDOW_SECONDS = 60.0
# How long a caller will wait for a free slot before we give up and return 429.
RATE_LIMIT_MAX_WAIT_SECONDS = 30.0

# ── Retrieval ─────────────────────────────────────────────────────────────────
# Deliberately smaller than the baseline service's 20+20: the agentic path sends
# chunks straight to generation with no LLM reranker, so precision at small K
# matters more than recall at large K.
RETRIEVE_TOP_K = 4
# Widened K used on the single retry pass after a relevance failure.
RETRIEVE_RETRY_TOP_K = 8

# Candidates pulled from the stores before local dedup narrows them to top_k.
# Retrieval is free (no generation quota), so fetching wide costs nothing but
# protects against duplicate or near-identical chunks crowding out the answer.
CANDIDATE_FETCH_MULTIPLIER = 5
MIN_CANDIDATE_FETCH = 20

# ── Local relevance grading thresholds (no LLM) ───────────────────────────────
# Chroma collection uses cosine space, so similarity = 1 - distance.
# Spec defaults (0.55 / 8.0), overridable from .env via AGENTIC_MIN_SIMILARITY
# and AGENTIC_MIN_BM25 because the right cut-off depends on which embedding
# model produced the stored vectors.
MIN_COSINE_SIMILARITY = settings.AGENTIC_MIN_SIMILARITY
MIN_BM25_SCORE = settings.AGENTIC_MIN_BM25

# ── Loop guard ────────────────────────────────────────────────────────────────
MAX_RETRIES = 1
# router -> retrieve -> grade -> (retrieve -> grade) -> generate -> verify -> END
# is 8 supersteps; 15 leaves headroom without ever allowing a runaway loop.
GRAPH_RECURSION_LIMIT = 15

# ── Response cache ────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS = 300.0   # 5 minutes
CACHE_MAX_ENTRIES = 256

# ── Generation ────────────────────────────────────────────────────────────────
# Reuse the model the rest of the app is configured with rather than hardcoding
# a different one, so a model change in .env applies everywhere.
GENERATION_MODEL = settings.GEMINI_GENERATION_MODEL
GENERATION_TEMPERATURE = 0.2
GENERATION_MAX_TOKENS = 1024
# Hard cap on context characters handed to the single LLM call.
MAX_CONTEXT_CHARS = 8000

# ── Canned greeting reply (0 API calls) ───────────────────────────────────────
GREETING_REPLY = (
    "Hello! I'm CogniDoc's agentic assistant. Ask me anything about your "
    "indexed documents — policies, specs, support runbooks — and I'll answer "
    "using only what's in them, with the source documents listed."
)
