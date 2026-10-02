"""
LangGraph nodes: Router -> Retriever -> Grader -> Generator -> Verifier.

Budget rule enforced by construction: only `generate_node` may call a Gemini
*generation* endpoint, and it calls it at most once per graph run. Routing,
relevance grading and citation verification are pure local Python — regex,
vector math and set arithmetic — so they cost zero quota.

The one unavoidable network call outside generation is the query *embedding*
in `retrieve_node`: the Chroma collection holds Gemini vectors, so a query must
be embedded by the same model to be comparable. It hits a different endpoint
than generation, is counted separately in `embedding_calls_used`, and goes
through the same rate limiter.
"""
import logging
import re
import time
from functools import lru_cache
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rank_bm25 import BM25Okapi
from sqlalchemy import func, select

from shared.config import settings
from shared.db.chroma import get_collection
from shared.db.postgres import AsyncSessionLocal, Chunk
from shared.ssl_config import google_client_args
from services.document_service.core.embedder import embed_query
from services.agentic_rag_service.config import (
    CANDIDATE_FETCH_MULTIPLIER,
    MIN_CANDIDATE_FETCH,
    GENERATION_MAX_TOKENS,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    GREETING_REPLY,
    MAX_CONTEXT_CHARS,
    MAX_RETRIES,
    MIN_BM25_SCORE,
    MIN_COSINE_SIMILARITY,
    RETRIEVE_RETRY_TOP_K,
    RETRIEVE_TOP_K,
)
from services.agentic_rag_service.graph.state import AgenticRAGState
from services.agentic_rag_service.rate_limiter import (
    RateLimitExceeded,
    get_rate_limiter,
)

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "Information not found in available documents."


# ══════════════════════════════════════════════════════════════════════════════
# 1. ROUTER NODE — 0 API calls, sub-millisecond
# ══════════════════════════════════════════════════════════════════════════════

# Anchored at the start of the query so "hi" routes to greeting but
# "hierarchy of approval limits" does not (\b prevents the prefix match).
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|ok|okay"
    r"|cool|bye|goodbye|who are you|what are you|help)\b",
    re.IGNORECASE,
)

# A greeting-looking prefix followed by a real question is a retrieval query:
# "hi, what is the leave policy" must not be answered with a canned hello.
_SUBSTANTIVE_RE = re.compile(
    r"\b(what|when|where|which|who|why|how|list|show|explain|summar|describe"
    r"|policy|policies|procedure|process|document|report|cost|price|limit"
    r"|requirement|step|config|error|setup|install)\w*\b",
    re.IGNORECASE,
)


def router_node(state: AgenticRAGState) -> dict[str, Any]:
    """
    Classify intent locally. Greetings short-circuit to END with a canned
    reply and zero API calls.
    """
    query = state.get("cleaned_query", "").strip()
    path = [*state.get("path", []), "router"]

    if not query:
        return {
            "intent": "greeting",
            "answer": GREETING_REPLY,
            "path": path,
        }

    is_greeting = bool(_GREETING_RE.match(query))
    has_substance = bool(_SUBSTANTIVE_RE.search(query))

    # Short pure-greeting, or a greeting with no question words in it.
    if is_greeting and not has_substance:
        logger.info("[router] greeting -> 0 API calls | %r", query)
        return {
            "intent": "greeting",
            "answer": GREETING_REPLY,
            "citations": [],
            "grounded": True,
            "confidence_score": 1.0,
            "path": path,
        }

    logger.info("[router] retrieval | %r", query)
    return {"intent": "retrieval", "path": path}


def route_after_router(state: AgenticRAGState) -> str:
    """Conditional edge: greeting -> END, everything else -> retrieve."""
    return "greeting" if state.get("intent") == "greeting" else "retrieval"


# ══════════════════════════════════════════════════════════════════════════════
# 2. RETRIEVE NODE — 0 generation calls (1 embedding call)
# ══════════════════════════════════════════════════════════════════════════════

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "if", "in", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your",
    "please", "tell", "about", "would", "could", "should", "explain", "list",
    "show", "give",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _content_tokens(text: str) -> list[str]:
    """Tokens with stopwords and 1-character noise removed."""
    return [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 1]


class _BM25Index:
    """
    Cached BM25 index over all chunk rows in PostgreSQL.

    Rebuilding on every query (what the baseline service does) is the single
    biggest local cost at any real corpus size, so the index is cached and
    invalidated on chunk-row count change — which is exactly when an upload,
    reindex or delete has altered the corpus.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._records: list[dict[str, Any]] = []
        self._row_count: int = -1
        self._built_at: float = 0.0

    async def _load(self) -> None:
        async with AsyncSessionLocal() as db:
            count = (await db.execute(select(func.count(Chunk.id)))).scalar() or 0

            if self._bm25 is not None and count == self._row_count:
                return  # Corpus unchanged — reuse.

            result = await db.execute(
                select(Chunk).order_by(Chunk.document_id, Chunk.chunk_index)
            )
            chunks = result.scalars().all()

        if not chunks:
            self._bm25 = None
            self._records = []
            self._row_count = 0
            return

        self._records = [
            {
                "text": c.text,
                "document_id": str(c.document_id),
                "chunk_index": c.chunk_index,
                "chroma_id": c.chroma_id,
            }
            for c in chunks
        ]
        corpus = [_content_tokens(r["text"]) or ["_"] for r in self._records]
        self._bm25 = BM25Okapi(corpus)
        self._row_count = len(chunks)
        self._built_at = time.monotonic()
        logger.info("[bm25] index built over %d chunks", self._row_count)

    async def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return top_k records with their raw BM25 scores."""
        await self._load()
        if self._bm25 is None:
            return []

        tokens = _content_tokens(query) or _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # BM25Okapi's IDF term goes negative for a word present in more than
        # half the corpus, so on a small index every score can be <= 0. Dropping
        # those outright silently disables the entire sparse arm, which is
        # exactly what happened on this corpus. Prefer positive-scoring hits,
        # but fall back to the ranking rather than returning nothing.
        positive = [i for i in ranked if scores[i] > 0]
        chosen = positive[:top_k] if positive else ranked[:top_k]

        return [
            {**self._records[i], "bm25_score": max(0.0, float(scores[i]))}
            for i in chosen
        ]


@lru_cache
def _get_bm25_index() -> _BM25Index:
    return _BM25Index()


async def _dense_search(query: str, top_k: int) -> tuple[list[dict[str, Any]], int]:
    """
    Cosine-similarity search against ChromaDB.

    Returns (results, embedding_calls_used). The query must be embedded with the
    same Gemini model that produced the stored vectors, so this costs one
    embedding call — rate-limited like any other Gemini request.
    """
    limiter = get_rate_limiter()
    try:
        await limiter.acquire()
        vector = await embed_query(query)
    except RateLimitExceeded:
        raise
    except Exception as exc:
        logger.warning("[dense] query embedding failed: %s", exc)
        return [], 1

    try:
        collection = get_collection()
        res = collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.warning("[dense] Chroma query failed: %s", exc)
        return [], 1

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: list[dict[str, Any]] = []
    for text, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        # Chroma cosine space returns distance in [0, 2]; similarity = 1 - d.
        similarity = 1.0 - float(dist)
        out.append(
            {
                "text": text or "",
                "document_id": str(meta.get("document_id", "")),
                "chunk_index": meta.get("chunk_index", 0),
                "filename": meta.get("filename", "Unknown Document"),
                "category": meta.get("category", ""),
                "similarity": similarity,
                "bm25_score": 0.0,
            }
        )
    return out, 1


def _merge(
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Union dense and sparse hits, keyed on (document_id, chunk_index), keeping
    the best similarity and BM25 score seen for each chunk.

    Unlike the baseline service's RRF this preserves the *raw* scores, because
    the grader needs absolute values to threshold against — a fused rank score
    carries no information about whether anything was actually a good match.
    """
    merged: dict[str, dict[str, Any]] = {}

    for item in [*dense, *sparse]:
        key = f"{item.get('document_id', '')}:{item.get('chunk_index', 0)}"
        if key in merged:
            existing = merged[key]
            existing["similarity"] = max(
                existing.get("similarity", 0.0), item.get("similarity", 0.0)
            )
            existing["bm25_score"] = max(
                existing.get("bm25_score", 0.0), item.get("bm25_score", 0.0)
            )
            # Sparse records carry no filename; fill it in from whichever hit has one.
            if existing.get("filename", "Unknown Document") == "Unknown Document":
                existing["filename"] = item.get("filename", "Unknown Document")
        else:
            merged[key] = {
                "text": item.get("text", ""),
                "document_id": str(item.get("document_id", "")),
                "chunk_index": item.get("chunk_index", 0),
                "filename": item.get("filename", "Unknown Document"),
                "category": item.get("category", ""),
                "similarity": float(item.get("similarity", 0.0)),
                "bm25_score": float(item.get("bm25_score", 0.0)),
            }

    ranked = sorted(
        merged.values(),
        key=lambda d: (d["similarity"], d["bm25_score"]),
        reverse=True,
    )

    # Collapse chunks with identical text before truncating to top_k.
    #
    # The same file uploaded (or reindexed) more than once produces a separate
    # documents row per upload, so its chunks are genuinely distinct records
    # with distinct document_ids — the key-based merge above cannot tell them
    # apart. Without this, a corpus holding three copies of one PDF fills the
    # entire top-4 with three identical passages, wasting context and showing
    # the user the same citation three times.
    deduped: list[dict[str, Any]] = []
    seen_text: set[int] = set()
    for item in ranked:
        fingerprint = hash(" ".join(item.get("text", "").split()).lower())
        if fingerprint in seen_text:
            continue
        seen_text.add(fingerprint)
        deduped.append(item)
        if len(deduped) >= top_k:
            break

    return deduped


async def retrieve_node(state: AgenticRAGState) -> dict[str, Any]:
    """Hybrid retrieval over the shared ChromaDB + PostgreSQL chunk stores."""
    retry = state.get("retry_count", 0)
    path = [*state.get("path", []), "retrieve" if retry == 0 else "retrieve_retry"]

    # On the retry pass the query has been broadened by the grader.
    query = state.get("cleaned_query") or state.get("question", "")
    top_k = RETRIEVE_TOP_K if retry == 0 else RETRIEVE_RETRY_TOP_K

    # Over-fetch candidates, then dedupe, then cut to top_k.
    #
    # Asking the store for exactly top_k lets duplicate vectors consume the
    # whole budget: a corpus holding three copies of one PDF returns three
    # identical passages, which dedupe collapses to one, leaving the agent with
    # a fraction of the context it asked for. The baseline service hides this
    # by fetching 20 and never deduping. Fetch wide, narrow locally.
    fetch_k = max(top_k * CANDIDATE_FETCH_MULTIPLIER, MIN_CANDIDATE_FETCH)

    try:
        dense, embed_calls = await _dense_search(query, fetch_k)
    except RateLimitExceeded as exc:
        return {
            "retrieved_docs": [],
            "is_relevant": False,
            "should_retry": False,
            "error": f"rate_limited:{exc.retry_after:.1f}",
            "path": path,
        }

    sparse = await _get_bm25_index().search(query, fetch_k)
    docs = _merge(dense, sparse, top_k)

    logger.info(
        "[retrieve] pass=%d fetch_k=%d top_k=%d dense=%d sparse=%d -> unique=%d",
        retry, fetch_k, top_k, len(dense), len(sparse), len(docs),
    )

    return {
        "retrieved_docs": docs,
        # Consumed by re-entering this node; cleared so the grader decides afresh.
        "should_retry": False,
        "embedding_calls_used": state.get("embedding_calls_used", 0) + embed_calls,
        "path": path,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. GRADE DOCUMENTS NODE — 0 API calls, local math only
# ══════════════════════════════════════════════════════════════════════════════

def _broaden_query(question: str) -> str:
    """
    Local query expansion for the single retry pass — no LLM.

    Keeps the longest content words (the most discriminative terms) and applies
    crude suffix stripping so "reimbursements" also matches "reimbursement".
    """
    tokens = _content_tokens(question)
    if not tokens:
        return question

    stemmed: list[str] = []
    for t in sorted(set(tokens), key=len, reverse=True)[:6]:
        for suffix in ("ies", "es", "s", "ing", "ed"):
            if len(t) > 4 and t.endswith(suffix):
                t = t[: -len(suffix)]
                break
        stemmed.append(t)

    return " ".join(stemmed)


def grade_documents_node(state: AgenticRAGState) -> dict[str, Any]:
    """
    Decide locally whether the retrieved chunks are good enough to answer from.

    Thresholds are absolute, not relative: vector search always returns its
    top-K even when nothing matches, so "the best of K" says nothing about
    whether any of them are relevant.
    """
    docs = state.get("retrieved_docs", []) or []
    retry = state.get("retry_count", 0)
    path = [*state.get("path", []), "grade"]

    if state.get("error", "").startswith("rate_limited"):
        return {"is_relevant": False, "should_retry": False, "path": path}

    if not docs:
        max_sim, max_bm25 = 0.0, 0.0
    else:
        max_sim = max(float(d.get("similarity", 0.0)) for d in docs)
        max_bm25 = max(float(d.get("bm25_score", 0.0)) for d in docs)

    is_relevant = max_sim >= MIN_COSINE_SIMILARITY or max_bm25 >= MIN_BM25_SCORE

    logger.info(
        "[grade] max_sim=%.3f (>=%.2f) max_bm25=%.2f (>=%.1f) -> relevant=%s retry=%d",
        max_sim, MIN_COSINE_SIMILARITY, max_bm25, MIN_BM25_SCORE, is_relevant, retry,
    )

    if is_relevant:
        # Confidence blends normalised similarity with a saturating BM25 term.
        confidence = min(1.0, max(0.0, max_sim)) * 0.7 + min(1.0, max_bm25 / 20.0) * 0.3
        return {
            "is_relevant": True,
            "should_retry": False,
            "confidence_score": round(confidence, 3),
            "path": path,
        }

    # Loop guard: exactly one broadened retry, then give up. `should_retry` is
    # what routing keys off, and it is only ever True on the first failure.
    if retry < MAX_RETRIES:
        broadened = _broaden_query(state.get("question", ""))
        logger.info("[grade] retrying with broadened query %r", broadened)
        return {
            "is_relevant": False,
            "should_retry": True,
            "retry_count": retry + 1,
            "cleaned_query": broadened,
            "path": path,
        }

    logger.info("[grade] retry exhausted -> proceeding without relevant context")
    return {
        "is_relevant": False,
        "should_retry": False,
        "confidence_score": 0.0,
        "path": path,
    }


def route_after_grading(state: AgenticRAGState) -> str:
    """
    Conditional edge. 'retry' re-enters retrieve at most once, because only the
    first grading failure can set should_retry — so the graph cannot cycle.
    """
    if state.get("should_retry") and state.get("retry_count", 0) <= MAX_RETRIES:
        return "retry"
    return "generate"


# ══════════════════════════════════════════════════════════════════════════════
# 4. GENERATE NODE — exactly 1 Gemini generation call (0 when nothing to answer)
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are CogniDoc, an enterprise knowledge assistant.

Rules you must follow exactly:
1. Answer ONLY from the CONTEXT below. Never use outside knowledge.
2. Think step by step about which context passages support your answer before
   writing it, then give only the final answer — do not show your reasoning.
3. After each factual claim, cite its source inline as [Source: <document name>]
   using the document name exactly as it appears in the context header.
4. If the context does not contain enough information to answer, reply with
   exactly: Information not found in available documents.
5. Be concise and professional.
"""

_HUMAN_PROMPT = """\
CONTEXT:
{context}

---

QUESTION: {question}

Answer using only the context above, citing each claim as [Source: <document name>].
"""

_prompt = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
)


@lru_cache
def _get_generation_chain():
    """Singleton LCEL chain — building it per call would re-create the client."""
    model = ChatGoogleGenerativeAI(
        model=GENERATION_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=GENERATION_TEMPERATURE,
        max_output_tokens=GENERATION_MAX_TOKENS,
        client_args=google_client_args(),
    )
    return _prompt | model | StrOutputParser()


def _build_context(docs: list[dict[str, Any]]) -> str:
    """
    Render chunks with a document-name header so the model can cite by name and
    the verifier can check those names against what was actually retrieved.
    """
    blocks: list[str] = []
    for d in docs:
        header = f"[Document: {d.get('filename', 'Unknown Document')}"
        chunk_idx = d.get("chunk_index")
        if chunk_idx is not None:
            header += f" | chunk {chunk_idx}"
        header += "]"
        blocks.append(f"{header}\n{d.get('text', '')}")

    context = "\n\n---\n\n".join(blocks)
    return context[:MAX_CONTEXT_CHARS]


async def generate_node(state: AgenticRAGState) -> dict[str, Any]:
    """
    The only node permitted to spend generation quota, and only once.

    Two cases cost 0 calls rather than 1, because the answer is already known
    and spending quota on it would violate the budget mandate for no benefit:
      - the rate limiter refused the retrieval embedding
      - grading failed after the retry, so there is nothing to ground an answer in
    """
    path = [*state.get("path", []), "generate"]
    docs = state.get("retrieved_docs", []) or []

    if state.get("error", "").startswith("rate_limited"):
        retry_after = state["error"].split(":", 1)[1]
        return {
            "answer": (
                "The assistant is at its per-minute request limit. "
                f"Please try again in about {retry_after} seconds."
            ),
            "citations": [],
            "grounded": False,
            "confidence_score": 0.0,
            "path": path,
        }

    if not state.get("is_relevant") or not docs:
        logger.info("[generate] nothing relevant -> canned reply, 0 API calls")
        return {
            "answer": NOT_FOUND_MESSAGE,
            "citations": [],
            "grounded": True,   # Correctly declining to answer *is* grounded.
            "confidence_score": 0.0,
            "path": path,
        }

    limiter = get_rate_limiter()
    try:
        await limiter.acquire()
    except RateLimitExceeded as exc:
        return {
            "answer": (
                "The assistant is at its per-minute request limit. "
                f"Please try again in about {exc.retry_after:.0f} seconds."
            ),
            "citations": [],
            "grounded": False,
            "confidence_score": 0.0,
            "error": f"rate_limited:{exc.retry_after:.1f}",
            "path": path,
        }

    context = _build_context(docs)

    try:
        chain = _get_generation_chain()
        answer = (await chain.ainvoke({"question": state.get("question", ""), "context": context})).strip()
    except Exception as exc:
        logger.error("[generate] Gemini call failed: %s", exc)
        return {
            "answer": "The answer service is temporarily unavailable. Please try again.",
            "citations": [],
            "grounded": False,
            "confidence_score": 0.0,
            "error": f"generation_failed:{exc}",
            # The call was attempted and consumed a slot, so count it.
            "api_calls_used": state.get("api_calls_used", 0) + 1,
            "path": path,
        }

    return {
        "answer": answer or NOT_FOUND_MESSAGE,
        "api_calls_used": state.get("api_calls_used", 0) + 1,
        "path": path,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. VERIFY GROUNDING NODE — 0 API calls, local verification
# ══════════════════════════════════════════════════════════════════════════════

_CITATION_RE = re.compile(r"\[Source:\s*([^\]]+?)\s*\]", re.IGNORECASE)


def verify_grounding_node(state: AgenticRAGState) -> dict[str, Any]:
    """
    Verify the answer against what was actually retrieved — locally.

    Two independent checks:
      1. Citation validity — every [Source: X] must name a document that really
         is in retrieved_docs. A citation to anything else is a fabricated
         source, which is the most damaging hallucination in a RAG product.
      2. Lexical grounding — the answer's content words must overlap the
         context. A fluent answer sharing almost no vocabulary with its context
         was written from model priors, not from the documents.
    """
    path = [*state.get("path", []), "verify"]
    answer = state.get("answer", "") or ""
    docs = state.get("retrieved_docs", []) or []

    # A canned/declined answer needs no verification.
    if not docs or answer.strip() == NOT_FOUND_MESSAGE or state.get("error"):
        return {
            "citations": [],
            "grounded": state.get("grounded", not bool(state.get("error"))),
            "path": path,
        }

    retrieved_names = {
        str(d.get("filename", "")).strip().lower()
        for d in docs
        if d.get("filename")
    }

    cited_raw = [c.strip() for c in _CITATION_RE.findall(answer)]
    valid_names: set[str] = set()
    invalid_names: list[str] = []

    for cited in cited_raw:
        low = cited.lower()
        # Accept exact match or either-way substring (models often drop the
        # file extension, or cite "Policy.pdf" as "Policy").
        match = next(
            (
                name
                for name in retrieved_names
                if name == low or low in name or name in low
            ),
            None,
        )
        if match:
            valid_names.add(match)
        else:
            invalid_names.append(cited)

    # Lexical overlap between answer and context.
    context_tokens = set()
    for d in docs:
        context_tokens.update(_content_tokens(d.get("text", "")))

    answer_tokens = _content_tokens(_CITATION_RE.sub("", answer))
    if answer_tokens:
        overlap = sum(1 for t in answer_tokens if t in context_tokens) / len(answer_tokens)
    else:
        overlap = 0.0

    grounded = not invalid_names and overlap >= 0.45

    # Citations for the UI: the documents the answer actually used, falling back
    # to everything retrieved when the model cited nothing explicitly.
    if valid_names:
        used = [
            d for d in docs
            if str(d.get("filename", "")).strip().lower() in valid_names
        ]
    else:
        used = docs

    seen: set[str] = set()
    citations: list[dict[str, Any]] = []
    for d in used:
        name = str(d.get("filename", "Unknown Document"))
        if name in seen:
            continue
        seen.add(name)
        citations.append(
            {
                "fileName": name,
                "chunkLocation": f"Chunk {d.get('chunk_index', 0)}",
                "similarity": round(float(d.get("similarity", 0.0)), 3),
                "bm25Score": round(float(d.get("bm25_score", 0.0)), 2),
            }
        )

    # Fold verification into the confidence the grader produced.
    confidence = float(state.get("confidence_score", 0.0))
    confidence = confidence * (0.5 + 0.5 * min(1.0, overlap))
    if invalid_names:
        confidence *= 0.5
        logger.warning("[verify] fabricated citations: %s", invalid_names)

    logger.info(
        "[verify] overlap=%.2f valid_citations=%d invalid=%d -> grounded=%s",
        overlap, len(valid_names), len(invalid_names), grounded,
    )

    return {
        "citations": citations,
        "grounded": grounded,
        "confidence_score": round(min(1.0, max(0.0, confidence)), 3),
        "path": path,
    }
