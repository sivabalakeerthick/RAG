"""
Multi-Turn Query Rewriter & Co-reference Resolver.

Takes the user's latest follow-up question and previous chat turns, and reformulates
it into a standalone, fully-grounded search query suitable for dense (ChromaDB)
and sparse (BM25) search.

Cost: 1 fast call (Gemini Flash Lite) only when conversation history is present.
If no history exists or the query is standalone, it short-circuits with 0 API calls.
"""
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from shared.config import settings
from shared.gemini_key_manager import get_gemini_api_key
from shared.ssl_config import google_client_args

logger = logging.getLogger(__name__)

_REWRITE_SYSTEM_PROMPT = """\
You are an expert search query reformulator for an enterprise knowledge retrieval engine.
Your task is to rewrite the user's latest question into a single, complete, self-contained search query based on the conversation history.

Rules:
1. Resolve all pronouns (it, this, that, they, them, he, she) and ambiguous references to the concrete entities discussed in previous turns.
2. Preserve all specific domain keywords, document names, and policy terms.
3. If the user's question is already fully self-contained (does not rely on previous turns), output it EXACTLY as-is.
4. If the question is a greeting, appreciation, or goodbye (e.g., "hi", "thanks", "ok"), output it as-is.
5. NEVER answer the question. Output ONLY the rewritten search query and nothing else.
6. Do NOT include quotes, bullet points, prefixes, or explanations in your output.
"""


def _extract_text(content: Any) -> str:
    """Extract string content from LangChain response whether it is str, list, or dict."""
    if not content:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"].strip())
                elif item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]).strip())
            else:
                parts.append(str(item).strip())
        return " ".join(p for p in parts if p).strip()
    if isinstance(content, dict) and "text" in content:
        return str(content["text"]).strip()
    return str(content).strip()


async def rewrite_query_for_search(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Reformulate a multi-turn user question into a self-contained search query.

    Args:
        query: Raw query from the user (e.g. "can it be extended?").
        history: Chronological list of message dicts: [{'role': 'user'|'assistant', 'content': '...'}]

    Returns:
        Self-contained query string optimized for dense and sparse retrieval.
    """
    cleaned = " ".join(query.strip().split())
    if not cleaned:
        return cleaned

    # Fast Short-Circuit 1: No history -> Cannot be a follow-up (0 API calls)
    if not history or len(history) == 0:
        return cleaned

    # Fast Short-Circuit 2: Exact single greeting -> 0 API calls
    if cleaned.lower() in ("hi", "hello", "hey", "thanks", "thank you", "bye", "ok", "okay"):
        return cleaned

    # Format history snippet (last 4 messages = 2 turns)
    history_snippets = []
    for m in history[-4:]:
        role = "User" if m.get("role") in ("user", "human") else "Assistant"
        content = m.get("content", "").strip()
        # Truncate assistant answers so prompt tokens remain tiny (<100 tokens)
        if len(content) > 300:
            content = content[:300] + "..."
        history_snippets.append(f"{role}: {content}")

    history_text = "\n".join(history_snippets)
    prompt_input = (
        f"<conversation_history>\n{history_text}\n</conversation_history>\n\n"
        f"<latest_question>\n{cleaned}\n</latest_question>"
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_GENERATION_MODEL,
            google_api_key=get_gemini_api_key(),
            max_output_tokens=128,
            client_args=google_client_args(),
        )

        response = await llm.ainvoke([
            SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=prompt_input),
        ])

        raw_text = _extract_text(getattr(response, "content", response))
        # Clean any accidental wrapping quotes
        rewritten = re.sub(r'^["\']|["\']$', '', raw_text).strip()

        if rewritten and rewritten.lower() != cleaned.lower():
            logger.info(
                "[QueryRewriter] Follow-up detected: %r -> rewritten to: %r",
                cleaned, rewritten,
            )
            return rewritten

        return cleaned

    except Exception as exc:
        logger.warning(
            "[QueryRewriter] Rewriter call failed (%s), falling back to original query: %r",
            exc, cleaned,
        )
        return cleaned
