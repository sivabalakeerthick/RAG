"""
LangChain Google Generative AI Embeddings wrapper.
Uses langchain_google_genai.GoogleGenerativeAIEmbeddings.
Includes batching, key rotation, and exponential backoff on 429 quota exhaustion.
"""
import asyncio
import logging
import re
from fastapi import HTTPException
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from shared.config import settings
from shared.gemini_key_manager import get_gemini_api_key, get_gemini_key_count
from shared.ssl_config import google_client_args

logger = logging.getLogger(__name__)

# Free-tier quota is 100 embed requests/min. Slicing into batches of 30 allows
# safe rotation across multiple keys and prevents large documents from instantly tripping quota.
BATCH_SIZE = 30


def get_embeddings_model(key: str | None = None) -> GoogleGenerativeAIEmbeddings:
    """Returns LangChain GoogleGenerativeAIEmbeddings instance with specified or rotated API key."""
    model_name = settings.GEMINI_EMBEDDING_MODEL
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    api_key = key or get_gemini_api_key()
    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=api_key,
        client_args=google_client_args(),
    )


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using LangChain's aembed_documents with batching,
    per-batch key rotation, and automatic backoff on 429 RESOURCE_EXHAUSTED.
    """
    if not texts:
        return []

    num_keys = get_gemini_key_count()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        max_attempts = max(3, num_keys * 2)
        attempt = 0
        batch_embedded = False

        while attempt < max_attempts and not batch_embedded:
            attempt += 1
            embedder = get_embeddings_model()
            try:
                embeddings = await embedder.aembed_documents(batch)
                all_embeddings.extend(embeddings)
                batch_embedded = True
                if len(texts) > BATCH_SIZE:
                    await asyncio.sleep(0.1)
            except Exception as exc:
                err_str = str(exc)
                is_rate_limited = "RESOURCE_EXHAUSTED" in err_str or "429" in err_str
                if is_rate_limited and attempt < max_attempts:
                    if num_keys > 1:
                        logger.warning(
                            "[embedder] Key rate-limited on batch %d..%d, rotating to next key (attempt %d/%d)",
                            i, i + len(batch), attempt, max_attempts,
                        )
                        await asyncio.sleep(0.5)
                    else:
                        match = re.search(r"retry\s+(?:in\s+)?([0-9]+(?:\.[0-9]+)?)s?", err_str, re.IGNORECASE)
                        wait_sec = min(float(match.group(1)), 45.0) if match else min(5.0 * attempt, 25.0)
                        logger.warning(
                            "[embedder] Gemini rate limit reached. Waiting %.1fs before retry (attempt %d/%d)...",
                            wait_sec, attempt, max_attempts,
                        )
                        await asyncio.sleep(wait_sec)
                else:
                    logger.error("[embedder] Batch embedding failed permanently: %s", exc)
                    if is_rate_limited:
                        raise HTTPException(
                            status_code=429,
                            detail=(
                                "Google Gemini free-tier embedding quota (100 requests/min) exceeded. "
                                "Please wait ~45 seconds before uploading, or configure multiple "
                                "Gemini API keys in .env (GEMINI_API_KEYS=key1,key2,...) to expand quota."
                            ),
                        )
                    raise exc

    return all_embeddings


async def embed_query(query: str) -> list[float]:
    """
    Embed a single query string using LangChain's aembed_query.
    """
    embedder = get_embeddings_model()
    return await embedder.aembed_query(query)
