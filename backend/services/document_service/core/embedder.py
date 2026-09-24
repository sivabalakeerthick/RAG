"""
LangChain Google Generative AI Embeddings wrapper.
Uses langchain_google_genai.GoogleGenerativeAIEmbeddings.
"""
from functools import lru_cache
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from shared.config import settings
from shared.ssl_config import google_client_args


@lru_cache
def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """Returns singleton LangChain GoogleGenerativeAIEmbeddings instance."""
    model_name = settings.GEMINI_EMBEDDING_MODEL
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=settings.GEMINI_API_KEY,
        client_args=google_client_args(),
    )


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using LangChain's aembed_documents.
    """
    embedder = get_embeddings_model()
    return await embedder.aembed_documents(texts)


async def embed_query(query: str) -> list[float]:
    """
    Embed a single query string using LangChain's aembed_query.
    """
    embedder = get_embeddings_model()
    return await embedder.aembed_query(query)
