"""
Centralised settings loaded from environment variables.
All services import from this module via: from shared.config import settings
"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BACKEND_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ── PostgreSQL ────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/cognidoc"

    # ── ChromaDB ─────────────────────────────────────────
    # PersistentClient stores data on disk at this path.
    CHROMA_PERSIST_PATH: str = str(BACKEND_DIR / "chroma_data")

    # ── Gemini ───────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_GENERATION_MODEL: str = "gemini-3.5-flash-lite"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-2"

    # ── Groq ─────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # ── Auth0 ─────────────────────────────────────────────
    AUTH0_DOMAIN: str = ""
    AUTH0_AUDIENCE: str = ""

    # ── Inter-service URLs ────────────────────────────────
    DOCUMENT_SERVICE_URL: str = "http://localhost:8001"
    RAG_SERVICE_URL: str = "http://localhost:8002"
    JUDGE_SERVICE_URL: str = "http://localhost:8003"
    METRICS_SERVICE_URL: str = "http://localhost:8004"
    AGENTIC_RAG_SERVICE_URL: str = "http://localhost:8005"

    # ── RAG Hyperparameters ───────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    DENSE_TOP_K: int = 20
    SPARSE_TOP_K: int = 20
    RERANK_TOP_K: int = 10
    FINAL_TOP_K: int = 5

    # ── Agentic RAG (port 8005) ───────────────────────────
    # Local relevance thresholds used by the agentic grader. Exposed here so
    # they can be tuned from .env without a code change — the right value
    # depends on the embedding model, and too high a bar makes the agent answer
    # "Information not found" for queries it could actually have handled.
    #
    # 0.25 is measured against this stack, not guessed: with the 3072-dim
    # gemini-embedding-2 vectors in ChromaDB, a clearly relevant chunk scores
    # ~0.32 cosine similarity and a clearly irrelevant one ~0.18. Thresholds
    # in the 0.55-0.85 range quoted for MiniLM/ada-style embeddings reject
    # everything here. Re-measure with scripts against real queries if the
    # embedding model in .env changes.
    AGENTIC_MIN_SIMILARITY: float = 0.25
    AGENTIC_MIN_BM25: float = 8.0

    # ── Upload ────────────────────────────────────────────
    UPLOAD_DIR: str = str(BACKEND_DIR / "uploads")
    MAX_UPLOAD_SIZE_MB: int = 25

    # ── CORS ─────────────────────────────────────────────
    ALLOWED_ORIGINS: str = "http://localhost:4200"

    # ── TLS ──────────────────────────────────────────────
    # Set false only behind an HTTPS-inspecting corporate proxy. See
    # shared/ssl_config.py.
    SSL_VERIFY: bool = True

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
