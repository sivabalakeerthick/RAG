"""
SQLAlchemy async engine + ORM base + session factory.
All services use get_db() as a FastAPI dependency.
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Enum, Float, ForeignKey,
    Integer, JSON, String, Text, text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from shared.config import settings


# ── Engine & Session ──────────────────────────────────────────────────────────

engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncSession:
    """FastAPI dependency that yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── ORM Base ──────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── ORM Models ────────────────────────────────────────────────────────────────

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(512), nullable=False)
    original_filename = Column(String(512), nullable=False)
    file_size = Column(String(50), nullable=False)
    category = Column(String(128), nullable=False)
    status = Column(
        Enum("Indexed", "Processing", name="doc_status"),
        nullable=False,
        default="Processing",
    )
    chunks_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)           # stored for BM25
    tokens = Column(Integer, nullable=False)
    chroma_id = Column(String(256), nullable=True) # ChromaDB document ID
    created_at = Column(DateTime, default=datetime.utcnow)


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)          # list of source dicts
    retrieval_method = Column(String(50), default="hybrid")
    latency_ms = Column(Float, nullable=True)
    # Judge scores (filled async by judge service)
    judge_faithfulness = Column(Float, nullable=True)
    judge_relevance = Column(Float, nullable=True)
    judge_completeness = Column(Float, nullable=True)
    judge_verdict = Column(String(20), nullable=True)  # PASS | FAIL
    judge_reasoning = Column(Text, nullable=True)
    user_feedback = Column(String(20), nullable=True)  # thumbs_up | thumbs_down
    # ── Agentic RAG (port 8005) ───────────────────────────────────────────────
    # Which pipeline produced the row, so agentic and baseline runs can be
    # compared side by side instead of being averaged together.
    agent_mode = Column(String(20), nullable=True, default="standard")   # standard | agentic
    # Greetings retrieve 0 chunks; tagging them lets the metrics service exclude
    # them so they cannot dilute retrieval precision/recall/F1.
    intent = Column(String(30), nullable=True, default="retrieval")      # greeting | retrieval
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Table Creation ────────────────────────────────────────────────────────────

# Idempotent column additions. create_all() only creates missing *tables*, it
# never alters an existing one — so every column added after a table first went
# live needs an explicit ADD COLUMN IF NOT EXISTS here.
_MIGRATIONS = (
    "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS user_feedback VARCHAR(20);",
    "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS agent_mode VARCHAR(20) DEFAULT 'standard';",
    "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS intent VARCHAR(30) DEFAULT 'retrieval';",
    # Backfill rows written before these columns existed, so metrics filtering
    # on agent_mode/intent does not silently drop all historical queries.
    "UPDATE query_logs SET agent_mode = 'standard' WHERE agent_mode IS NULL;",
    "UPDATE query_logs SET intent = 'retrieval' WHERE intent IS NULL;",
)


async def create_tables():
    """Create all tables on startup if they don't exist, and ensure schema migrations."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Each migration gets its own transaction: one failure (e.g. insufficient
    # privileges) must not roll back the ones that already succeeded.
    for statement in _MIGRATIONS:
        try:
            async with engine.begin() as conn:
                await conn.execute(text(statement))
        except Exception:
            pass
