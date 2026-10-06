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
    content_hash = Column(String(64), nullable=True)   # SHA-256 file dedup


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


class ChatSession(Base):
    """Multi-turn conversation session — shared by both RAG pipelines."""
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_mode = Column(String(20), nullable=False, default="standard")  # standard | agentic
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SessionMessage(Base):
    """Individual message within a chat session (sliding window of last 4 used by LLM)."""
    __tablename__ = "session_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)        # 'user' | 'assistant'
    content = Column(Text, nullable=False)
    tool_calls = Column(JSON, nullable=True)          # Agentic audit trail
    created_at = Column(DateTime, default=datetime.utcnow)


class Category(Base):
    """Knowledge Base categories created and managed by administrators."""
    __tablename__ = "categories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    name = Column(String(128), unique=True, nullable=False)
    description = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, server_default=text("CURRENT_TIMESTAMP"))


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
    # ── Native PostgreSQL Full-Text Search (GIN) & Fast Filtering Indexes ─────
    "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING gin(to_tsvector('english', text));",
    # ── Agentic RAG v2.0: File deduplication & session storage ────────────────
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);",
    "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);",
    "CREATE INDEX IF NOT EXISTS idx_session_messages_session_id ON session_messages(session_id);",
    # Prune stale sessions older than 30 days (safe to re-run)
    "DELETE FROM chat_sessions WHERE updated_at < NOW() - INTERVAL '30 days';",
    # ── Category Management ──────────────────────────────────────────────────
    "CREATE TABLE IF NOT EXISTS categories ("
    "    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
    "    name VARCHAR(128) UNIQUE NOT NULL,"
    "    description VARCHAR(256),"
    "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    ");",
    "ALTER TABLE categories ALTER COLUMN id SET DEFAULT gen_random_uuid();",
    "ALTER TABLE categories ALTER COLUMN created_at SET DEFAULT CURRENT_TIMESTAMP;",
    "CREATE INDEX IF NOT EXISTS idx_categories_name ON categories(name);",
    # Seed default categories with explicit UUIDs
    "INSERT INTO categories (id, name, description) VALUES "
    "(gen_random_uuid(), 'Security & Policy', 'Security & Compliance'), "
    "(gen_random_uuid(), 'IT Support', 'IT & Infrastructure'), "
    "(gen_random_uuid(), 'Engineering', 'Engineering & API Specs'), "
    "(gen_random_uuid(), 'HR & Operations', 'HR & Legal Policies') "
    "ON CONFLICT (name) DO NOTHING;",
    # Seed any custom categories already in documents table
    "INSERT INTO categories (id, name, description) "
    "SELECT gen_random_uuid(), category, category || ' Category' "
    "FROM documents "
    "WHERE category IS NOT NULL AND category != '' "
    "ON CONFLICT (name) DO NOTHING;",
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
