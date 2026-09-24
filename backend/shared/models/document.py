"""Shared Pydantic API models for documents — matches Angular DocumentFile interface."""
from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentStatus(str, Enum):
    indexed = "Indexed"
    processing = "Processing"


# ── API Response Models (camelCase to match Angular) ─────────────────────────

class DocumentFile(BaseModel):
    """Returned to the frontend — mirrors Angular DocumentFile interface."""
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    size: str
    category: str
    chunksCount: int
    status: DocumentStatus
    lastUpdated: str


class VectorChunk(BaseModel):
    """A single vector chunk returned to the chunk inspector drawer."""
    id: str
    chunkIndex: int
    tokens: int
    textExcerpt: str
    # Embedding state in ChromaDB. A similarity score is only defined relative
    # to a query, so a stored chunk reports whether it is embedded instead.
    embedded: bool
    embeddingDim: int | None = None


class DashboardMetrics(BaseModel):
    """Dashboard stat cards data."""
    indexedDocuments: int
    processingDocuments: int
    vectorChunks: int          # chunk rows in PostgreSQL
    monthlyQueries: int
    groundingRate: str
    avgLatencyMs: float
    precision: str = "N/A"
    recall: str = "N/A"
    f1Score: str = "N/A"
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    userFeedbackCount: int = 0
    positiveFeedbackCount: int = 0
    negativeFeedbackCount: int = 0


class IndexHealth(BaseModel):
    """
    PostgreSQL ↔ ChromaDB reconciliation, served by the document service (the
    owner of the persistent Chroma client).
    """
    chunkRows: int      # chunk rows in PostgreSQL
    vectors: int        # vectors in ChromaDB
    inSync: bool


# ── Internal DB Models (snake_case) ──────────────────────────────────────────

class DocumentCreate(BaseModel):
    name: str
    original_filename: str
    file_size: str
    category: str


class DocumentUpdate(BaseModel):
    status: DocumentStatus | None = None
    chunks_count: int | None = None


class DocumentMetadataUpdate(BaseModel):
    """PATCH /api/documents/{id} body — editable metadata from the admin table."""
    name: str | None = Field(default=None, min_length=1, max_length=512)
    category: str | None = Field(default=None, min_length=1, max_length=128)


class ChunkCreate(BaseModel):
    document_id: UUID
    chunk_index: int
    text: str
    tokens: int
    chroma_id: str
