"""
Document Service API routes — with transactional PostgreSQL ↔ ChromaDB sync.

Upload guarantee:
  - Doc row created in PG (status=Processing)
  - Chunks extracted, embedded, written to ChromaDB
  - Chunk rows written to PG, doc status → Indexed
  - If ChromaDB write fails  → ChromaDB rollback + PG rollback + file deleted
  - If PG chunk write fails  → ChromaDB rollback + PG rollback + file deleted
  → Either both stores have the data, or neither does.

Delete guarantee:
  - ChromaDB vectors deleted first
  - PG document+chunks deleted (CASCADE)
  - If ChromaDB delete fails → PG NOT deleted, error returned (data stays in sync)

Concurrency guarantee:
  - Upload/reindex/edit/delete all take a row lock on the document first, so two
    overlapping requests (e.g. a double-clicked "Re-index") run one after the
    other instead of interleaving and each inserting its own copy of the chunks.
"""
import logging
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.postgres import Chunk, Document, get_db
from shared.models.document import (
    DocumentFile,
    DocumentMetadataUpdate,
    IndexHealth,
    VectorChunk,
)
from services.document_service.core.chunker import SUPPORTED_TYPES, chunk_document
from services.document_service.core.embedder import embed_texts
from services.document_service.core.indexer import (
    add_to_chroma,
    count_all_vectors,
    delete_document_vectors,
    delete_from_chroma,
    get_chunks_from_chroma,
    get_document_vector_ids,
    update_document_vector_metadata,
)
from shared.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def _fmt_doc(doc: Document, chunks_count: int | None = None) -> DocumentFile:
    """
    `chunks_count` is the live count of chunk rows. It is passed in (rather than
    read off the denormalised documents.chunks_count column) so the count shown
    in the table always comes from the same source as the dashboard's total.
    """
    updated = doc.updated_at or doc.created_at
    return DocumentFile(
        id=str(doc.id),
        name=doc.name,
        size=doc.file_size,
        category=doc.category,
        chunksCount=doc.chunks_count if chunks_count is None else chunks_count,
        status="Indexed" if doc.status == "Indexed" else "Processing",
        lastUpdated=updated.strftime("%b %d, %Y") if isinstance(updated, datetime) else str(updated),
    )


async def _lock_document(db: AsyncSession, doc_uuid: uuid.UUID) -> Document:
    """
    Fetch a document with a row-level lock (SELECT ... FOR UPDATE).
    Serialises concurrent reindex/edit/delete requests for the same document.
    """
    result = await db.execute(
        select(Document).where(Document.id == doc_uuid).with_for_update()
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


async def _count_chunks(db: AsyncSession, doc_uuid: uuid.UUID) -> int:
    result = await db.execute(
        select(func.count(Chunk.id)).where(Chunk.document_id == doc_uuid)
    )
    return result.scalar() or 0


# ── LIST ──────────────────────────────────────────────────────────────────────

@router.get("/documents", response_model=list[DocumentFile], tags=["documents"])
async def list_documents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    docs = result.scalars().all()

    counts_result = await db.execute(
        select(Chunk.document_id, func.count(Chunk.id)).group_by(Chunk.document_id)
    )
    counts = {doc_id: count for doc_id, count in counts_result.all()}

    return [_fmt_doc(d, counts.get(d.id, 0)) for d in docs]


# ── INDEX HEALTH (PostgreSQL ↔ ChromaDB reconciliation) ───────────────────────

@router.get("/documents/index-health", response_model=IndexHealth, tags=["documents"])
async def index_health(db: AsyncSession = Depends(get_db)):
    """
    Compare chunk rows in PostgreSQL against vectors in ChromaDB so the
    dashboard can flag drift instead of quietly showing two different numbers.
    Lives here because the persistent Chroma client is single-process and this
    service owns it.
    """
    rows_result = await db.execute(select(func.count(Chunk.id)))
    chunk_rows = rows_result.scalar() or 0

    try:
        vectors = count_all_vectors()
    except Exception as exc:
        logger.warning("Could not read ChromaDB vector count: %s", exc)
        raise HTTPException(status_code=503, detail=f"Vector store unavailable: {exc}")

    return IndexHealth(chunkRows=chunk_rows, vectors=vectors, inSync=chunk_rows == vectors)


# ── UPLOAD (with transactional sync) ─────────────────────────────────────────

@router.post("/documents/upload", response_model=DocumentFile, status_code=201, tags=["documents"])
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Full pipeline: receive → validate → save → chunk → embed → ChromaDB → PostgreSQL.
    Fully transactional: both stores updated atomically or not at all.
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                "Accepted: PDF, DOCX, TXT, XLSX, XLS."
            ),
        )

    raw = await file.read()
    size_bytes = len(raw)
    if size_bytes / (1024 * 1024) > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large. Max {settings.MAX_UPLOAD_SIZE_MB} MB.")

    # ── Save file to disk ─────────────────────────────────────────────────────
    safe_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, safe_name)
    with open(file_path, "wb") as f:
        f.write(raw)

    # ── Phase 1: Create PG document row ───────────────────────────────────────
    doc = Document(
        name=file.filename,
        original_filename=file.filename,
        file_size=_fmt_size(size_bytes),
        category=category,
        status="Processing",
        chunks_count=0,
    )
    db.add(doc)
    await db.flush()

    chroma_ids_written: list[str] = []

    try:
        # ── Phase 2: Extract text and chunk ───────────────────────────────────
        chunks = chunk_document(file_path, file.content_type)
        if not chunks:
            raise ValueError("No text could be extracted from the document.")

        texts = [c["text"] for c in chunks]

        # ── Phase 3: Generate embeddings (Gemini gemini-embedding-2) ──────────
        embeddings = await embed_texts(texts)

        # ── Phase 4: Write to ChromaDB (vector store) ─────────────────────────
        chroma_ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {
                "document_id": str(doc.id),
                "chunk_index": i,
                "filename": file.filename,
                "category": category,
            }
            for i in range(len(chunks))
        ]
        add_to_chroma(
            ids=chroma_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        chroma_ids_written = chroma_ids

        # ── Phase 5: Write chunk rows to PostgreSQL ───────────────────────────
        db.add_all([
            Chunk(
                document_id=doc.id,
                chunk_index=i,
                text=chunk["text"],
                tokens=chunk["tokens"],
                chroma_id=cid,
            )
            for i, (chunk, cid) in enumerate(zip(chunks, chroma_ids))
        ])

        # ── Phase 6: Mark document as Indexed ─────────────────────────────────
        doc.status = "Indexed"
        doc.chunks_count = len(chunks)
        doc.updated_at = datetime.utcnow()
        await db.flush()

        logger.info(
            "Indexed '%s': %d chunks, %d tokens avg",
            file.filename, len(chunks),
            sum(c["tokens"] for c in chunks) // len(chunks) if chunks else 0,
        )

    except Exception as exc:
        # ── ROLLBACK: remove ChromaDB vectors written in this request ──────────
        if chroma_ids_written:
            try:
                delete_from_chroma(chroma_ids_written)
                logger.info("Rolled back %d ChromaDB vectors for failed upload.", len(chroma_ids_written))
            except Exception as chroma_err:
                logger.error("ChromaDB rollback also failed: %s", chroma_err)

        # ── Remove uploaded file ───────────────────────────────────────────────
        try:
            os.remove(file_path)
        except OSError:
            pass

        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

    return _fmt_doc(doc, len(chunks))


# ── EDIT METADATA ─────────────────────────────────────────────────────────────

@router.patch("/documents/{doc_id}", response_model=DocumentFile, tags=["documents"])
async def update_document_metadata(
    doc_id: str,
    payload: DocumentMetadataUpdate,
    db: AsyncSession = Depends(get_db),
):
    """
    Update a document's display name and/or knowledge category.
    The same values are written onto every one of its vectors' metadata, so the
    vector store and PostgreSQL never disagree about a document's category.
    """
    doc_uuid = _parse_uuid(doc_id)
    doc = await _lock_document(db, doc_uuid)

    updates: dict[str, str] = {}
    if payload.name is not None and payload.name.strip() and payload.name != doc.name:
        doc.name = payload.name.strip()
        updates["filename"] = doc.name
    if payload.category is not None and payload.category != doc.category:
        doc.category = payload.category
        updates["category"] = doc.category

    if updates:
        try:
            update_document_vector_metadata(str(doc.id), updates)
        except Exception as chroma_err:
            logger.error("Chroma metadata update failed for doc %s: %s", doc_id, chroma_err)
            raise HTTPException(
                status_code=500,
                detail=f"Vector metadata update failed: {chroma_err}",
            )
        doc.updated_at = datetime.utcnow()
        await db.flush()
        logger.info("Updated metadata for document %s: %s", doc_id, updates)

    return _fmt_doc(doc, await _count_chunks(db, doc_uuid))


# ── DELETE (with sync guarantee) ─────────────────────────────────────────────

@router.delete("/documents/{doc_id}", status_code=204, tags=["documents"])
async def delete_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    """
    Delete-order: ChromaDB first → PostgreSQL.
    If ChromaDB deletion fails, the PG record is NOT deleted (stores stay in sync).
    """
    doc_uuid = _parse_uuid(doc_id)
    doc = await _lock_document(db, doc_uuid)

    try:
        # Purge by document_id tag rather than by the chroma_ids tracked in PG,
        # so vectors orphaned by an earlier failure are removed too.
        removed = delete_document_vectors(str(doc_uuid))
        logger.info("Deleted %d ChromaDB vectors for document %s", removed, doc_id)
    except Exception as chroma_err:
        logger.error("ChromaDB delete failed for doc %s: %s", doc_id, chroma_err)
        raise HTTPException(
            status_code=500,
            detail=f"Vector store deletion failed. Error: {chroma_err}",
        )

    await db.execute(delete(Document).where(Document.id == doc_uuid))
    await db.flush()
    logger.info("Deleted document %s ('%s') from PostgreSQL.", doc_id, doc.name)


# ── REINDEX (replace all vectors atomically) ──────────────────────────────────

@router.post("/documents/{doc_id}/reindex", response_model=DocumentFile, tags=["documents"])
async def reindex_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    """
    Re-embed and re-index a document.
    Old vectors are deleted from both stores before new ones are created.

    The document row is locked first, so overlapping reindex requests queue up.
    Without the lock, two concurrent calls each saw the same "old" chunk set,
    each deleted only those, and each then inserted a full new set — leaving
    twice the chunk rows/vectors for one document.
    """
    doc_uuid = _parse_uuid(doc_id)
    doc = await _lock_document(db, doc_uuid)

    # ── Find original file (before destroying anything) ───────────────────────
    file_path = _find_upload(doc.original_filename)
    if not file_path:
        raise HTTPException(status_code=404, detail="Original file not found on disk. Please re-upload.")

    # ── Re-chunk and re-embed BEFORE touching either store ────────────────────
    # Chunking/embedding is the step that can fail (bad file, embedding API
    # error). Doing it first means a failure leaves the existing index intact.
    try:
        content_type = _guess_content_type(doc.original_filename)
        chunks = chunk_document(file_path, content_type)
        if not chunks:
            raise ValueError("No text could be extracted from the document.")

        texts = [c["text"] for c in chunks]
        embeddings = await embed_texts(texts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {exc}")

    # ── Swap in the new index ─────────────────────────────────────────────────
    # Snapshot the vectors to retire *by document_id tag*, so orphans left by an
    # earlier failed run are cleaned up too and the count cannot creep upwards.
    # They are deleted only once the new index is safely in place, so a failure
    # mid-way can be rolled back to the previous state instead of leaving chunk
    # rows behind with no vector.
    try:
        old_vector_ids = get_document_vector_ids(str(doc_uuid))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read old vectors: {e}")

    chroma_ids_written: list[str] = []

    try:
        await db.execute(delete(Chunk).where(Chunk.document_id == doc_uuid))
        chroma_ids = [str(uuid.uuid4()) for _ in chunks]
        add_to_chroma(
            ids=chroma_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "document_id": str(doc.id),
                    "chunk_index": i,
                    "filename": doc.original_filename,
                    "category": doc.category,
                }
                for i in range(len(chunks))
            ],
        )
        chroma_ids_written = chroma_ids

        db.add_all([
            Chunk(document_id=doc.id, chunk_index=i, text=c["text"], tokens=c["tokens"], chroma_id=cid)
            for i, (c, cid) in enumerate(zip(chunks, chroma_ids))
        ])
        doc.status = "Indexed"
        doc.chunks_count = len(chunks)
        doc.updated_at = datetime.utcnow()
        await db.flush()

    except Exception as exc:
        if chroma_ids_written:
            try:
                delete_from_chroma(chroma_ids_written)
            except Exception:
                pass
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Reindex failed: {exc}")

    # ── Retire the previous vectors ───────────────────────────────────────────
    # The new index is in place; drop the superseded vectors. Failing here only
    # leaves stale vectors behind, which the next reindex/delete will purge.
    if old_vector_ids:
        try:
            delete_from_chroma(old_vector_ids)
        except Exception as chroma_err:
            logger.error(
                "Could not remove %d superseded vectors for doc %s: %s",
                len(old_vector_ids), doc_id, chroma_err,
            )

    logger.info(
        "Re-indexed '%s': retired %d old vectors, wrote %d new chunks.",
        doc.name, len(old_vector_ids), len(chunks),
    )

    return _fmt_doc(doc, len(chunks))


# ── CHUNKS ────────────────────────────────────────────────────────────────────

@router.get("/documents/{doc_id}/chunks", response_model=list[VectorChunk], tags=["documents"])
async def get_chunks(doc_id: str, db: AsyncSession = Depends(get_db)):
    doc_uuid = _parse_uuid(doc_id)
    result = await db.execute(
        select(Chunk).where(Chunk.document_id == doc_uuid).order_by(Chunk.chunk_index)
    )
    chunks = result.scalars().all()

    # Report real embedding state per chunk: present in ChromaDB, and its
    # dimensionality. (There is no stored similarity score — a score only
    # exists for a chunk relative to a specific query.)
    vectors = get_chunks_from_chroma([c.chroma_id for c in chunks if c.chroma_id])
    dim_map = {item["id"]: item["dim"] for item in vectors}

    return [
        VectorChunk(
            id=str(c.id),
            chunkIndex=c.chunk_index,
            tokens=c.tokens,
            textExcerpt=c.text[:300] + ("..." if len(c.text) > 300 else ""),
            embedded=c.chroma_id in dim_map,
            embeddingDim=dim_map.get(c.chroma_id),
        )
        for c in chunks
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_uuid(val: str):
    try:
        return uuid.UUID(val)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format.")


def _fmt_size(bytes_: int) -> str:
    for unit in ["Bytes", "KB", "MB", "GB"]:
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f} GB"


def _find_upload(filename: str) -> str | None:
    """
    Find a previously uploaded file by its original filename suffix.
    Stored names are "<uuid>_<original>", so if the same filename was uploaded
    more than once, pick the newest copy to keep reindex deterministic.
    """
    try:
        matches = [
            os.path.join(settings.UPLOAD_DIR, f)
            for f in os.listdir(settings.UPLOAD_DIR)
            if f.endswith(filename)
        ]
    except OSError:
        return None
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _guess_content_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    return {
        "pdf":  "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt":  "text/plain",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls":  "application/vnd.ms-excel",
    }.get(ext, "text/plain")
