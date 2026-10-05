"""
Document Service — port 8001
Handles document upload, chunking, embedding, indexing, and CRUD.
"""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

SERVICE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SERVICE_DIR.parent.parent
for p in (str(BACKEND_DIR), str(SERVICE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.config import settings
from shared.db.postgres import create_tables
from shared.error_handler import register_error_handlers
from shared.logger import setup_logger
from shared.logging_middleware import RequestLoggingMiddleware
from api.routes import router

logger = setup_logger("document_service")

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Document Service database tables...")
    await create_tables()
    logger.info("Document Service ready.")
    yield


app = FastAPI(
    title="CogniDoc Document Service",
    description="Upload, chunk, embed, and manage enterprise documents.",
    version="1.0.0",
    lifespan=lifespan,
)

register_error_handlers(app)
app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "service": "document-service"}
