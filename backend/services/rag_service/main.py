"""
RAG Service — port 8002
Hybrid retrieval (dense + BM25) → reranking → Gemini generation.
"""
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

from shared.db.postgres import create_tables
from shared.error_handler import register_error_handlers
from shared.logger import setup_logger
from shared.logging_middleware import RequestLoggingMiddleware
from api.routes import router

logger = setup_logger("rag_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing RAG Service database tables...")
    await create_tables()
    logger.info("RAG Service ready.")
    yield


app = FastAPI(
    title="CogniDoc RAG Service",
    description="Hybrid RAG: dense + BM25 retrieval, Gemini reranking and generation.",
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
    return {"status": "ok", "service": "rag-service"}
