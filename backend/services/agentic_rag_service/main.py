"""
Agentic RAG Service — port 8005.

LangGraph-orchestrated, budget-aware RAG. Completely separate from the baseline
rag_service on port 8002, which stays untouched as the fallback.

Per query: at most 1 Gemini generation call (0 for greetings and cache hits).
Routing, relevance grading and citation verification run locally.
"""
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

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
from services.agentic_rag_service.api.routes import router
from services.agentic_rag_service.graph.workflow import get_app as get_graph_app

logger = setup_logger("agentic_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs the same idempotent migrations as every other service, so whichever
    # service boots first adds the agent_mode/intent columns.
    logger.info("Initializing Agentic RAG Service database tables...")
    await create_tables()
    # Compile the graph at startup rather than on the first request, so the
    # first user query is not slowed by graph construction.
    get_graph_app()
    logger.info("Agentic RAG Service ready on port 8005.")
    yield


app = FastAPI(
    title="CogniDoc Agentic RAG Service",
    description=(
        "LangGraph agentic RAG — local intent routing, hybrid retrieval, local "
        "relevance grading with one bounded retry, a single Gemini generation "
        "call, and local citation verification. Free-tier safe at 12 RPM."
    ),
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
    return {"status": "ok", "service": "agentic-rag-service"}
