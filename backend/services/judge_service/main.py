"""
Judge Service — port 8003
Uses Groq LLM to evaluate RAG answer quality (faithfulness, relevance, completeness).
Called asynchronously by the RAG service — never blocks the chat response.
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
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    yield


app = FastAPI(
    title="CogniDoc Judge Service",
    description="LLM-as-Judge using Groq — evaluates RAG answer quality.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "service": "judge-service"}
