"""
API Gateway — port 8000
Validates Auth0 JWTs and reverse-proxies to internal microservices.
"""
import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SERVICE_DIR.parent.parent
for p in (str(BACKEND_DIR), str(SERVICE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from contextlib import asynccontextmanager
import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from shared.config import settings
from shared.auth.jwt_validator import verify_token

_proxy_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _proxy_client
    _proxy_client = httpx.AsyncClient(
        timeout=300.0,
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
    )
    yield
    if _proxy_client:
        await _proxy_client.aclose()


app = FastAPI(
    title="CogniDoc API Gateway",
    description="Authenticated reverse proxy for all CogniDoc microservices.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Service URL map ────────────────────────────────────────────────────────────

SERVICE_MAP = {
    "/api/documents": settings.DOCUMENT_SERVICE_URL,
    "/api/chat":      settings.RAG_SERVICE_URL,
    "/api/metrics":   settings.METRICS_SERVICE_URL,
    "/api/judge":     settings.JUDGE_SERVICE_URL,
}


def _resolve_service(path: str) -> str:
    for prefix, url in SERVICE_MAP.items():
        if path.startswith(prefix):
            return url
    raise HTTPException(status_code=404, detail=f"No service registered for path: {path}")


# ── Proxy handler ─────────────────────────────────────────────────────────────

@app.api_route(
    "/api/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    dependencies=[Depends(verify_token)],
    tags=["proxy"],
    summary="Authenticated proxy to internal services",
)
async def proxy(request: Request, path: str):
    service_url = _resolve_service(f"/api/{path}")
    target_url = f"{service_url}/api/{path}"

    # Forward request (including multipart files) using pooled client.
    # IMPORTANT: preserve Content-Type so multipart boundary is not lost.
    client = _proxy_client or httpx.AsyncClient(timeout=300.0)
    try:
        body = await request.body()

        forward_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }

        upstream = await client.request(
            method=request.method,
            url=target_url,
            headers=forward_headers,
            params=dict(request.query_params),
            content=body,
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"Upstream service unavailable: {service_url}")

    # Exclude hop-by-hop & compression headers so Starlette formats the response correctly
    safe_response_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
    }

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=safe_response_headers,
        media_type=upstream.headers.get("content-type"),
    )


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "service": "gateway"}
