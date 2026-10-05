"""
FastAPI Request/Response Logging Middleware.
Logs incoming requests, status codes, latencies, and injects X-Request-ID headers.
"""
import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("cognidoc.http")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every HTTP request/response with client info, status code, and latency in milliseconds."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Don't clutter logs with periodic /health checks at INFO level
        is_health = request.url.path.endswith("/health")

        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        client_host = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path

        if not is_health:
            logger.info("[%s] --> %s %s (from %s)", request_id, method, path, client_host)

        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(
                "[%s] <-- %s %s - EXCEPTION after %.1fms: %s",
                request_id, method, path, duration_ms, exc
            )
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Health checks logged at debug level, normal endpoints at info
        log_fn = logger.debug if is_health else logger.info
        status_code = response.status_code

        # Log with warning level if 4xx/5xx error status
        if status_code >= 500:
            logger.error("[%s] <-- %s %s -> %d (%.1fms)", request_id, method, path, status_code, duration_ms)
        elif status_code >= 400:
            logger.warning("[%s] <-- %s %s -> %d (%.1fms)", request_id, method, path, status_code, duration_ms)
        else:
            log_fn("[%s] <-- %s %s -> %d (%.1fms)", request_id, method, path, status_code, duration_ms)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"
        return response
