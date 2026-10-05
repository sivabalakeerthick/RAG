"""
Global standardized error handler for CogniDoc microservices and gateway.
Provides consistent JSON error payloads across all endpoints:
{
    "error": True,
    "status_code": 404,
    "message": "Document not found",
    "detail": "...",
    "path": "/api/documents/123",
    "timestamp": "2026-10-03T02:30:00Z"
}
"""
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("cognidoc.error_handler")


def _format_error_response(
    status_code: int,
    message: str,
    detail: any,
    path: str,
    extra: dict | None = None,
) -> JSONResponse:
    payload = {
        "error": True,
        "status_code": status_code,
        "message": message,
        "detail": detail,
        "path": path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    return JSONResponse(status_code=status_code, content=payload)


def _clean_user_message(detail: any, status_code: int) -> str:
    """Translates developer exceptions and RPC payloads into polite, human-readable user messages."""
    raw = str(detail or "").strip()

    # Quota / Rate Limits
    if "RESOURCE_EXHAUSTED" in raw or "429" in raw or "quota" in raw.lower():
        return "AI service request limit reached. Please wait about 30–45 seconds before trying again."

    # Database / Storage connection issues
    if any(k in raw.lower() for k in ["operationalerror", "psycopg2", "chromadb", "sqlite3", "connection to server", "connection refused"]):
        return "The database service is temporarily busy or reconnecting. Please retry shortly."

    # File extraction / chunking issues
    if any(k in raw.lower() for k in ["badzipfile", "pdfsyntaxerror", "filedataerror", "no text could be extracted"]):
        return "Unable to extract readable text from this file. Please make sure the document is not corrupted or password-protected."

    # Python stack traces / internal exceptions / raw JSON payloads
    if any(k in raw for k in ["Traceback", "{'error'", '{"error"', "NoneType", "KeyError", "AttributeError", "IndexError", "ValueError"]):
        if status_code == 404:
            return "The requested document or resource could not be found."
        if status_code == 400:
            return "The request was invalid or contained unsupported parameters."
        return "An internal server error occurred while processing your request. Please try again."

    # If it is clean, relatively short, and doesn't contain raw code artifacts, keep it
    if len(raw) <= 180 and not any(ch in raw for ch in ["{", "}", "::", "Traceback", "line "]):
        return raw

    # Fallbacks by status code
    if status_code == 400:
        return "Invalid request. Please verify the input and try again."
    if status_code == 401:
        return "Your session has expired or requires authorization."
    if status_code == 403:
        return "You do not have permission to perform this action."
    if status_code == 404:
        return "The requested resource could not be found."
    if status_code == 422:
        return "Input validation failed. Please check the supplied fields."
    if status_code == 429:
        return "Service rate limit reached. Please wait a moment before retrying."

    return "An unexpected server error occurred. Please try again."


def register_error_handlers(app: FastAPI) -> None:
    """Register uniform global exception handlers on a FastAPI instance."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning(
            "HTTP %d on %s: %s", exc.status_code, request.url.path, exc.detail
        )
        user_msg = _clean_user_message(exc.detail, exc.status_code)
        return _format_error_response(
            status_code=exc.status_code,
            message=user_msg,
            detail=exc.detail,
            path=request.url.path,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
        return _format_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="Please check the information you entered and ensure all required fields are filled correctly.",
            detail=exc.errors(),
            path=request.url.path,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        # Handle RateLimitExceeded dynamically if raised
        if exc.__class__.__name__ == "RateLimitExceeded":
            retry_after = getattr(exc, "retry_after", 30.0)
            logger.warning("Rate limit hit on %s (retry_after=%.1fs)", request.url.path, retry_after)
            return _format_error_response(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                message="Gemini API rate limit reached. Please wait about 30 seconds before retrying.",
                detail={"retry_after": retry_after},
                path=request.url.path,
                extra={"retry_after": retry_after},
            )

        logger.exception("Unhandled 500 error on %s: %s", request.url.path, exc)
        user_msg = _clean_user_message(str(exc), 500)
        return _format_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=user_msg,
            detail=str(exc),
            path=request.url.path,
        )
