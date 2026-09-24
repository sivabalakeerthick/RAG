"""
TLS settings for outbound AI-provider calls.

Corporate networks that inspect HTTPS re-sign traffic with a private root CA
that Python's certifi bundle does not trust, so provider SDKs fail the
handshake with CERTIFICATE_VERIFY_FAILED. Setting SSL_VERIFY=false skips
verification for these calls. Keep it enabled anywhere the network is not
already inspecting the traffic.
"""
import ssl
from functools import lru_cache

import httpx

from shared.config import settings


@lru_cache
def _unverified_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def google_client_args() -> dict | None:
    """
    `client_args` for langchain_google_genai models, or None to keep defaults.

    google-genai sends sync calls over httpx (which names the option "verify")
    and async calls over aiohttp (which names it "ssl"), and langchain passes
    the same dict to both, so both keys are set; each transport discards the
    key it does not recognise.

    This has to be a context object rather than False. google-genai checks
    `if not args.get(...)`, so a falsy value reads as "unset" and gets replaced
    by the default certifi context, silently re-enabling verification.
    """
    if settings.SSL_VERIFY:
        return None
    ctx = _unverified_context()
    return {"verify": ctx, "ssl": ctx}


@lru_cache
def groq_client_args() -> dict:
    """`http_client`/`http_async_client` kwargs for ChatGroq, or empty."""
    if settings.SSL_VERIFY:
        return {}
    return {
        "http_client": httpx.Client(verify=False),
        "http_async_client": httpx.AsyncClient(verify=False),
    }
