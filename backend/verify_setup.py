#!/usr/bin/env python
"""
Comprehensive setup verification script for CogniDoc.
Checks database, services, and configuration before starting the app.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import text

from shared.config import settings
from shared.db.postgres import engine, create_tables
from shared.db.chroma import get_chroma_client
from services.document_service.core.embedder import get_embeddings_model


class Color:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_status(msg: str, ok: bool = True):
    symbol = f"{Color.GREEN}✓{Color.RESET}" if ok else f"{Color.RED}✗{Color.RESET}"
    print(f"  {symbol} {msg}")


async def verify_database():
    print(f"\n{Color.BOLD}Testing PostgreSQL Connection...{Color.RESET}")
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT version();"))
            version = result.scalar()
            print_status("PostgreSQL connected", True)
            print(f"     Version: {version.split(',')[0]}")
            await create_tables()
            print_status("Tables created/verified", True)
            return True
    except Exception as e:
        print_status(f"PostgreSQL failed: {e}", False)
        return False


def verify_chroma():
    print(f"\n{Color.BOLD}Testing ChromaDB Setup...{Color.RESET}")
    try:
        client = get_chroma_client()
        print_status("ChromaDB client initialized", True)

        persist_path = settings.CHROMA_PERSIST_PATH
        if os.path.exists(persist_path):
            print_status(f"ChromaDB persistent data found at: {persist_path}", True)
            size_mb = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(persist_path)
                for filename in filenames
            ) / (1024 * 1024)
            print(f"     Data size: {size_mb:.2f} MB")
        else:
            print_status(f"ChromaDB path will be created at: {persist_path}", True)
        return True
    except Exception as e:
        print_status(f"ChromaDB verification failed: {e}", False)
        return False


def verify_embeddings():
    print(f"\n{Color.BOLD}Testing Embeddings Model...{Color.RESET}")
    try:
        if not settings.GEMINI_API_KEY:
            print_status("GEMINI_API_KEY not set", False)
            print(f"     {Color.YELLOW}→ Set GEMINI_API_KEY in .env to use embeddings{Color.RESET}")
            return False

        model = get_embeddings_model()
        print_status("Gemini embedding model loaded", True)
        print(f"     Model: {settings.GEMINI_EMBEDDING_MODEL}")
        return True
    except Exception as e:
        print_status(f"Embedding model failed: {e}", False)
        print(f"     {Color.YELLOW}→ Check GEMINI_API_KEY in .env{Color.RESET}")
        return False


def verify_directories():
    print(f"\n{Color.BOLD}Checking Required Directories...{Color.RESET}")
    dirs_ok = True

    upload_dir = settings.UPLOAD_DIR
    if os.path.exists(upload_dir):
        print_status(f"Upload directory exists: {upload_dir}", True)
    else:
        try:
            os.makedirs(upload_dir, exist_ok=True)
            print_status(f"Created upload directory: {upload_dir}", True)
        except Exception as e:
            print_status(f"Failed to create upload dir: {e}", False)
            dirs_ok = False

    return dirs_ok


def verify_configuration():
    print(f"\n{Color.BOLD}Configuration Check...{Color.RESET}")

    checks = [
        ("Database URL", settings.DATABASE_URL, "postgresql://"),
        ("Chroma Path", settings.CHROMA_PERSIST_PATH, "/"),
        ("Upload Dir", settings.UPLOAD_DIR, "/"),
        ("Chunk Size", settings.CHUNK_SIZE, 0),
        ("Max Upload (MB)", settings.MAX_UPLOAD_SIZE_MB, 0),
        ("Auth0 Domain", settings.AUTH0_DOMAIN, ""),
    ]

    config_ok = True
    for name, value, valid_indicator in checks:
        if isinstance(valid_indicator, str):
            ok = valid_indicator in str(value) if value else False
        else:
            ok = bool(value and value > valid_indicator)

        if ok:
            print_status(f"{name}: {Color.BLUE}{value}{Color.RESET}", True)
        else:
            print_status(f"{name}: {Color.YELLOW}Not configured{Color.RESET}", False)
            if not value and name not in ["Auth0 Domain"]:
                config_ok = False

    return config_ok


async def main():
    print(f"\n{Color.BOLD}╔══════════════════════════════════════╗{Color.RESET}")
    print(f"{Color.BOLD}║   CogniDoc Setup Verification        ║{Color.RESET}")
    print(f"{Color.BOLD}╚══════════════════════════════════════╝{Color.RESET}")

    results = {
        "Configuration": verify_configuration(),
        "Upload Directory": verify_directories(),
        "PostgreSQL": await verify_database(),
        "ChromaDB": verify_chroma(),
        "Embeddings": verify_embeddings(),
    }

    print(f"\n{Color.BOLD}Summary{Color.RESET}")
    print("─" * 40)

    all_ok = True
    for check, ok in results.items():
        symbol = f"{Color.GREEN}✓{Color.RESET}" if ok else f"{Color.YELLOW}⚠{Color.RESET}"
        print(f"  {symbol} {check}")
        if not ok:
            all_ok = False

    print("─" * 40)

    if all_ok:
        print(f"\n{Color.GREEN}{Color.BOLD}✓ All systems ready!{Color.RESET}")
        print(f"\nYou can now start the backend with:")
        print(f"  {Color.BLUE}python run_all.py{Color.RESET}\n")
        return 0
    else:
        print(f"\n{Color.YELLOW}{Color.BOLD}⚠ Some checks failed{Color.RESET}")
        print(f"\nPlease fix the issues above and try again.\n")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
