"""
CogniDoc Multi-Service Runner (Local Development)
Runs all 6 microservices in a single console session from the backend root.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

# Fix Windows cp1252 encoding for unicode/emojis in terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BACKEND_DIR = Path(__file__).resolve().parent

SERVICES = [
    ("Document Service",    "services.document_service.main:app",    8001),
    ("RAG Service",         "services.rag_service.main:app",         8002),
    ("Judge Service",       "services.judge_service.main:app",       8003),
    ("Metrics Service",     "services.metrics_service.main:app",     8004),
    ("Agentic RAG Service", "services.agentic_rag_service.main:app", 8005),
    # Gateway last: it proxies to all of the above, so it should come up once
    # its upstreams are already listening.
    ("API Gateway",         "services.gateway.main:app",             8000),
]

def _check_interpreter():
    """
    Fail fast with one clear message if started by the wrong interpreter.

    Children are spawned with sys.executable, so whichever Python runs this
    file is the one every service inherits. Launching with the global Python
    instead of the venv produced six identical ModuleNotFoundError tracebacks
    with no indication of the actual cause, so the check happens up front.
    """
    venv_python = BACKEND_DIR / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = BACKEND_DIR / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists():  # POSIX layout
        venv_python = BACKEND_DIR / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = BACKEND_DIR / "venv" / "bin" / "python"

    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        print("=" * 68)
        print("[X] Wrong Python interpreter - dependencies are not installed here.")
        print("=" * 68)
        print("    missing    : %s" % exc.name)
        print("    running in : %s" % sys.executable)
        print()
        print("    All 6 services are launched with this same interpreter, so")
        print("    they would each fail the same way.")
        print()
        if venv_python.exists():
            print("    Run it with the project venv instead:")
            print()
            print("        %s run_all.py" % venv_python)
            print()
            print("    ...or activate the venv first, so the prompt shows (venv):")
            print()
            print("        .\\venv\\Scripts\\Activate.ps1")
            print("        python run_all.py")
        else:
            print("    No venv found at %s" % (BACKEND_DIR / "venv"))
            print("    Create one and install dependencies:")
            print()
            print("        python -m venv venv")
            print("        .\\venv\\Scripts\\python.exe -m pip install -r requirements.txt")
        print("=" * 68)
        sys.exit(1)


def main():
    _check_interpreter()

    print("=" * 60)
    print(">> Starting CogniDoc Backend Microservices...")
    print("=" * 60)
    print("   interpreter: %s" % sys.executable)

    processes = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_DIR)
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        for name, app_module, port in SERVICES:
            cmd = [
                sys.executable, "-m", "uvicorn", app_module,
                "--host", "127.0.0.1",
                "--port", str(port),
                "--reload"
            ]
            print(f"[*] Launching {name} on http://localhost:{port} ({app_module})...")
            p = subprocess.Popen(cmd, cwd=str(BACKEND_DIR), env=env)
            processes.append((name, p))
            time.sleep(1)  # Stagger startups slightly

        print("\n" + "=" * 60)
        print("[OK] All services are active!")
        print("   - API Gateway:      http://localhost:8000 (Swagger: /docs)")
        print("   - Document Service: http://localhost:8001 (Swagger: /docs)")
        print("   - RAG Service:      http://localhost:8002 (Swagger: /docs)")
        print("   - Judge Service:    http://localhost:8003 (Swagger: /docs)")
        print("   - Metrics Service:  http://localhost:8004 (Swagger: /docs)")
        print("   - Agentic RAG:      http://localhost:8005 (Swagger: /docs)")
        print("Press Ctrl+C to terminate all services.")
        print("=" * 60 + "\n")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[!] Stopping all services...")
        for name, p in processes:
            print(f"Terminating {name}...")
            if sys.platform == "win32":
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(p.pid)],
                        capture_output=True,
                        check=False,
                    )
                except Exception:
                    p.terminate()
            else:
                p.terminate()
        for _, p in processes:
            try:
                p.wait(timeout=3)
            except Exception:
                pass
        print("All services stopped.")

if __name__ == "__main__":
    main()
