"""
CogniDoc Multi-Service Runner (Local Development)
Runs all 5 microservices in a single console session from the backend root.
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
    ("Document Service", "services.document_service.main:app", 8001),
    ("RAG Service",      "services.rag_service.main:app",      8002),
    ("Judge Service",    "services.judge_service.main:app",    8003),
    ("Metrics Service",  "services.metrics_service.main:app",  8004),
    ("API Gateway",      "services.gateway.main:app",          8000),
]

def main():
    print("=" * 60)
    print(">> Starting CogniDoc Backend Microservices...")
    print("=" * 60)

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
