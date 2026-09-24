# CogniDoc Backend (Python Microservices)

Enterprise RAG backend — 5 FastAPI microservices with hybrid retrieval, LLM-as-Judge, PostgreSQL, and ChromaDB.

## Architecture

```
Frontend (Angular :4200)
    ↓ Bearer JWT
Gateway (:8000) ── JWT validation (Auth0 JWKS)
    ├──► Document Service (:8001)  PDF/DOCX/TXT/XLSX/XLS → chunk → embed → ChromaDB + PG
    ├──► RAG Service (:8002)       Dense+BM25 → RRF → Gemini Rerank → Gemini generation
    │        └──► Judge Service (:8003)  [async] Groq evaluation (openai/gpt-oss-120b)
    └──► Metrics Service (:8004)   PostgreSQL aggregation
```

---

## Quick Start (No Docker)

### 1. Prerequisites
- **Python 3.11+**
- **PostgreSQL** installed and running on port 5432
- **Node.js & Angular CLI** (for frontend)

### 2. Configure Environment Variables
Inside `s:\POC-siva\backend`:
1. Check `.env` (already copied from `.env.example`).
2. Verify your API keys:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   GROQ_API_KEY=your_groq_api_key
   DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5432/cognidoc
   ```

### 3. Create the Database in PostgreSQL
Open **SQL Shell (psql)** or **pgAdmin** and run:
```sql
CREATE DATABASE cognidoc;
```
*(All tables are automatically created on first boot by SQLAlchemy).*

### 4. Run the Backend
Activate the virtual environment and start all 5 microservices with one command:
```powershell
cd s:\POC-siva\backend
.\.venv\Scripts\Activate.ps1
python run_all.py
```

### 5. Run the Frontend
In a separate terminal:
```powershell
cd s:\POC-siva\frontend
ng serve
```
Open your browser at **http://localhost:4200**.

---

## Swagger UI (Interactive API Docs)

Once running, you can test each service directly in your browser:

| Service | Swagger URL |
|---|---|
| API Gateway | http://localhost:8000/docs |
| Document Service | http://localhost:8001/docs |
| RAG Service | http://localhost:8002/docs |
| Judge Service | http://localhost:8003/docs |
| Metrics Service | http://localhost:8004/docs |

---

## Postman Collection
Import `postman/CogniDoc_API.postman_collection.json` into Postman to test all endpoints.
