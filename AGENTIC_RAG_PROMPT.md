# Specification & Master Prompt: Budget-Aware Agentic RAG Microservice (LangGraph + Gemini Free Tier)

## 1. Core Mandates & Architecture Guardrails

### A. Non-Negotiable Rules & Zero-Damage Isolation
1. **Completely Separate Microservice**:
   - Must reside strictly in `backend/services/agentic_rag_service/` running on **port 8005**.
   - The existing `rag_service` on port 8002 must remain **100% untouched, fully functional, and runnable as a baseline fallback**.
   - Do NOT modify or break any existing endpoints (`/api/chat/*`, `/api/documents/*`, `/api/judge/*`, `/api/metrics/*`).
2. **Gemini Free Tier Rate Limit (15 RPM strictly enforced)**:
   - Cap Gemini LLM calls to **strictly <= 1 call per query** (and **0 calls for greetings/casual chit-chat**).
   - All routing, document relevance grading, and citation verification must execute locally via Python heuristics, vector math, and regex—never via extra LLM calls.
   - Implement an in-memory Sliding Window / Token Bucket Rate Limiter capped at 12 RPM to prevent HTTP 429 `ResourceExhausted` errors.
   - Graph `recursion_limit` must be set to prevent infinite loops (max 1 retry iteration).
3. **Strict Angular 22 & Styling Rules**:
   - **Angular 22 Zoneless Change Detection**: Strictly use Angular Signals (`signal()`, `computed()`) for all async reactive state updates. Never rely on Zone.js or raw property assignments inside async callbacks.
   - **Preserve Existing Design System**: Must strictly match the current UI/UX, typography, dark/light theme tokens, borders, glassmorphism card styling, responsive layouts, and existing chat bubble designs. No generic or conflicting styles.
4. **Evaluation Metrics Integrity (TP, TN, FP, FN, F1, Precision, Recall)**:
   - Queries must be logged to `query_logs` with `agent_mode = 'agentic'` and an `intent` tag (`'greeting'` vs `'retrieval'`).
   - Non-retrieval greetings (0 chunks) must be filtered out of document retrieval metrics to prevent diluting the F1 score.

---

## 2. Package Management & Dependencies (Explicit Updates)

### A. Backend Dependencies (`backend/requirements.txt`)
Add the following required packages to `backend/requirements.txt` without bumping or breaking existing pinned packages:
```text
langgraph>=0.2.0
langchain>=0.2.0
langchain-core>=0.2.0
langchain-google-genai>=1.0.0
```
*Note: Ensure compatibility with existing dependencies (`fastapi`, `uvicorn`, `chromadb`, `google-generativeai`).*

### B. Frontend Dependencies (`frontend/package.json`)
- Maintain full compatibility with Angular 22 (`@angular/core`: `^22.x`).
- Ensure all Lucide icons used for the toggle switch (e.g. `Zap`, `Brain`, `Bot`, `Sparkles`) exist in `lucide-angular` in `frontend/package.json`. If missing, add them or use existing SVG iconography matching the project.

---

## 3. Directory Structure to Create
```
backend/
└── services/
    └── agentic_rag_service/
        ├── __init__.py
        ├── main.py                     # FastAPI application on port 8005
        ├── config.py                   # Service-specific configs & rate limits
        ├── rate_limiter.py             # Sliding window 12 RPM token limiter
        ├── cache.py                    # In-memory query response cache (TTL 5 mins)
        ├── graph/
        │   ├── __init__.py
        │   ├── state.py                # TypedDict Graph State
        │   ├── nodes.py                # Router, Retriever, Grader, Generator, Verifier
        │   └── workflow.py             # LangGraph StateGraph builder & compiled app
        └── api/
            ├── __init__.py
            └── routes.py               # POST /api/agent/query, GET /api/agent/health
```

---

## 4. Detailed Component Specifications

### A. Graph State Definition (`graph/state.py`)
```python
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class AgenticRAGState(TypedDict):
    question: str
    cleaned_query: str
    intent: str                      # 'greeting' | 'retrieval' | 'out_of_scope'
    retrieved_docs: List[Dict[str, Any]]
    is_relevant: bool
    retry_count: int
    answer: str
    citations: List[Dict[str, Any]]
    confidence_score: float
    grounded: bool
    api_calls_used: int
```

### B. Graph Nodes & 15-RPM Optimization Logic (`graph/nodes.py`)

1. `router_node` (0 API Calls / < 1ms):
   - Fast regex and keyword patterns:
     `r"^(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|ok|cool|bye|who are you|help)\b"`
   - If matched: Set `intent = 'greeting'`, return instant canned response, route directly to `END`.
   - If not matched: Set `intent = 'retrieval'`, route to `retrieve_node`.

2. `retrieve_node` (0 API Calls):
   - Reuses existing `backend/shared/db/chroma.py` and `BM25` hybrid retriever.
   - Fetches top-K chunks (K=4) from shared indices.

3. `grade_documents_node` (0 API Calls - Local Math Only):
   - Evaluates ChromaDB cosine distance (`similarity = 1 - distance`) and BM25 score.
   - If `max(similarity) >= 0.55` or `top_bm25 >= 8.0`: `is_relevant = True`.
   - If both below threshold: `is_relevant = False`.
   - **Loop Guard**: If `is_relevant == False` and `retry_count == 0`:
     - Increment `retry_count = 1`.
     - Perform local keyword extraction/stemming to broaden search term and re-route to `retrieve_node`.
     - If `retry_count >= 1`: Route to `generate_node` with instruction to output *"Information not found in available documents."* (prevents infinite loop).

4. `generate_node` (Exactly 1 Gemini API Call):
   - Uses Gemini 1.5 Flash via `RateLimiter` guard.
   - Single system-prompted Chain-of-Thought request:
     - Constrains answer strictly to retrieved chunks.
     - Instructs model to cite exact document name and page number for each claim.
     - Instructs model to state clearly if the answer is not supported by context.

5. `verify_grounding_node` (0 API Calls - Local Python Verification):
   - Verifies citations locally: checks if quoted chunks and cited document names actually exist in `retrieved_docs`.
   - Flags answer as `grounded = True` if citations match context; marks `grounded = False` if citations reference missing documents.

---

### C. Rate Limiter (`rate_limiter.py`)
- Thread-safe / async sliding window token limiter:
  - Max 12 requests per 60-second rolling window.
  - If a 13th request arrives within the window, `await asyncio.sleep()` until a slot frees up, or return a clean, user-friendly 429 message instead of crashing.

---

### D. Database Schema & Logging
- Reuse PostgreSQL table `query_logs`.
- Ensure columns exist:
  - `agent_mode` VARCHAR(20) DEFAULT `'standard'` (`'standard'` | `'agentic'`)
  - `intent` VARCHAR(30) DEFAULT `'retrieval'` (`'greeting'` | `'retrieval'`)
- Ensure that `metrics_service` filters out rows where `intent == 'greeting'` when computing TP, TN, FP, FN, Precision, Recall, and F1.

---

### E. Gateway & Process Orchestration Updates

1. `backend/services/gateway/main.py`:
   - Add upstream destination: `AGENTIC_RAG_SERVICE_URL = "http://localhost:8005"`
   - Add proxy route:
     ```python
     @app.api_route("/api/agent/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
     async def proxy_agent(request: Request, path: str):
         # Route to port 8005 using persistent httpx client pool
     ```

2. `backend/run_all.py`:
   - Add 5th process to `SERVICES` list:
     ```python
     ("Agentic RAG Service", 8005, "services.agentic_rag_service.main:app")
     ```
   - Keep Windows-safe process-tree termination with `taskkill /F /T /PID`.

---

### F. Frontend Chat Integration (Angular 22 Zoneless)

1. **Model & Service Updates**:
   - In `chat.model.ts`: Add `agentMode?: 'standard' | 'agentic'`.
   - In `rag.service.ts`: Add `queryAgent(question: string, documentId?: string)` calling `/api/agent/query`.
2. **UI Component (`home.html` / `home.ts`)**:
   - Add a subtle mode toggle switch in the chat header or above the input bar:
     - `Standard RAG ⚡` (calls `/api/chat/query` on port 8002)
     - `Agentic RAG 🧠` (calls `/api/agent/query` on port 8005)
   - Use an Angular signal: `activeMode = signal<'standard' | 'agentic'>('standard')`.
   - Retain identical styling, message bubble layout, citations display, and 👍 / 👎 feedback buttons.

---

## 5. Verification & Testing Checklist
- [ ] Dependencies: `pip install -r backend/requirements.txt` installs LangGraph and LangChain cleanly without version conflicts.
- [ ] Port Isolation: Port 8005 starts cleanly without conflicts. All existing services (8000–8004) continue working with zero regression.
- [ ] Greeting Detection (0 Calls): Typing "Hello", "Good morning", or "Thanks" yields an instant reply with 0 Gemini calls recorded.
- [ ] Information Retrieval (1 Call): Complex PDF questions trigger ChromaDB retrieval and exactly 1 Gemini API call.
- [ ] 15 RPM Safety: Rapidly firing 5 queries in succession queues cleanly without encountering 429 Too Many Requests.
- [ ] Feedback & Metrics: 👍 and 👎 submissions from Agentic mode persist to `query_logs` and reflect accurately on the admin dashboard under TP, TN, FP, FN.
- [ ] UI Consistency: Chat interface maintains exact colors, fonts, margins, and dark/light theme consistency.
- [ ] Windows Clean Teardown: Pressing CTRL+C on `run_all.py` cleanly kills all 5 microservices without leaving orphaned background processes on ports 8001–8005.
