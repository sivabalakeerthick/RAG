"""Agentic RAG Service (port 8005) — LangGraph orchestrated, budget-aware RAG.

Fully isolated from the baseline rag_service on port 8002, which remains the
untouched fallback. Shares only the read-only retrieval stores (ChromaDB +
PostgreSQL chunks) and the query_logs table.
"""
