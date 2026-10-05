"""
LangGraph StateGraph builder and compiled application.

Graph shape:

                      ┌─────────┐
                      │ router  │
                      └────┬────┘
              greeting     │      retrieval
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
      (END)                            ┌─────────────┐
                                       │  retrieve   │◀────┐
                                       └──────┬──────┘     │
                                              ▼            │ retry (max 1)
                                       ┌─────────────┐     │
                                       │    grade    │─────┘
                                       └──────┬──────┘
                                              ▼ generate
                                       ┌─────────────┐
                                       │  generate   │  ← only LLM call
                                       └──────┬──────┘
                                              ▼
                                       ┌─────────────┐
                                       │   verify    │
                                       └──────┬──────┘
                                              ▼
                                            (END)

The only cycle is grade -> retrieve, and `should_retry` can only be True on the
first grading failure, so the cycle executes at most once. `recursion_limit`
is a second, independent backstop.
"""
import logging

from langgraph.graph import END, START, StateGraph

from services.agentic_rag_service.config import GRAPH_RECURSION_LIMIT
from services.agentic_rag_service.graph.nodes import (
    generate_node,
    grade_documents_node,
    retrieve_node,
    route_after_grading,
    route_after_router,
    router_node,
    verify_grounding_node,
)
from services.agentic_rag_service.graph.state import AgenticRAGState, initial_state

logger = logging.getLogger(__name__)


def build_workflow() -> StateGraph:
    """Assemble the uncompiled graph."""
    graph = StateGraph(AgenticRAGState)

    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_documents_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_grounding_node)

    graph.add_edge(START, "router")

    # Greetings never touch retrieval or generation — 0 API calls.
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"greeting": END, "retrieval": "retrieve"},
    )

    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        route_after_grading,
        {"retry": "retrieve", "generate": "generate"},
    )

    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)

    return graph


_compiled = None


def get_app():
    """Compiled graph singleton. Compiling is cheap but not free."""
    global _compiled
    if _compiled is None:
        _compiled = build_workflow().compile()
        logger.info("[graph] agentic RAG workflow compiled")
    return _compiled


async def run_workflow(
    question: str,
    category: str | None = None,
    document_id: str | None = None,
) -> AgenticRAGState:
    """
    Execute the graph for one question and return the final state.

    recursion_limit caps total supersteps as a hard stop independent of the
    should_retry logic, so a future edit that reintroduces a loop fails loudly
    instead of spinning on the Gemini quota.
    """
    app = get_app()
    result = await app.ainvoke(
        initial_state(question, category=category, document_id=document_id),
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
    )
    return result
