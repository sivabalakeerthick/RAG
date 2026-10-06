"""
LangGraph StateGraph builder for True Agentic RAG.

Graph shape:

              ┌───────────────────┐
              │  guard_and_route  │
              └─────────┬─────────┘
        greeting/blocked│         │ substantive / compound
        ┌───────────────┘         ▼
        ▼                      ┌─────┐
      (END)                    │agent│◀────────────┐
                               └──┬──┘             │
                    needs data    │    no tools    │
                   ┌──────────────┴───────┐        │
                   ▼                      ▼        │
               ┌───────┐             ┌────────┐    │
               │ tools │────────────►│verifier│    │
               └───────┘             └───┬────┘    │
                   │                     │         │
                   └─────────────────────┘─────────┘
                                         ▼
                                       (END)

Max 2 tool execution iterations enforced by route_after_agent.
"""
import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from services.agentic_rag_service.config import GRAPH_RECURSION_LIMIT
from services.agentic_rag_service.graph.nodes import (
    agent_node,
    guard_and_route_node,
    route_after_agent,
    route_after_guard,
    tool_execution_node,
    verifier_node,
)
from services.agentic_rag_service.graph.state import AgenticRAGState, initial_state
from shared.query_rewriter import rewrite_query_for_search

logger = logging.getLogger(__name__)


def build_workflow() -> StateGraph:
    """Assemble the autonomous tool-calling graph."""
    graph = StateGraph(AgenticRAGState)

    graph.add_node("guard_and_route", guard_and_route_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_execution_node)
    graph.add_node("verifier", verifier_node)

    graph.add_edge(START, "guard_and_route")

    # Tier-1 fast-path: pure greetings / blocked injections exit immediately
    graph.add_conditional_edges(
        "guard_and_route",
        route_after_guard,
        {"end": END, "agent": "agent"},
    )

    # Agent either emits tool calls or produces final answer
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "verifier": "verifier"},
    )

    # Tool observations loop back into the agent
    graph.add_edge("tools", "agent")

    # Verifier finalizes and exits
    graph.add_edge("verifier", END)

    return graph


_compiled = None


def get_app():
    """Compiled graph singleton."""
    global _compiled
    if _compiled is None:
        _compiled = build_workflow().compile()
        logger.info("[graph] true agentic RAG workflow compiled")
    return _compiled


async def run_workflow(
    question: str,
    category: Optional[str] = None,
    document_id: Optional[str] = None,
    session_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> AgenticRAGState:
    """
    Execute the tool-calling graph for one question with session context.
    """
    app = get_app()

    rewritten_query = question
    initial_api_calls = 0
    if history and len(history) > 0:
        rewritten_query = await rewrite_query_for_search(question, history=history)
        if rewritten_query.strip().lower() != question.strip().lower():
            initial_api_calls = 1

    state = initial_state(
        question=question,
        category=category,
        document_id=document_id,
        session_id=session_id,
        history=history,
        rewritten_query=rewritten_query,
        initial_api_calls=initial_api_calls,
    )
    result = await app.ainvoke(
        state,
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
    )
    return result
