"""
Judge Service API route: POST /api/judge/evaluate

Evaluates a RAG answer using LangChain ChatGroq (openai/gpt-oss-120b):
  - Faithfulness:   Is every claim grounded in the retrieved context?
  - Relevance:      Does the answer address the user's question?
  - Completeness:   Does the answer cover all aspects of the question?

Scores are parsed via LangChain JsonOutputParser and written back to query_logs in PostgreSQL.
"""
from fastapi import APIRouter, Depends, HTTPException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import settings
from shared.db.postgres import QueryLog, get_db
from shared.models.chat import JudgeEvalRequest, JudgeScore
from shared.ssl_config import groq_client_args
from shared.ssl_config import google_client_args
from langchain_google_genai import ChatGoogleGenerativeAI


router = APIRouter()

_parser = JsonOutputParser(pydantic_object=JudgeScore)

_JUDGE_PROMPT = PromptTemplate(
    template=(
        "You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.\n"
        "Evaluate the AI-generated answer based on the question and retrieved context.\n\n"
        "QUESTION:\n{question}\n\n"
        "RETRIEVED CONTEXT:\n{context}\n\n"
        "AI ANSWER:\n{answer}\n\n"
        "Rate the answer on each criterion from 0 to 10:\n"
        "- faithfulness: Every factual claim in the answer is supported by the context (10 = fully grounded)\n"
        "- relevance: The answer directly addresses the question (10 = perfectly on-topic)\n"
        "- completeness: The answer covers all aspects of the question (10 = nothing important missing)\n\n"
        "Verdict rule: 'PASS' if average score >= 7, otherwise 'FAIL'.\n\n"
        "{format_instructions}\n"
    ),
    input_variables=["question", "context", "answer"],
    partial_variables={"format_instructions": _parser.get_format_instructions()},
)


def _get_judge_chain():
    # llm = ChatGroq(
    #     model=settings.GROQ_MODEL,
    #     api_key=settings.GROQ_API_KEY,
    #     temperature=0.0,
    #     max_tokens=256,
    #     **groq_client_args(),
    # )
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_GENERATION_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.0,
        max_output_tokens=1024,
        client_args=google_client_args(),
    )
    return _JUDGE_PROMPT | llm | _parser


@router.post("/judge/evaluate", response_model=JudgeScore, tags=["judge"])
async def evaluate(req: JudgeEvalRequest, db: AsyncSession = Depends(get_db)):
    """
    Evaluate the RAG answer via LangChain ChatGoogleGenerativeAI chain and write scores back to query_logs.
    Called asynchronously by the RAG service background task.
    """
    import uuid

    # Validate query_log_id is a valid UUID
    try:
        query_log_uuid = uuid.UUID(req.query_log_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid query_log_id format. Must be a valid UUID.")

    chain = _get_judge_chain()

    try:
        scores = await chain.ainvoke({
            "question": req.query,
            "context": req.context[:4000],
            "answer": req.answer,
        })
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LangChain GoogleGenerativeAI evaluation failed: {exc}")

    # Validate and clamp scores
    faithfulness = max(0.0, min(10.0, float(scores.get("faithfulness", 0))))
    relevance = max(0.0, min(10.0, float(scores.get("relevance", 0))))
    completeness = max(0.0, min(10.0, float(scores.get("completeness", 0))))
    verdict = str(scores.get("verdict", "FAIL")).upper()
    reasoning = str(scores.get("reasoning", ""))

    # Write scores back to PostgreSQL query_logs
    await db.execute(
        update(QueryLog)
        .where(QueryLog.id == query_log_uuid)
        .values(
            judge_faithfulness=faithfulness,
            judge_relevance=relevance,
            judge_completeness=completeness,
            judge_verdict=verdict,
            judge_reasoning=reasoning,
        )
    )
    await db.commit()

    return JudgeScore(
        faithfulness=faithfulness,
        relevance=relevance,
        completeness=completeness,
        verdict=verdict,
        reasoning=reasoning,
    )
