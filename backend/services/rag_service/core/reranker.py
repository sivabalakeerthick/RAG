import asyncio
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from shared.config import settings
from shared.ssl_config import google_client_args

# 1. Update Pydantic models to expect a list of scores linked by an ID
class PassageScore(BaseModel):
    chunk_id: int = Field(description="The integer ID of the passage provided in the prompt")
    score: float = Field(description="Relevance score from 0.0 to 1.0")

class RerankResponse(BaseModel):
    results: list[PassageScore]

_parser = JsonOutputParser(pydantic_object=RerankResponse)

# 2. Update the prompt to accept multiple passages
_prompt = PromptTemplate(
    template=(
        "You are an expert search reranker. Score the relevance of EACH passage to the query on a scale from 0.0 to 1.0.\n"
        "Return the scores for all passages.\n\n"
        "{format_instructions}\n\n"
        "QUERY: {query}\n\n"
        "PASSAGES:\n{passages}\n"
    ),
    input_variables=["query", "passages"],
    partial_variables={"format_instructions": _parser.get_format_instructions()},
)

def _get_reranker_chain():
    model = ChatGoogleGenerativeAI(
        model=settings.GEMINI_GENERATION_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.0,
        # INCREASED TOKENS: The LLM needs more tokens to output a list of 10 JSON objects
        max_output_tokens=1024, 
        client_args=google_client_args(),
    )
    return _prompt | model | _parser

async def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Rerank candidate chunks using a single LangChain Gemini request."""
    if not chunks:
        return []

    # 3. Format all 10 chunks into a single string, assigning an ID to each
    passages_text = ""
    for i, chunk in enumerate(chunks):
        passages_text += f"--- PASSAGE ID: {i} ---\n{chunk['text'][:800]}\n\n"

    chain = _get_reranker_chain()
    
    try:
        # 4. Send ONE single request to Gemini
        result = await chain.ainvoke({
            "query": query,
            "passages": passages_text,
        })
        
        # 5. Map the returned scores back to their original IDs
        # Default to 0.0 if the LLM hallucinated or skipped an ID
        score_map = {item["chunk_id"]: item["score"] for item in result.get("results", [])}
        
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(score_map.get(i, chunk.get("rrf_score", 0.0)))
            
    except Exception as e:
        print(f"Listwise reranking failed: {e}")
        # Fallback to rrf_score if the LLM request fails entirely
        for chunk in chunks:
            chunk["rerank_score"] = chunk.get("rrf_score", 0.0)

    # 6. Sort and return the chunks based on the new Gemini scores
    sorted_chunks = sorted(chunks, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    #print(sorted_chunks)
    return sorted_chunks