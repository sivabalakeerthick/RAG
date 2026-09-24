"""
LangChain Gemini answer generator.
Uses an LCEL chain with ChatPromptTemplate, ChatGoogleGenerativeAI, and StrOutputParser.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from shared.config import settings
from shared.ssl_config import google_client_args

_SYSTEM_PROMPT = """\
You are CogniDoc, an enterprise knowledge assistant.
Answer the user's question using ONLY the information in the provided context.
If the context does not contain enough information, clearly state that.
Be concise and professional.
Do not include inline citations, document IDs, or section numbers in your answer —
the source documents are listed separately in the UI.
"""

_HUMAN_PROMPT = """\
CONTEXT:
{context}

---

QUESTION: {question}

Answer based strictly on the context above.
"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", _HUMAN_PROMPT),
])


def _get_generator_chain():
    model = ChatGoogleGenerativeAI(
        model=settings.GEMINI_GENERATION_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.0,
        max_output_tokens=1024,
        client_args=google_client_args(),
    )
    return _prompt | model | StrOutputParser()


async def generate_answer(question: str, context: str) -> str:
    """Generate a grounded answer using LangChain LCEL chain."""
    chain = _get_generator_chain()
    response = await chain.ainvoke({
        "question": question,
        "context": context[:8000],
    })
    return response.strip()
