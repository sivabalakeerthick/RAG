"""
Advanced Multi-Strategy Document Text Extraction and Chunking.
Uses industry-standard high-precision extraction libraries:
  - PDF: pymupdf4llm (Markdown conversion with tables & headers)
  - DOCX: mammoth (Clean Markdown conversion preserving headings & lists)
  - XLSX/XLS/CSV: pandas (Markdown pipe table conversion)
  - TXT/MD: MarkdownHeaderTextSplitter + Token-based Recursive Character Splitter
"""
import logging
from pathlib import Path

from docx import Document as DocxDocument
import pymupdf as fitz  # PyMuPDF (1.28+ deprecates `import fitz`)
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
import mammoth
import pandas as pd
import pymupdf4llm
import tiktoken

from shared.config import settings

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")

SUPPORTED_TYPES: dict[str, str] = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xls",
    "text/csv": "csv",
    "application/octet-stream": "auto",
}


def _extract_pdf(file_path: str) -> str:
    """Extract PDF to Markdown preserving headers and tables using pymupdf4llm."""
    try:
        md_text = pymupdf4llm.to_markdown(file_path)
        if md_text and md_text.strip():
            return md_text.strip()
    except Exception as exc:
        logger.warning("pymupdf4llm extraction failed, falling back to PyMuPDF fitz: %s", exc)

    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append(f"# Page {page_num}\n{text}")
    return "\n\n".join(pages)


def _extract_docx(file_path: str) -> str:
    """Extract DOCX to Markdown preserving headings, lists, and tables using mammoth."""
    try:
        with open(file_path, "rb") as f:
            result = mammoth.convert_to_markdown(f)
            md_text = result.value
            if md_text and md_text.strip():
                return md_text.strip()
    except Exception as exc:
        logger.warning("mammoth DOCX extraction failed, falling back to python-docx: %s", exc)

    doc = DocxDocument(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(f"| {row_text} |")
    return "\n\n".join(paragraphs)


def _extract_spreadsheet(file_path: str, fmt: str) -> str:
    """Extract XLSX/XLS/CSV into formatted Markdown tables using pandas."""
    try:
        if fmt == "csv":
            df = pd.read_csv(file_path)
            if not df.empty:
                return f"# Data Table\n\n{df.to_markdown(index=False)}"
        else:
            excel_file = pd.ExcelFile(file_path)
            sections = []
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                if not df.empty:
                    sections.append(f"# Sheet: {sheet_name}\n\n{df.to_markdown(index=False)}")
            if sections:
                return "\n\n---\n\n".join(sections)
    except Exception as exc:
        logger.warning("pandas spreadsheet extraction failed: %s", exc)

    return Path(file_path).read_text(encoding="utf-8", errors="ignore")


def _extract_txt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8", errors="ignore")


def _detect_format(file_path: str) -> str:
    ext = file_path.rsplit(".", 1)[-1].lower()
    return {
        "pdf": "pdf", "docx": "docx", "txt": "txt", "md": "txt",
        "xlsx": "xlsx", "xls": "xls", "csv": "csv",
    }.get(ext, "txt")


def _extract_text(file_path: str, content_type: str) -> str:
    fmt = SUPPORTED_TYPES.get(content_type, "auto")
    if fmt == "auto":
        fmt = _detect_format(file_path)

    if fmt == "pdf":
        return _extract_pdf(file_path)
    elif fmt == "docx":
        return _extract_docx(file_path)
    elif fmt in ("xlsx", "xls", "csv"):
        return _extract_spreadsheet(file_path, fmt)
    else:
        return _extract_txt(file_path)


def chunk_document(file_path: str, content_type: str) -> list[dict]:
    """
    Multi-strategy document chunker:
      1. Uses pymupdf4llm / mammoth / pandas for high-fidelity Markdown extraction.
      2. If headers (#, ##) exist, applies MarkdownHeaderTextSplitter first.
      3. Splits sub-sections using RecursiveCharacterTextSplitter with tiktoken.
    """
    text = _extract_text(file_path, content_type)
    if not text.strip():
        raise ValueError("Document appears to be empty or unreadable.")

    # 1. Structural Header Splitting (if markdown headers are present)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    try:
        header_docs = markdown_splitter.split_text(text)
    except Exception:
        header_docs = []

    # 2. Token-level Recursive Splitting
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    chunks = []
    if header_docs:
        for doc in header_docs:
            sub_chunks = text_splitter.split_text(doc.page_content)
            header_prefix = " > ".join(
                str(v) for k, v in doc.metadata.items() if k.startswith("Header")
            )
            for sub in sub_chunks:
                if sub.strip():
                    full_text = f"[{header_prefix}]\n{sub}" if header_prefix else sub
                    token_count = len(_enc.encode(full_text))
                    chunks.append({"text": full_text, "tokens": token_count})
    else:
        split_texts = text_splitter.split_text(text)
        for chunk_text in split_texts:
            if chunk_text.strip():
                token_count = len(_enc.encode(chunk_text))
                chunks.append({"text": chunk_text, "tokens": token_count})

    return chunks
