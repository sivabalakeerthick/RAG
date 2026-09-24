"""
Document text extraction and fixed-size chunking with token overlap.
Supports: PDF (PyMuPDF), DOCX (python-docx), TXT, XLSX (openpyxl), XLS (xlrd).
"""
import tiktoken
from pathlib import Path

from shared.config import settings

_enc = tiktoken.get_encoding("cl100k_base")

# ── MIME type → extractor map ─────────────────────────────────────────────────

SUPPORTED_TYPES: dict[str, str] = {
    "application/pdf":                                                           "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  "docx",
    "text/plain":                                                                "txt",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":        "xlsx",
    "application/vnd.ms-excel":                                                  "xls",
    # Browsers sometimes send this for .xlsx
    "application/octet-stream":                                                  "auto",
}


def _extract_pdf(file_path: str) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append(f"[Page {page_num}]\n{text}")
    return "\n\n".join(pages)


def _extract_docx(file_path: str) -> str:
    from docx import Document as DocxDocument
    doc = DocxDocument(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    # Also extract tables
    for table in doc.tables:
        for row in table.rows:
            row_text = "\t".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)
    return "\n".join(paragraphs)


def _extract_txt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8", errors="ignore")


def _extract_xlsx(file_path: str) -> str:
    import openpyxl
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    sections = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            row_text = "\t".join(
                str(cell) if cell is not None else "" for cell in row
            )
            if row_text.strip():
                rows.append(row_text)
        if rows:
            sections.append(f"[Sheet: {sheet_name}]\n" + "\n".join(rows))
    wb.close()
    return "\n\n".join(sections)


def _extract_xls(file_path: str) -> str:
    import xlrd
    wb = xlrd.open_workbook(file_path)
    sections = []
    for sheet in wb.sheets():
        rows = []
        for row_idx in range(sheet.nrows):
            row_text = "\t".join(
                str(sheet.cell_value(row_idx, col)) for col in range(sheet.ncols)
            )
            if row_text.strip():
                rows.append(row_text)
        if rows:
            sections.append(f"[Sheet: {sheet.name}]\n" + "\n".join(rows))
    return "\n\n".join(sections)


def _detect_format(file_path: str) -> str:
    """Detect format by extension as fallback when MIME is ambiguous."""
    ext = file_path.rsplit(".", 1)[-1].lower()
    return {
        "pdf": "pdf", "docx": "docx", "txt": "txt",
        "xlsx": "xlsx", "xls": "xls",
    }.get(ext, "txt")


def _extract_text(file_path: str, content_type: str) -> str:
    """Extract raw text from a file based on its MIME type."""
    fmt = SUPPORTED_TYPES.get(content_type, "auto")
    if fmt == "auto":
        fmt = _detect_format(file_path)

    extractors = {
        "pdf":  _extract_pdf,
        "docx": _extract_docx,
        "txt":  _extract_txt,
        "xlsx": _extract_xlsx,
        "xls":  _extract_xls,
    }
    extractor = extractors.get(fmt)
    if not extractor:
        raise ValueError(f"Unsupported format: {fmt}")
    return extractor(file_path)


from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_document(file_path: str, content_type: str) -> list[dict]:
    """
    Extract text and split using LangChain's RecursiveCharacterTextSplitter.from_tiktoken_encoder.

    Returns:
        list of {"text": str, "tokens": int}
    """
    text = _extract_text(file_path, content_type)
    if not text.strip():
        raise ValueError("Document appears to be empty or unreadable.")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    split_texts = splitter.split_text(text)

    chunks = []
    for chunk_text in split_texts:
        if chunk_text.strip():
            token_count = len(_enc.encode(chunk_text))
            chunks.append({"text": chunk_text, "tokens": token_count})

    return chunks
