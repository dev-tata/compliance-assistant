from __future__ import annotations

from pathlib import Path

from app.schemas.documents import DocumentRecord
from app.schemas.parsing import ParsedDocument
from app.services.parsing.docx_parser import parse_docx
from app.services.parsing.pdf_parser import parse_pdf
from app.services.parsing.xlsx_parser import parse_xlsx


def parse_document(document: DocumentRecord) -> ParsedDocument:
    suffix = Path(document.stored_at).suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(document)
    if suffix == ".docx":
        return parse_docx(document)
    if suffix == ".xlsx":
        return parse_xlsx(document)

    raise ValueError(f"Unsupported file type: {suffix}")
