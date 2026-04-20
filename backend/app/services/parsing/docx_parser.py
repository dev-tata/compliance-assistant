from __future__ import annotations

from app.schemas.documents import DocumentRecord
from app.schemas.parsing import ParsedDocument
from app.services.parsing.docling_parser import parse_with_docling


def parse_docx(document: DocumentRecord) -> ParsedDocument:
    return parse_with_docling(document)
