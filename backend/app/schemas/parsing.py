from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ParsedTable(BaseModel):
    table_id: str
    table_type: str = "unknown"
    headers: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    table_markdown: str = ""
    source_format: str | None = None
    extraction_confidence: float | None = None


class ParsedSection(BaseModel):
    section_id: str
    section_label: str | None = None
    parent_section_label: str | None = None
    heading_title: str | None = None
    heading_level: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    sheet_name: str | None = None
    text: str | None = None
    tables: list[ParsedTable] = Field(default_factory=list)
    subsections: list["ParsedSection"] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    source_filename: str
    stored_filename: str
    stored_at: str
    parser_used: str
    metadata: dict[str, Any]
    sections: list[ParsedSection] = Field(default_factory=list)


ParsedSection.model_rebuild()
