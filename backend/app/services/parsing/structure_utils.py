from __future__ import annotations

from uuid import uuid4

from app.schemas.parsing import ParsedSection, ParsedTable


def make_section(
    *,
    section_label: str | None,
    heading_title: str | None,
    heading_level: int | None,
    parent_section_label: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    sheet_name: str | None = None,
    text: str | None = None,
    tables: list[ParsedTable] | None = None,
    subsections: list[ParsedSection] | None = None,
) -> ParsedSection:
    resolved_tables = list(tables or [])
    return ParsedSection(
        section_id=str(uuid4()),
        section_label=section_label,
        parent_section_label=parent_section_label,
        heading_title=heading_title,
        heading_level=heading_level,
        page_start=page_start,
        page_end=page_end,
        sheet_name=sheet_name,
        text=text,
        tables=resolved_tables,
        subsections=subsections or [],
    )


def count_sections(sections: list[ParsedSection]) -> int:
    total = 0
    for section in sections:
        total += 1 + count_sections(section.subsections)
    return total
