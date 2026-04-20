from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.schemas.parsing import ParsedTable


def make_parsed_table(
    *,
    rows: list[dict[str, Any]],
    source_format: str,
    table_type: str | None = None,
    extraction_confidence: float | None = None,
) -> ParsedTable | None:
    normalized_rows = _normalize_rows(rows)
    if not normalized_rows:
        return None

    headers = collect_row_headers(normalized_rows)
    resolved_type = table_type or infer_table_type(normalized_rows, headers)
    return ParsedTable(
        table_id=str(uuid4()),
        table_type=resolved_type,
        headers=headers,
        rows=normalized_rows,
        table_markdown=rows_to_markdown(normalized_rows),
        source_format=source_format,
        extraction_confidence=extraction_confidence,
    )


def collect_row_headers(rows: list[dict[str, Any]]) -> list[str]:
    headers: list[str] = []
    for row in rows:
        for key in row.keys():
            candidate = str(key).strip()
            if candidate and candidate not in headers:
                headers.append(candidate)
    return headers


def merge_table_rows(tables: list[ParsedTable]) -> list[dict[str, Any]] | None:
    merged: list[dict[str, Any]] = []
    for table in tables:
        merged.extend(table.rows)
    return merged or None


def rows_to_markdown(rows: list[dict[str, Any]]) -> str:
    headers = collect_row_headers(rows)
    if not headers:
        return ""

    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    data_rows: list[str] = []
    for row in rows:
        values = [_markdown_cell(row.get(header)) for header in headers]
        data_rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header_row, separator, *data_rows])


def infer_table_type(rows: list[dict[str, Any]], headers: list[str] | None = None) -> str:
    resolved_headers = headers or collect_row_headers(rows)
    lowered = {header.strip().lower() for header in resolved_headers}
    if lowered in (
        {"label", "value"},
        {"key", "value"},
        {"field", "value"},
        {"name", "value"},
        {"item", "value"},
        {"parameter", "value"},
        {"metric", "value"},
        {"description", "value"},
    ):
        return "key_value"
    if len(resolved_headers) == 1:
        return "list"
    if len(resolved_headers) >= 2:
        return "matrix"
    return "unknown"


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_row: dict[str, Any] = {}
        for key, value in row.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            if value is None:
                continue
            value_text = str(value).strip()
            if not value_text:
                continue
            normalized_row[key_text] = value_text
        if normalized_row:
            normalized_rows.append(normalized_row)
    return normalized_rows


def _markdown_cell(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return ""
    return text.replace("|", "\\|").replace("\n", "<br>")
