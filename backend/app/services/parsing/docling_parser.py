from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter

from app.schemas.documents import DocumentRecord
from app.schemas.parsing import ParsedDocument, ParsedSection, ParsedTable
from app.services.parsing.structure_utils import count_sections, make_section
from app.services.parsing.table_utils import make_parsed_table

logger = logging.getLogger(__name__)


def parse_with_docling(document: DocumentRecord) -> ParsedDocument:
    file_path = Path(document.stored_at)
    source_format = file_path.suffix.lower().lstrip(".") or "document"
    logger.info("Attempting Docling parse for %s", document.source_filename)

    converter = DocumentConverter()
    conversion_result = converter.convert(str(file_path))
    docling_document = getattr(conversion_result, "document", None)
    if docling_document is None:
        raise RuntimeError(
            f"Docling returned no structured document for file: {file_path.name}"
        )

    sections = _build_sections_from_docling(docling_document, source_format)
    if not sections:
        raise RuntimeError(
            f"Docling conversion produced no structured sections for file: {file_path.name}"
        )

    metadata: dict[str, object] = {
        "structure": "sections",
        "sections_detected": count_sections(sections),
        "docling_enabled": True,
        "parser_backend": "docling",
        "source_format": source_format,
    }

    page_count = _extract_docling_page_count(conversion_result, docling_document)
    if page_count is not None:
        metadata["page_count"] = page_count
        metadata["pages_with_text"] = page_count

    return ParsedDocument(
        source_filename=document.source_filename,
        stored_filename=document.stored_filename,
        stored_at=document.stored_at,
        parser_used="docling",
        metadata=metadata,
        sections=sections,
    )


def _build_sections_from_docling(
    docling_document: Any, source_format: str
) -> list[ParsedSection]:
    iterator = getattr(docling_document, "iterate_items", None)
    body = getattr(docling_document, "body", None)
    if not callable(iterator):
        raise RuntimeError("Docling document does not expose iterate_items().")

    sections: list[ParsedSection] = []
    current_section: ParsedSection | None = None
    preamble_lines: list[str] = []

    for item, depth in iterator(root=body, with_groups=True):
        if _is_heading_item(item):
            heading_text = _normalize_text(_extract_item_text(item))
            if not heading_text:
                continue

            section_label, heading_title = _split_heading_label(heading_text)
            heading_level = _coerce_heading_level(item, depth, section_label)
            section = make_section(
                section_label=section_label,
                heading_title=heading_title,
                heading_level=heading_level,
                page_start=_extract_item_page(item),
                page_end=_extract_item_page(item),
                subsections=[],
            )
            sections.append(section)
            current_section = section
            continue

        if _is_table_item(item):
            table = _build_table_from_docling_item(item, source_format)
            if table is None:
                continue

            if current_section is None:
                current_section = make_section(
                    section_label="document_body",
                    heading_title="Document Body",
                    heading_level=1,
                    subsections=[],
                )
                sections.append(current_section)

            current_section.tables.append(table)
            _update_section_pages(current_section, item)
            continue

        text = _normalize_text(_extract_item_text(item))
        if not text:
            continue

        if current_section is None:
            preamble_lines.append(text)
            continue

        _append_text(current_section, text)
        _update_section_pages(current_section, item)

    if preamble_lines:
        preamble_text = "\n\n".join(_collapse_blank_lines(preamble_lines)).strip()
        if preamble_text:
            sections.insert(
                0,
                make_section(
                    section_label="document_body",
                    heading_title="Document Body",
                    heading_level=1,
                    text=preamble_text,
                ),
            )

    return _postprocess_sections(sections, source_format)


def _is_heading_item(item: Any) -> bool:
    return type(item).__name__ in {"TitleItem", "SectionHeaderItem"}


def _is_table_item(item: Any) -> bool:
    return type(item).__name__ == "TableItem"


def _extract_item_text(item: Any) -> str | None:
    for attribute_name in ("text", "orig"):
        value = getattr(item, attribute_name, None)
        if isinstance(value, str) and value.strip():
            return value

    export_text = getattr(item, "export_to_text", None)
    if callable(export_text):
        value = export_text()
        if isinstance(value, str) and value.strip():
            return value

    return None


def _split_heading_label(heading_text: str) -> tuple[str | None, str]:
    match = re.match(r"^(?P<label>\d+(?:\.\d+)*)(?:\.)?\s+(?P<title>.+)$", heading_text)
    if not match:
        return None, heading_text
    return match.group("label"), match.group("title").strip()


def _coerce_heading_level(item: Any, depth: int, section_label: str | None) -> int:
    for attribute_name in ("level", "heading_level"):
        value = getattr(item, attribute_name, None)
        if isinstance(value, int) and value > 0:
            return value

    if section_label:
        return max(1, section_label.count(".") + 1)

    return max(1, depth + 1)


def _extract_item_page(item: Any) -> int | None:
    prov = getattr(item, "prov", None)
    if prov is None:
        return None

    provenance_items = prov if isinstance(prov, list) else [prov]
    page_numbers: list[int] = []
    for provenance in provenance_items:
        for attribute_name in ("page_no", "page"):
            value = getattr(provenance, attribute_name, None)
            if isinstance(value, int):
                page_numbers.append(value)

    if not page_numbers:
        return None
    return min(page_numbers)


def _update_section_pages(section: ParsedSection, item: Any) -> None:
    page = _extract_item_page(item)
    if page is None:
        return

    if section.page_start is None or page < section.page_start:
        section.page_start = page
    if section.page_end is None or page > section.page_end:
        section.page_end = page


def _append_text(section: ParsedSection, text: str) -> None:
    if not section.text:
        section.text = text
        return

    if section.text.endswith(text):
        return

    section.text = f"{section.text}\n\n{text}"


def _collapse_blank_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if previous_blank:
                continue
            previous_blank = True
            cleaned.append("")
            continue
        previous_blank = False
        cleaned.append(line)
    return cleaned


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""

    normalized_lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            normalized_lines.append(line)

    return "\n".join(normalized_lines).strip()


def _build_table_from_docling_item(item: Any, source_format: str) -> ParsedTable | None:
    headers, rows = _extract_docling_table_rows(item)
    if not rows:
        return None

    table = make_parsed_table(
        rows=rows,
        source_format=source_format,
        table_type=_infer_table_type(headers),
        extraction_confidence=0.97,
    )
    if table is not None:
        table.headers = headers
        table.table_markdown = _rows_to_markdown(headers, rows)
    return table


def _postprocess_sections(
    sections: list[ParsedSection], source_format: str
) -> list[ParsedSection]:
    flat_sections = _normalize_section_labels(sections)
    flat_sections = _fold_inline_sections(flat_sections)
    if source_format != "pdf":
        flat_sections = _split_signature_sections(flat_sections, source_format)

    for section in flat_sections:
        section.tables = _normalize_tables_for_section(section)
        if section.section_label:
            section.heading_level = section.section_label.count(".") + 1
        elif section.heading_level is None:
            section.heading_level = 1
        section.subsections = []
        if not section.text:
            section.text = None

    return _nest_sections_by_label(flat_sections)


def _split_signature_sections(
    sections: list[ParsedSection], source_format: str
) -> list[ParsedSection]:
    processed: list[ParsedSection] = []
    for section in sections:
        signature_text = _extract_signature_block(section.text)
        if not signature_text:
            processed.append(section)
            continue

        body_text = (section.text or "").replace(signature_text, "").strip()
        updated_section = section.model_copy(
            update={
                "text": body_text or None,
            }
        )
        processed.append(updated_section)
        processed.append(
            make_section(
                section_label=None,
                heading_title="Signatures",
                heading_level=section.heading_level,
                page_start=section.page_start,
                page_end=section.page_end,
                text=None,
                tables=_build_signature_tables(
                    signature_text, source_format=source_format
                ),
                subsections=[],
            )
        )
    return processed


def _extract_signature_block(text: str | None) -> str | None:
    if not text:
        return None

    signature_patterns = (
        r"\bcreated by\s*\(date/sign",
        r"\breviewed by\s*\(date/sign",
        r"\bapproved by\s*\(date/sign",
        r"\bcreated by\b",
        r"\breviewed by\b",
        r"\bapproved by\b",
        r"\bauthor reviewer\b",
        r"\bapprover\b",
        r"\bauthor\b\s*\n+\s*\breviewer\b",
    )
    normalized_text = text.lower()
    marker_positions: list[int] = []
    for pattern in signature_patterns:
        match = re.search(pattern, normalized_text, flags=re.IGNORECASE | re.MULTILINE)
        if match is not None:
            marker_positions.append(match.start())
    if not marker_positions:
        return None

    start = min(marker_positions)
    signature_text = text[start:].strip()
    if not signature_text:
        return None
    return signature_text


def _build_signature_tables(signature_text: str, *, source_format: str) -> list[ParsedTable]:
    rows = _extract_signature_rows(signature_text)
    if not rows:
        return []

    table = make_parsed_table(
        rows=rows,
        source_format=source_format,
        table_type="matrix",
        extraction_confidence=0.9,
    )
    if table is None:
        return []

    table.headers = ["Role", "Date/Sign", "Name"]
    table.table_type = "matrix"
    table.table_markdown = _rows_to_markdown(table.headers, table.rows)
    return [table]


def _extract_signature_rows(signature_text: str) -> list[dict[str, str]]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", signature_text) if block.strip()]
    rows: list[dict[str, str]] = []
    pending_roles: list[str] | None = None

    for block in blocks:
        normalized = " ".join(block.split())
        if not normalized:
            continue

        role_entries = _parse_signature_role_block(normalized)
        if role_entries:
            pending_roles = role_entries
            continue

        if pending_roles and "___" in normalized:
            continue

        if pending_roles:
            names = _split_signature_names(block, len(pending_roles))

            for index, role in enumerate(pending_roles):
                rows.append(
                    {
                        "Role": role,
                        "Date/Sign": "date/sign",
                        "Name": names[index] if index < len(names) else "",
                    }
                )
            pending_roles = None

    return [row for row in rows if row.get("Role") or row.get("Name")]


def _parse_signature_role_block(text: str) -> list[str] | None:
    pattern = re.compile(
        r"(Created by|Reviewed by|Approved by|Author|Reviewer|Approver)\s*(?:\([^)]*date/sign[^)]*\)|Date:)?",
        flags=re.IGNORECASE,
    )
    matches = [match.group(1).strip() for match in pattern.finditer(text)]
    if not matches:
        return None
    return matches


def _split_signature_names(block: str, expected_count: int) -> list[str]:
    pieces = [piece.strip(" ,") for piece in re.split(r"\n+", block) if piece.strip()]
    if len(pieces) == expected_count and all("," in piece for piece in pieces):
        return pieces

    flat = " ".join(pieces)
    matches = _extract_name_title_spans(flat)
    if len(matches) >= expected_count:
        return matches[:expected_count]

    comma_splits = [piece.strip(" ,") for piece in re.split(r"(?<=,)\s+(?=[A-Z])", flat) if piece.strip()]
    if len(comma_splits) >= expected_count:
        return comma_splits[:expected_count]

    if len(pieces) == expected_count:
        return pieces

    return pieces[:expected_count]


def _extract_name_title_spans(text: str) -> list[str]:
    name_anchor_pattern = re.compile(
        r"\b([A-Z][a-z]+(?:[-'][A-Z][a-z]+)?(?:\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?){1,3}),"
    )
    anchors = list(name_anchor_pattern.finditer(text))
    if not anchors:
        return []

    spans: list[str] = []
    for index, anchor in enumerate(anchors):
        start = anchor.start()
        name_end = anchor.end()
        next_start = anchors[index + 1].start() if index + 1 < len(anchors) else len(text)
        title = text[name_end:next_start].strip(" ,")
        if title:
            spans.append(f"{text[start:name_end].strip(' ,')} {title}".strip())
        else:
            spans.append(text[start:name_end].strip(" ,"))
    return spans


def _normalize_section_labels(sections: list[ParsedSection]) -> list[ParsedSection]:
    normalized: list[ParsedSection] = []
    current_top_root: str | None = None

    for section in sections:
        label = section.section_label
        if label:
            parts = label.split(".")
            if len(parts) == 1:
                current_top_root = parts[0]
            elif current_top_root and parts[0] != current_top_root:
                parts[0] = current_top_root
                section.section_label = ".".join(parts)
        normalized.append(section)

    return normalized


def _fold_inline_sections(sections: list[ParsedSection]) -> list[ParsedSection]:
    kept: list[ParsedSection] = []
    for section in sections:
        if _should_fold_inline_section(section):
            anchor = _find_fold_anchor(kept)
            if anchor is not None:
                _merge_inline_section(anchor, section)
                continue
        kept.append(section)
    return kept


def _should_fold_inline_section(section: ParsedSection) -> bool:
    if section.section_label:
        return False
    if section.heading_title == "Document Body":
        return False
    if section.tables:
        return False
    return True


def _find_fold_anchor(sections: list[ParsedSection]) -> ParsedSection | None:
    for section in reversed(sections):
        if section.section_label or section.heading_title == "Document Body":
            return section
    return sections[-1] if sections else None


def _merge_inline_section(anchor: ParsedSection, inline_section: ParsedSection) -> None:
    parts: list[str] = []
    if inline_section.heading_title:
        parts.append(inline_section.heading_title)
    if inline_section.text:
        parts.append(inline_section.text)

    merged_text = "\n\n".join(part for part in parts if part).strip()
    if merged_text:
        _append_text(anchor, merged_text)

    if inline_section.tables:
        anchor.tables.extend(inline_section.tables)

    _merge_section_pages(anchor, inline_section)


def _merge_section_pages(target: ParsedSection, source: ParsedSection) -> None:
    for attr_name, reducer in (("page_start", min), ("page_end", max)):
        target_value = getattr(target, attr_name)
        source_value = getattr(source, attr_name)
        if source_value is None:
            continue
        if target_value is None:
            setattr(target, attr_name, source_value)
            continue
        setattr(target, attr_name, reducer(target_value, source_value))


def _nest_sections_by_label(sections: list[ParsedSection]) -> list[ParsedSection]:
    roots: list[ParsedSection] = []
    label_map: dict[str, ParsedSection] = {}

    for section in sections:
        if not section.section_label:
            roots.append(section)
            continue

        parent = None
        parent_label = _parent_label(section.section_label)
        while parent_label:
            parent = label_map.get(parent_label)
            if parent is not None:
                break
            parent_label = _parent_label(parent_label)

        section.parent_section_label = parent.section_label if parent is not None else None
        if parent is not None:
            parent.subsections.append(section)
        else:
            roots.append(section)
        label_map[section.section_label] = section

    return roots


def _parent_label(label: str) -> str | None:
    if "." not in label:
        return None
    return label.rsplit(".", 1)[0]


def _normalize_tables_for_section(section: ParsedSection) -> list[ParsedTable]:
    normalized_tables: list[ParsedTable] = []
    for table in section.tables:
        normalized_table = _normalize_table(section, table)
        if normalized_table is None:
            continue
        normalized_tables.append(normalized_table)
    return normalized_tables


def _normalize_table(section: ParsedSection, table: ParsedTable) -> ParsedTable | None:
    headers = list(table.headers)
    rows = [dict(row) for row in table.rows]
    if not headers or not rows:
        return None

    if _is_numeric_header_pair(headers):
        old_headers = list(headers)
        normalized_headers = _preferred_pair_headers(section)
        if _looks_like_label_value_rows(rows, old_headers):
            headers = normalized_headers
            rows = [
                {
                    normalized_headers[0]: row.get(old_headers[0], ""),
                    normalized_headers[1]: row.get(old_headers[1], ""),
                }
                for row in rows
                if row.get(old_headers[0]) or row.get(old_headers[1])
            ]

    promoted_headers, promoted_rows = _promote_header_row(headers, rows)
    if promoted_headers and promoted_rows:
        headers = promoted_headers
        rows = promoted_rows

    headers = _rename_generic_headers(section, headers, rows)

    if table.table_type == "matrix":
        rows = [row for row in rows if not _is_placeholder_matrix_row(row, headers)]
        if not rows:
            return None

    if _is_low_signal_single_column_table(headers, rows):
        return None

    rebuilt = make_parsed_table(
        rows=rows,
        source_format=table.source_format or "document",
        table_type=_infer_table_type(headers),
        extraction_confidence=table.extraction_confidence,
    )
    if rebuilt is None:
        return None

    rebuilt.headers = headers
    rebuilt.table_type = _infer_table_type(headers)
    rebuilt.table_markdown = _rows_to_markdown(headers, rows)

    return rebuilt


def _preferred_pair_headers(section: ParsedSection) -> list[str]:
    title = (section.heading_title or "").upper()
    if "LEVEL" in title:
        return ["Level", "Definition"]
    return ["Label", "Value"]


def _promote_header_row(
    headers: list[str], rows: list[dict[str, str]]
) -> tuple[list[str] | None, list[dict[str, str]] | None]:
    if len(headers) < 2 or not rows:
        return None, None

    if not all(header.isdigit() or header.startswith("column_") for header in headers):
        return None, None

    first_row = rows[0]
    candidate_headers = [_normalize_text(first_row.get(header, "")) for header in headers]
    if not all(candidate_headers):
        return None, None
    if not _looks_like_header_values(candidate_headers):
        return None, None

    new_headers = _normalize_headers(candidate_headers)
    new_rows: list[dict[str, str]] = []
    for row in rows[1:]:
        mapped_row: dict[str, str] = {}
        for index, old_header in enumerate(headers):
            value = row.get(old_header, "")
            if value:
                mapped_row[new_headers[index]] = value
        if mapped_row:
            new_rows.append(mapped_row)

    if not new_rows:
        return None, None
    return new_headers, new_rows


def _looks_like_header_values(values: list[str]) -> bool:
    for value in values:
        if len(value.split()) > 4:
            return False
        if re.search(r"[.!?]", value):
            return False
    return True


def _rename_generic_headers(
    section: ParsedSection, headers: list[str], rows: list[dict[str, str]]
) -> list[str]:
    title = (section.heading_title or "").upper()
    renamed = list(headers)

    if len(renamed) == 2 and renamed[1].lower().startswith("column_"):
        if "RISK CONTROL" in title:
            renamed[1] = "Description"
        elif "REFERENCE" in title:
            renamed[1] = "ID"
        elif any(len((row.get(renamed[1], "") or "").split()) > 4 for row in rows):
            renamed[1] = "Description"

    if len(renamed) == 2 and renamed[0].lower().startswith("column_"):
        if "REFERENCE" in title:
            renamed[0] = "Title"
        elif any(len((row.get(renamed[0], "") or "").split()) <= 4 for row in rows):
            renamed[0] = "Item"

    return _apply_header_renames(renamed, headers, rows)


def _apply_header_renames(
    renamed_headers: list[str], original_headers: list[str], rows: list[dict[str, str]]
) -> list[str]:
    if renamed_headers == original_headers:
        return original_headers

    for row in rows:
        updated_row: dict[str, str] = {}
        for index, original_header in enumerate(original_headers):
            new_header = renamed_headers[index]
            if original_header in row:
                updated_row[new_header] = row[original_header]
        row.clear()
        row.update(updated_row)

    return renamed_headers


def _is_numeric_header_pair(headers: list[str]) -> bool:
    return len(headers) == 2 and all(header.isdigit() for header in headers)


def _looks_like_label_value_rows(rows: list[dict[str, str]], headers_old: list[str]) -> bool:
    first_values = [row.get(headers_old[0], "").strip() for row in rows]
    second_values = [row.get(headers_old[1], "").strip() for row in rows]
    if not all(first_values) or not all(second_values):
        return False
    return any(len(value.split()) <= 3 for value in first_values) and any(len(value.split()) > 3 for value in second_values)


def _is_placeholder_matrix_row(row: dict[str, str], headers: list[str]) -> bool:
    if not headers:
        return False
    if len(row) != 1:
        return False
    first_header = headers[0]
    value = row.get(first_header, "").strip().lower()
    if not value:
        return False

    header_tokens = [token for token in re.split(r"\s+", first_header.strip().lower()) if token]
    return value in header_tokens


def _is_low_signal_single_column_table(headers: list[str], rows: list[dict[str, str]]) -> bool:
    if len(headers) != 1:
        return False
    values = [row.get(headers[0], "").strip().lower() for row in rows if row.get(headers[0])]
    if len(values) < 3:
        return False
    return set(values).issubset({"low", "medium", "high", "yes", "no", "n/a"})


def _extract_docling_table_rows(item: Any) -> tuple[list[str], list[dict[str, str]]]:
    export_to_dataframe = getattr(item, "export_to_dataframe", None)
    if callable(export_to_dataframe):
        try:
            dataframe = export_to_dataframe()
            if dataframe is not None and not dataframe.empty:
                headers = _normalize_headers([str(column) for column in dataframe.columns.tolist()])
                rows: list[dict[str, str]] = []
                for record in dataframe.fillna("").to_dict(orient="records"):
                    row = _normalize_row(record, headers)
                    if row:
                        rows.append(row)
                if rows:
                    return headers, rows
        except Exception as exc:
            logger.debug("Docling dataframe table export failed: %s", exc)

    exported_dict = _export_docling_item_dict(item)
    if not exported_dict:
        return [], []

    table_data = exported_dict.get("data")
    if isinstance(table_data, dict):
        headers, rows = _extract_rows_from_table_data(table_data)
        if rows:
            return headers, rows

    for key in ("table_cells", "cells"):
        cells = exported_dict.get(key)
        if isinstance(cells, list):
            headers, rows = _extract_rows_from_cells(cells)
            if rows:
                return headers, rows

    return [], []


def _export_docling_item_dict(item: Any) -> dict[str, Any] | None:
    exporter = getattr(item, "export_to_dict", None)
    if callable(exporter):
        try:
            value = exporter()
            if isinstance(value, dict):
                return value
        except Exception as exc:
            logger.debug("Docling item export_to_dict failed: %s", exc)
    return None


def _extract_rows_from_table_data(table_data: dict[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
    raw_rows = table_data.get("grid") or table_data.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        return [], []

    normalized_grid: list[list[str]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, list):
            continue
        normalized_row: list[str] = []
        for cell in raw_row:
            normalized_row.append(_extract_cell_text(cell))
        if any(cell for cell in normalized_row):
            normalized_grid.append(normalized_row)

    if not normalized_grid:
        return [], []

    header_row = normalized_grid[0]
    body_rows = normalized_grid[1:] if len(normalized_grid) > 1 else []
    if not body_rows:
        body_rows = normalized_grid
        header_row = [f"column_{index + 1}" for index in range(len(body_rows[0]))]

    headers = _normalize_headers(header_row)
    rows: list[dict[str, str]] = []
    for raw_row in body_rows:
        row = {
            headers[index]: value
            for index, value in enumerate(raw_row[: len(headers)])
            if value
        }
        if row:
            rows.append(row)
    return headers, rows


def _extract_rows_from_cells(cells: list[Any]) -> tuple[list[str], list[dict[str, str]]]:
    matrix: dict[int, dict[int, str]] = {}
    max_col = -1
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        row_index = _coerce_int(cell.get("row_index"), cell.get("start_row_offset_idx"), default=None)
        col_index = _coerce_int(cell.get("col_index"), cell.get("start_col_offset_idx"), default=None)
        if row_index is None or col_index is None:
            continue

        matrix.setdefault(row_index, {})[col_index] = _extract_cell_text(cell)
        max_col = max(max_col, col_index)

    if not matrix or max_col < 0:
        return [], []

    ordered_rows: list[list[str]] = []
    for row_index in sorted(matrix):
        ordered_rows.append([matrix[row_index].get(col_index, "") for col_index in range(max_col + 1)])

    return _extract_rows_from_table_data({"rows": ordered_rows})


def _extract_cell_text(cell: Any) -> str:
    if isinstance(cell, str):
        return _normalize_text(cell)
    if isinstance(cell, dict):
        for key in ("text", "content", "label"):
            value = cell.get(key)
            if isinstance(value, str) and value.strip():
                return _normalize_text(value)
    return ""


def _normalize_headers(headers: list[str]) -> list[str]:
    normalized: list[str] = []
    used: set[str] = set()
    for index, header in enumerate(headers):
        value = _normalize_text(header) or f"column_{index + 1}"
        if value in used:
            suffix = 2
            candidate = f"{value}_{suffix}"
            while candidate in used:
                suffix += 1
                candidate = f"{value}_{suffix}"
            value = candidate
        used.add(value)
        normalized.append(value)
    return normalized


def _normalize_row(record: dict[str, Any], headers: list[str]) -> dict[str, str]:
    row: dict[str, str] = {}
    values = list(record.values())
    for index, header in enumerate(headers):
        if index >= len(values):
            break
        value = _normalize_text(str(values[index]))
        if value:
            row[header] = value
    return row


def _rows_to_markdown(headers: list[str], rows: list[dict[str, str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = []
    for row in rows:
        values = [row.get(header, "") for header in headers]
        row_lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header_line, separator_line, *row_lines])


def _coerce_int(*values: Any, default: int | None) -> int | None:
    for value in values:
        if isinstance(value, int):
            return value
    return default


def _extract_docling_page_count(conversion_result: Any, docling_document: Any) -> int | None:
    for candidate in (
        getattr(conversion_result, "pages", None),
        getattr(docling_document, "pages", None),
    ):
        if candidate is not None:
            try:
                return len(candidate)
            except TypeError:
                continue

    for candidate in (
        getattr(conversion_result, "page_count", None),
        getattr(docling_document, "page_count", None),
        getattr(docling_document, "num_pages", None),
    ):
        if isinstance(candidate, int):
            return candidate

    return None


def _infer_table_type(headers: list[str]) -> str:
    lowered = {header.strip().lower() for header in headers}
    if lowered in (
        {"label", "value"},
        {"key", "value"},
        {"level", "definition"},
        {"risk", "definition"},
    ):
        return "key_value"
    if lowered == {"role", "responsibility"}:
        return "matrix"
    if len(headers) == 1:
        return "list"
    return "matrix"
