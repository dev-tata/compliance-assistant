from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.schemas.deliverables import DeliverableExtractionRequest, DeliverableItem
from app.services.deliverable_methods.extraction_method_common import (
    PROMPT_VERSION,
    build_parser_version,
    build_document_extraction_instructions,
    infer_requirement_type,
    normalize_whitespace,
    postprocess_deliverables,
    save_deliverable_extraction_response,
    serialize_sections_for_prompt,
    serialize_sections_for_prompt_legacy,
    should_skip_heading,
    summarize_section_text,
)
from app.services.llm.errors import LLMGenerationError
from app.services.llm.factory import get_llm_service
from app.services.llm.json_utils import extract_json_object

def run_llm_deliverable_extraction(
    *,
    scope_id: str,
    case_payload: dict[str, Any],
    request: DeliverableExtractionRequest,
    case_id: str | None = None,
    document_stored_filename: str | None = None,
    source_filename: str | None = None,
):
    print(
        f"[EXTRACT] started scope_id={scope_id} provider={request.provider} "
        f"model={request.model} mode=llm",
        flush=True,
    )
    llm_service = get_llm_service(request.provider, request.model)
    source_sections = _collect_source_sections(case_payload)
    print(
        f"[EXTRACT] section counts scope_id={scope_id} source_sections={len(source_sections)}",
        flush=True,
    )
    extraction_sections = [
        section
        for section in source_sections
        if not should_skip_heading(section.get("heading_title"), section.get("text"))
    ]
    print(
        f"[EXTRACT] extraction sections scope_id={scope_id} count={len(extraction_sections)}",
        flush=True,
    )

    extracted_items = _extract_document_requirements(
        llm_service=llm_service,
        sections=extraction_sections,
        request=request,
    )

    normalized_deliverables = postprocess_deliverables(
        extracted_items,
        source_sections=source_sections,
    )
    print(
        f"[EXTRACT] normalized items scope_id={scope_id} raw_items={len(extracted_items)} "
        f"normalized_items={len(normalized_deliverables)}",
        flush=True,
    )

    return save_deliverable_extraction_response(
        scope_id=scope_id,
        deliverables=normalized_deliverables,
        extraction_provider=request.provider,
        extraction_model=request.model,
        parser_version=build_parser_version(case_payload),
        case_id=case_id,
        document_stored_filename=document_stored_filename,
        document_content_hash=case_payload.get("procedures", [{}])[0].get("content_hash"),
        source_filename=source_filename,
    )


def _collect_source_sections(case_payload: dict[str, Any]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for document in case_payload.get("procedures", []):
        parsed_json = document.get("parsed_json") or {}
        sections.extend(
            _flatten_sections(
                parsed_json.get("sections", []),
                source_filename=document.get("source_filename"),
            )
        )
    return sections


def _flatten_sections(
    sections: list[dict[str, Any]],
    *,
    source_filename: str | None,
    parent_heading: str | None = None,
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for section in sections:
        heading_title = normalize_whitespace(section.get("heading_title"))
        combined_heading = " / ".join(
            part for part in (parent_heading, heading_title) if part
        ) or heading_title
        section_ref = f"{source_filename or 'document'}::{section.get('section_label') or 'unlabeled'}::{combined_heading or 'untitled'}"
        flattened.append(
            {
                "section_ref": section_ref,
                "source_document": source_filename,
                "section_label": section.get("section_label") or "",
                "heading_title": combined_heading,
                "page_start": section.get("page_start"),
                "page_end": section.get("page_end"),
                "text": summarize_section_text(section.get("text"), max_chars=5200),
                "tables": section.get("tables", []),
            }
        )
        flattened.extend(
            _flatten_sections(
                section.get("subsections", []),
                source_filename=source_filename,
                parent_heading=combined_heading,
            )
        )
    return flattened


def _extract_document_requirements(
    *,
    llm_service: Any,
    sections: list[dict[str, Any]],
    request: DeliverableExtractionRequest,
) -> list[DeliverableItem]:
    if not sections:
        return []

    legacy_payload = {
        "prompt_version": PROMPT_VERSION,
        "sections": serialize_sections_for_prompt_legacy(sections),
    }
    payload = {
        "prompt_version": PROMPT_VERSION,
        "sections": serialize_sections_for_prompt(sections),
    }
    before_prompt = _build_document_extraction_prompt(payload=legacy_payload, instructions=request.instructions)
    after_prompt = _build_document_extraction_prompt(payload=payload, instructions=request.instructions)
    print(
        f"[PROMPT] version={PROMPT_VERSION} "
        f"before_prompt_length={len(before_prompt)} after_prompt_length={len(after_prompt)}",
        flush=True,
    )
    raw_response = llm_service.generate(after_prompt, temperature=0.0)
    section_lookup = {
        normalize_whitespace(section.get("section_ref")): section
        for section in sections
    }
    try:
        payload = extract_json_object(raw_response)
        items = [
            _coerce_item(item, _resolve_item_section(item, section_lookup))
            for item in payload.get("deliverables", [])
        ]
    except (ValidationError, ValueError, TypeError) as exc:
        print(
            f"[EXTRACT] document extraction JSON parsing failed "
            f"sections={len(sections)} raw_preview={raw_response[:500]!r} error={exc}",
            flush=True,
        )
        raise LLMGenerationError(f"Invalid deliverable extraction response from model: {exc}") from exc
    return items


def _build_document_extraction_prompt(*, payload: dict[str, Any], instructions: str | None) -> str:
    prompt_parts = [
        *build_document_extraction_instructions(),
        f"Document payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
    ]
    if instructions:
        prompt_parts.append(f"Additional user instructions:\n{instructions}")
    return "\n\n".join(prompt_parts)


def _resolve_item_section(
    item: dict[str, Any],
    section_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    section_ref = normalize_whitespace(item.get("section_ref"))
    section = section_lookup.get(section_ref)
    if section is not None:
        return section

    return {
        "section_ref": section_ref,
        "source_document": normalize_whitespace(item.get("source_document")),
        "section_label": normalize_whitespace(item.get("section_label")),
        "heading_title": normalize_whitespace(item.get("heading_title")),
    }


def _coerce_item(item: dict[str, Any], section: dict[str, Any]) -> DeliverableItem:
    requirement_text = normalize_whitespace(item.get("requirement_text"))
    heading_title = normalize_whitespace(item.get("heading_title") or section.get("heading_title"))
    section_label = normalize_whitespace(item.get("section_label") or section.get("section_label"))
    source_document = item.get("source_document") or section.get("source_document")
    source_quote = normalize_whitespace(item.get("source_quote"))
    requirement_type = _normalize_requirement_type(
        item.get("requirement_type"),
        requirement_text=requirement_text,
        heading_title=heading_title,
    )
    return DeliverableItem(
        section_label=section_label,
        heading_title=heading_title,
        requirement_text=requirement_text,
        requirement_type=requirement_type,
        mandatory=bool(item.get("mandatory", True)),
        source_quote=source_quote,
        source_document=source_document,
        required_by_procedure=True,
        validated_confidence=0.0,
    )


def _normalize_requirement_type(
    raw_value: Any,
    *,
    requirement_text: str,
    heading_title: str,
) -> str:
    allowed_values = {
        "document_output",
        "recorded_information",
        "approval_or_signoff",
        "update_or_notification",
        "archival_or_storage",
        "change_control",
        "validation_activity",
    }
    normalized = normalize_whitespace(str(raw_value or "")).lower()
    alias_map = {
        "risk_management_obligation": "recorded_information",
        "risk_control_obligation": "recorded_information",
        "documentation_requirement": "document_output",
        "documentation": "document_output",
        "record_requirement": "recorded_information",
        "recording_requirement": "recorded_information",
        "approval_requirement": "approval_or_signoff",
        "update_requirement": "update_or_notification",
        "archive_requirement": "archival_or_storage",
        "storage_requirement": "archival_or_storage",
        "change_requirement": "change_control",
        "test_requirement": "validation_activity",
    }
    if normalized in alias_map:
        return alias_map[normalized]
    if normalized in allowed_values:
        return normalized
    return infer_requirement_type(requirement_text, heading_title)
