from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from app.schemas.deliverables import (
    DeliverableExtractionMethod,
    DeliverableExtractionResponse,
    DeliverableItem,
    RequirementType,
)
from app.services.document_service import DELIVERABLES_DIR, current_timestamp

PROMPT_VERSION = "deliverable_extraction_v3_single_call"

OBLIGATION_TERMS = ("shall", "must", "required", "needs to")
EXCLUDED_HEADINGS = {
    "aim",
    "purpose",
    "scope",
    "scope and application",
    "responsibilities",
    "conclusion",
    "document history",
}
ALLOWED_EXCLUDED_HEADINGS = {"references"}


def build_document_extraction_instructions() -> list[str]:
    return [
        "You are an information extraction engine for validation procedures.",
        "Extract only atomic explicit obligation items from the provided procedure document sections.",
        "Be exhaustive. If a section contains multiple explicit obligations, extract all of them.",
        "Preserve the operational detail of each obligation. Do not shorten away conditions, qualifiers, actors, targets, or required outputs.",
        "Hard rules:",
        "- Use only the provided sections and tables.",
        "- Extract only obligations explicitly stated in this document.",
        "- Ignore responsibilities, aims, conclusions, and background statements unless they contain an explicit obligation.",
        "- Ignore referenced SOP contents unless this document explicitly requires that content here.",
        "- Return one item per atomic obligation.",
        "- Prefer a complete standalone requirement sentence over a sentence fragment.",
        "- Each item must include an exact short source_quote copied from the section.",
        "- The source_quote must support the requirement_text directly.",
        "- Do not summarize a whole document or whole section into one broad item.",
        "- Do not include optional guidance unless the wording is mandatory.",
        "- Every item must include the matching section_ref from the input.",
        "- Do not repeat source_document, section_label, or heading_title in the output.",
        "- If an obligation is expressed as part of a longer sentence, rewrite requirement_text into a complete clear sentence while preserving the original meaning exactly.",
        "- Keep requirement_text specific. Do not replace concrete terms with vague wording.",
        "- Capture obligations stated in running text, bullet-like text, and tables.",
        "Use this exact JSON structure only:",
        "{"
        '"deliverables":['
        "{"
        '"section_ref":"<section_ref from input>",'
        '"requirement_text":"A validation plan shall be established and approved prior to initiating validation.",'
        '"requirement_type":"document_output",'
        '"mandatory":true,'
        '"source_quote":"A validation plan ... shall be established and approved prior to initiating validation."'
        "}"
        "]"
        "}",
    ]


def build_deliverable_result_path(
    *,
    scope_id: str,
    case_id: str | None = None,
    document_stored_filename: str | None = None,
    document_content_hash: str | None = None,
    extraction_provider: str | None = None,
    extraction_model: str | None = None,
    method: DeliverableExtractionMethod | None = None,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if case_id:
        filename = f"case_{case_id}_deliverables_{timestamp}_{uuid4().hex}.json"
    elif document_content_hash and extraction_provider and extraction_model and method:
        provider_slug = _slugify_filename_part(extraction_provider)
        model_slug = _slugify_filename_part(extraction_model)
        filename = (
            f"document_{document_content_hash}_{provider_slug}_{model_slug}_{method}"
            f"_deliverables_{timestamp}_{uuid4().hex}.json"
        )
    elif document_stored_filename:
        filename = f"document_{document_stored_filename}_deliverables_{timestamp}_{uuid4().hex}.json"
    else:
        filename = f"deliverables_{scope_id}_{timestamp}_{uuid4().hex}.json"
    return DELIVERABLES_DIR / filename


def save_deliverable_extraction_response(
    *,
    scope_id: str,
    method: DeliverableExtractionMethod,
    deliverables: list[DeliverableItem],
    extraction_provider: str,
    extraction_model: str,
    parser_version: str,
    case_id: str | None = None,
    document_stored_filename: str | None = None,
    document_content_hash: str | None = None,
    source_filename: str | None = None,
) -> DeliverableExtractionResponse:
    saved_path = build_deliverable_result_path(
        scope_id=scope_id,
        case_id=case_id,
        document_stored_filename=document_stored_filename,
        document_content_hash=document_content_hash,
        extraction_provider=extraction_provider,
        extraction_model=extraction_model,
        method=method,
    )
    created_at = current_timestamp()
    response = DeliverableExtractionResponse(
        case_id=case_id,
        document_stored_filename=document_stored_filename,
        source_filename=source_filename,
        extraction_provider=extraction_provider,
        extraction_model=extraction_model,
        prompt_version=PROMPT_VERSION,
        parser_version=parser_version,
        method=method,
        created_at=created_at,
        saved_at=saved_path.as_posix(),
        deliverables=deliverables,
    )
    saved_path.write_text(
        response.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )
    return response


def _slugify_filename_part(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return normalized or "unknown"


def build_parser_version(case_payload: dict[str, Any]) -> str:
    versions: set[str] = set()
    for document in case_payload.get("procedures", []):
        parsed_json = document.get("parsed_json") or {}
        parser_used = parsed_json.get("parser_used") or "unknown"
        metadata = parsed_json.get("metadata") or {}
        backend = metadata.get("parser_backend") or "unknown"
        source_format = metadata.get("source_format") or "unknown"
        versions.add(f"{parser_used}:{backend}:{source_format}")
    return ", ".join(sorted(versions)) if versions else "unknown"


def summarize_section_text(text: str | None, *, max_chars: int = 1800) -> str:
    if not text:
        return ""
    normalized = " ".join(text.split())
    return normalized[:max_chars]


def normalize_whitespace(value: str | None) -> str:
    return " ".join((value or "").split()).strip()


def normalize_heading(value: str | None) -> str:
    return normalize_whitespace(value).lower()


def is_explicit_obligation(text: str | None) -> bool:
    normalized = normalize_whitespace(text).lower()
    return any(term in normalized for term in OBLIGATION_TERMS)


def should_skip_heading(heading_title: str | None, text: str | None) -> bool:
    heading = normalize_heading(heading_title)
    if not heading:
        return False
    if heading in ALLOWED_EXCLUDED_HEADINGS:
        return not is_explicit_obligation(text)
    if heading in EXCLUDED_HEADINGS:
        return not is_explicit_obligation(text)
    return False


def serialize_section_for_prompt(section: dict[str, Any]) -> dict[str, Any]:
    tables = section.get("tables", [])
    serialized_tables: list[dict[str, Any]] = []
    for table in tables[:6]:
        serialized_tables.append(
            {
                "headers": table.get("headers", []),
                "table_markdown": table.get("table_markdown"),
            }
        )
    return {
        "section_ref": section.get("section_ref"),
        "source_document": section.get("source_document"),
        "section_label": section.get("section_label"),
        "heading_title": section.get("heading_title"),
        "text": section.get("text"),
        "tables": serialized_tables,
    }


def serialize_sections_for_prompt(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [serialize_section_for_prompt(section) for section in sections]


def postprocess_deliverables(
    deliverables: list[DeliverableItem],
    *,
    source_sections: list[dict[str, Any]],
) -> list[DeliverableItem]:
    source_map = {
        (section.get("source_document"), section.get("section_label"), section.get("heading_title")): section
        for section in source_sections
    }
    section_label_map = {
        (section.get("source_document"), section.get("section_label")): section
        for section in source_sections
    }

    normalized: list[DeliverableItem] = []
    for item in deliverables:
        key = (item.source_document, item.section_label, item.heading_title)
        section = source_map.get(key)
        if section is None:
            fallback_key = (item.source_document, item.section_label)
            section = section_label_map.get(fallback_key)
            if section is not None:
                print(
                    f"[POSTPROCESS] fallback source section match by section_label "
                    f"section_label={item.section_label!r} heading_title={item.heading_title!r} "
                    f"resolved_heading_title={section.get('heading_title')!r}",
                    flush=True,
                )
        if section is None:
            print(
                f"[POSTPROCESS] reject item reason=missing_source_section "
                f"section_label={item.section_label!r} heading_title={item.heading_title!r} "
                f"requirement={item.requirement_text!r}",
                flush=True,
            )
            continue
        if should_skip_heading(item.heading_title, section.get("text")):
            print(
                f"[POSTPROCESS] reject item reason=skipped_heading "
                f"section_label={item.section_label!r} heading_title={item.heading_title!r} "
                f"requirement={item.requirement_text!r}",
                flush=True,
            )
            continue
        if not item.section_label or not item.heading_title:
            print(
                f"[POSTPROCESS] reject item reason=missing_section_metadata "
                f"section_label={item.section_label!r} heading_title={item.heading_title!r} "
                f"requirement={item.requirement_text!r}",
                flush=True,
            )
            continue
        if not _quote_matches_section(item.source_quote, section):
            print(
                f"[POSTPROCESS] reject item reason=quote_not_in_section "
                f"section_label={item.section_label!r} heading_title={item.heading_title!r} "
                f"quote={item.source_quote!r} requirement={item.requirement_text!r}",
                flush=True,
            )
            continue
        if _looks_broad(item):
            print(
                f"[POSTPROCESS] reject item reason=too_broad "
                f"section_label={item.section_label!r} heading_title={item.heading_title!r} "
                f"requirement={item.requirement_text!r}",
                flush=True,
            )
            continue
        if not _is_narrow_explicit_requirement(item):
            print(
                f"[POSTPROCESS] reject item reason=not_narrow_explicit_requirement "
                f"section_label={item.section_label!r} heading_title={item.heading_title!r} "
                f"quote={item.source_quote!r} requirement={item.requirement_text!r}",
                flush=True,
            )
            continue
        normalized.append(_normalize_item(item, section=section))

    return _deduplicate(normalized)


def infer_requirement_type(
    requirement_text: str,
    heading_title: str | None,
) -> RequirementType:
    text = normalize_whitespace(f"{heading_title or ''} {requirement_text}").lower()
    if any(term in text for term in ("approve", "approved", "signature", "sign-off")):
        return "approval_or_signoff"
    if any(term in text for term in ("archive", "stored", "preserved", "retain")):
        return "archival_or_storage"
    if any(term in text for term in ("update", "notify", "notification")):
        return "update_or_notification"
    if any(term in text for term in ("change request", "change", "deviation")):
        return "change_control"
    if any(term in text for term in ("validate", "testing", "test", "revalidation")):
        return "validation_activity"
    if any(term in text for term in ("report", "plan", "specification", "document", "record")):
        return "document_output"
    return "recorded_information"


def compute_deliverable_confidence(
    item: DeliverableItem,
    *,
    section: dict[str, Any] | None = None,
) -> float:
    score = 0.28
    requirement_text = normalize_whitespace(item.requirement_text)
    source_quote = normalize_whitespace(item.source_quote)

    if item.mandatory:
        score += 0.1
    if item.section_label.strip():
        score += 0.07
    if normalize_whitespace(item.heading_title):
        score += 0.07
    if normalize_whitespace(item.source_document):
        score += 0.05
    if requirement_text and any(term in requirement_text.lower() for term in OBLIGATION_TERMS):
        score += 0.12
    if source_quote:
        score += 0.1

    req_words = set(re.findall(r"[a-z0-9]+", requirement_text.lower()))
    quote_words = set(re.findall(r"[a-z0-9]+", source_quote.lower()))
    if req_words and quote_words:
        overlap = len(req_words & quote_words) / len(req_words)
        score += min(0.12, overlap * 0.12)

    word_count = len(requirement_text.split())
    if 6 <= word_count <= 30:
        score += 0.08
    elif word_count < 4 or word_count > 45:
        score -= 0.08

    if section is not None:
        if _quote_matches_section(source_quote, section):
            score += 0.11
        if is_explicit_obligation(section.get("text")):
            score += 0.05
        if should_skip_heading(item.heading_title, section.get("text")):
            score -= 0.15

    return round(min(0.98, max(0.0, score)), 4)


def _normalize_item(item: DeliverableItem, *, section: dict[str, Any] | None = None) -> DeliverableItem:
    requirement_text = normalize_whitespace(item.requirement_text)
    source_quote = normalize_whitespace(item.source_quote)
    return item.model_copy(
        update={
            "section_label": item.section_label.strip(),
            "heading_title": normalize_whitespace(item.heading_title),
            "requirement_text": requirement_text,
            "source_quote": "" if source_quote == requirement_text else source_quote,
            "confidence": compute_deliverable_confidence(item, section=section),
        }
    )


def _quote_matches_section(source_quote: str, section: dict[str, Any]) -> bool:
    quote = _normalized_match_text(source_quote)
    if not quote:
        return False

    section_text = _normalized_match_text(section.get("text"))
    if quote in section_text:
        return True

    for table in section.get("tables", []):
        markdown = _normalized_match_text(table.get("table_markdown"))
        if quote in markdown:
            return True
    return False


def _normalized_match_text(value: Any) -> str:
    text = normalize_whitespace(str(value or ""))
    return re.sub(r"\s+", " ", text).lower()


def _looks_broad(item: DeliverableItem) -> bool:
    text = item.requirement_text.lower()
    broad_markers = (
        "the purpose of this activity",
        "the process applies to",
        "this sop applies",
        "responsible for",
        "is responsible for",
        "overall process",
    )
    if any(marker in text for marker in broad_markers):
        return True
    if text.count(" shall ") > 1:
        return True
    if len(text.split()) > 45:
        return True
    return False


def _is_narrow_explicit_requirement(item: DeliverableItem) -> bool:
    text = item.requirement_text.lower()
    quote = item.source_quote.lower()
    if not any(term in text for term in OBLIGATION_TERMS):
        return False
    if "sop " in text and not any(term in quote for term in OBLIGATION_TERMS):
        return False
    if not item.mandatory:
        return False
    return True


def _deduplicate(items: list[DeliverableItem]) -> list[DeliverableItem]:
    deduped: dict[tuple[str, str, str], DeliverableItem] = {}
    for item in items:
        key = _dedupe_key(item)
        existing = deduped.get(key)
        if existing is None or item.confidence > existing.confidence:
            deduped[key] = item
    return sorted(
        deduped.values(),
        key=lambda item: (
            item.source_document.lower(),
            item.section_label.lower(),
            item.requirement_text.lower(),
        ),
    )


def _dedupe_key(item: DeliverableItem) -> tuple[str, str, str]:
    return (
        item.source_document.strip().lower(),
        item.section_label.strip().lower(),
        normalize_whitespace(item.requirement_text).lower(),
    )
