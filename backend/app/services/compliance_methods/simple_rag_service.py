from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from app.schemas.compliance import (
    ComplianceAnalysis,
    ComplianceFinding,
    ComplianceLinkedRow,
    ComplianceRequest,
    ComplianceResponse,
)
from app.services.compliance_methods.compliance_method_common import (
    build_shared_output_instructions,
    build_compliance_result_path,
)
from app.services.compliance_scoring_service import compute_scores, enrich_analysis_for_scoring
from app.services.document_service import INDEXES_DIR, current_timestamp
from app.services.llm.errors import LLMGenerationError
from app.services.llm.factory import get_llm_service
from app.services.llm.json_utils import extract_json_object
from app.services.retrieval.simple_faiss import (
    build_faiss_index,
    build_record_section_chunks,
    fingerprint_chunks,
    load_cached_faiss_index,
    normalize_whitespace,
    save_cached_faiss_index,
    search_index,
)

RECORD_TOP_K = 10


def run_simple_rag_compliance(
    *,
    case_id: str,
    case_payload: dict[str, Any],
    request: ComplianceRequest,
) -> ComplianceResponse:
    deliverables = [
        item for item in case_payload.get("deliverables", [])
        if normalize_whitespace(item.get("requirement_text"))
    ]
    if not deliverables:
        raise LLMGenerationError(
            "simple_rag requires extracted deliverables. Generate or select deliverables first."
        )

    record_chunks = build_record_section_chunks(case_payload)
    if not record_chunks:
        raise LLMGenerationError("simple_rag requires parsed record sections.")

    llm_service = get_llm_service(request.provider, request.model)
    try:
        record_index = _get_or_create_cached_index(
            case_id=case_id,
            index_name="records",
            chunks=record_chunks,
        )
    except RuntimeError as exc:
        raise LLMGenerationError(str(exc)) from exc

    allowed_record_documents = {
        normalize_whitespace(document.get("source_filename") or document.get("stored_filename"))
        for document in case_payload.get("records", [])
    }
    retrieved_payload = [
        {
            "requirement_ref": f"REQ-{index + 1}",
            "deliverable": _serialize_deliverable(item),
            "retrieved_record_sections": [
                _serialize_record_chunk(section)
                for section in search_index(
                    index=record_index,
                    chunks=record_chunks,
                    query_text=item.get("requirement_text", ""),
                    top_k=RECORD_TOP_K,
                )
            ],
        }
        for index, item in enumerate(deliverables)
    ]

    analysis = _evaluate_deliverables_against_records(
        llm_service=llm_service,
        retrieved_payload=retrieved_payload,
        allowed_record_documents=allowed_record_documents,
        instructions=request.instructions,
    )
    analysis = enrich_analysis_for_scoring(analysis)
    scores = compute_scores(analysis)

    saved_path = build_compliance_result_path(case_id)
    response = ComplianceResponse(
        case_id=case_id,
        compliance_provider=request.provider,
        compliance_model=request.model,
        extraction_provider=case_payload.get("extraction_provider"),
        extraction_model=case_payload.get("extraction_model"),
        method="simple_rag",
        created_at=current_timestamp(),
        saved_at=saved_path.as_posix(),
        analysis=analysis,
        scores=scores,
        section_matches=[],
    )
    saved_path.write_text(
        response.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )
    return response


def _get_or_create_cached_index(
    *,
    case_id: str,
    index_name: str,
    chunks: list[dict[str, Any]],
) -> Any:
    fingerprint = fingerprint_chunks(chunks)
    index_dir = INDEXES_DIR / f"case_{case_id}" / index_name

    cached_index = load_cached_faiss_index(
        index_dir=index_dir,
        expected_fingerprint=fingerprint,
    )
    if cached_index is not None:
        return cached_index

    index, _ = build_faiss_index(chunks)
    save_cached_faiss_index(
        index_dir=index_dir,
        index=index,
        chunks=chunks,
        fingerprint=fingerprint,
    )
    return index


def _evaluate_deliverables_against_records(
    *,
    llm_service: Any,
    retrieved_payload: list[dict[str, Any]],
    allowed_record_documents: set[str],
    instructions: str | None,
) -> ComplianceAnalysis:
    prompt_parts = [
        *build_shared_output_instructions(),
        "You are evaluating multiple procedure deliverables against retrieved record sections.",
        "Use only the retrieved record sections provided for each requirement as evidence.",
        "Preserve the requirement text exactly from the deliverable payload where possible.",
        "Each procedure_to_record item must correspond to one requirement_ref in the same order.",
        "For satisfied items, linked_rows.gap and linked_rows.recommendation may be empty.",
        "For partial and not_satisfied items, provide a short gap and recommendation.",
        f"Requirement evaluations:\n{json.dumps(retrieved_payload, ensure_ascii=False, indent=2)}",
    ]
    if instructions:
        prompt_parts.append(f"Additional user instructions:\n{instructions}")
    prompt = "\n\n".join(prompt_parts)
    try:
        payload = extract_json_object(llm_service.generate(prompt, temperature=0.0))
    except ValueError as exc:
        raise LLMGenerationError(f"Invalid simple_rag aggregated evaluation response: {exc}") from exc

    try:
        analysis = ComplianceAnalysis(**payload)
    except ValidationError as exc:
        raise LLMGenerationError(f"Invalid simple_rag aggregated evaluation payload: {exc}") from exc

    normalized_findings: list[ComplianceFinding] = []
    for index, finding in enumerate(analysis.procedure_to_record):
        retrieved_sections = retrieved_payload[index].get("retrieved_record_sections", []) if index < len(retrieved_payload) else []
        normalized_findings.append(
            finding.model_copy(
                update={
                    "source_documents": _sanitize_source_documents(
                        finding.source_documents,
                        allowed=allowed_record_documents,
                        fallback=[item.get("source_document") for item in retrieved_sections],
                    ),
                    "evidence": _normalize_string_list(finding.evidence),
                }
            )
        )
    analysis = analysis.model_copy(
        update={
            "procedure_to_record": normalized_findings,
            "findings": normalized_findings,
            "overall_assessment": analysis.overall_assessment or _compute_overall_assessment(normalized_findings),
        }
    )
    if analysis.linked_rows:
        analysis = analysis.model_copy(
            update={
                "linked_rows": [
                    row.model_copy(
                        update={
                            "requirement": row.requirement or (
                                normalized_findings[index].requirement if index < len(normalized_findings) else ""
                            ),
                        }
                    )
                    for index, row in enumerate(analysis.linked_rows)
                ]
            }
        )
    else:
        analysis = analysis.model_copy(
            update={
                "linked_rows": [
                    ComplianceLinkedRow(
                        requirement_ref=f"REQ-{index + 1}",
                        requirement=finding.requirement,
                        status=finding.status,
                        gap="" if finding.status == "satisfied" else finding.requirement,
                        recommendation=""
                        if finding.status == "satisfied"
                        else _recommended_action_for_requirement(finding.requirement),
                    )
                    for index, finding in enumerate(normalized_findings)
                ],
                "gaps": [finding.requirement for finding in normalized_findings if finding.status != "satisfied"],
                "recommended_actions": [
                    _recommended_action_for_requirement(finding.requirement)
                    for finding in normalized_findings
                    if finding.status != "satisfied"
                ],
            }
        )
    return analysis


def _serialize_record_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_document": chunk.get("source_document"),
        "section_label": chunk.get("section_label"),
        "heading_title": chunk.get("heading_title"),
        "text": chunk.get("text"),
        "table_markdown": chunk.get("table_markdown"),
        "retrieval_score": chunk.get("retrieval_score"),
    }


def _serialize_deliverable(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_document": item.get("source_document"),
        "section_label": item.get("section_label"),
        "heading_title": item.get("heading_title"),
        "requirement_text": item.get("requirement_text"),
        "source_quote": item.get("source_quote"),
        "retrieval_score": item.get("retrieval_score"),
    }


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [normalize_whitespace(str(item)) for item in value if normalize_whitespace(str(item))]


def _sanitize_source_documents(
    value: Any,
    *,
    allowed: set[str],
    fallback: list[Any],
) -> list[str]:
    requested = _normalize_string_list(value)
    sanitized = [item for item in requested if item in allowed]
    if sanitized:
        return sanitized
    fallback_sanitized = [
        normalize_whitespace(str(item))
        for item in fallback
        if normalize_whitespace(str(item)) in allowed
    ]
    return list(dict.fromkeys(fallback_sanitized))


def _compute_overall_assessment(findings: list[ComplianceFinding]) -> str:
    statuses = [finding.status for finding in findings]
    if statuses and all(status == "satisfied" for status in statuses):
        return "satisfied"
    if any(status in {"satisfied", "partial"} for status in statuses):
        return "partial"
    return "not_satisfied"


def _recommended_action_for_requirement(requirement: str) -> str:
    text = normalize_whitespace(requirement)
    if not text:
        return "Address the missing requirement."

    replacements = [
        (r"^In cases where (.+?) shall record (.+)$", r"When \1, record \2."),
        (r"^(.+?) shall be documented\.?$", r"Document \1."),
        (r"^(.+?) shall be defined\.?$", r"Define \1."),
        (r"^(.+?) shall be implemented\.?$", r"Implement \1."),
        (r"^(.+?) shall be performed\.?$", r"Perform \1."),
        (r"^(.+?) shall be used\.?$", r"Use \1."),
        (r"^(.+?) shall be described\.?$", r"Describe \1."),
        (r"^(.+?) shall record (.+)$", r"Record \2 for \1."),
    ]
    for pattern, replacement in replacements:
        if re.match(pattern, text, flags=re.IGNORECASE):
            rewritten = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            rewritten = rewritten[0].upper() + rewritten[1:]
            return rewritten
    return f"Address requirement: {text}"
