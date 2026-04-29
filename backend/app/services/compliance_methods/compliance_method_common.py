from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from app.schemas.compliance import (
    ComplianceAnalysis,
    ComplianceEvidenceItem,
    ComplianceFinding,
    ComplianceLinkedRow,
    ComplianceMethod,
    ComplianceRequest,
    ComplianceResponse,
)
from app.services.compliance_scoring_service import enrich_analysis_for_scoring
from app.services.document_service import current_timestamp
from app.services.llm.errors import LLMGenerationError
from app.services.llm.factory import get_llm_service
from app.services.llm.json_utils import extract_json_object
from app.services.nli.evidence_contradiction_service import apply_evidence_contradiction_verification
from app.services.retrieval.faiss_retrieval import normalize_whitespace
from app.services.storage_paths import get_case_compliance_dir

PROMPT_FAMILY = "canonical_compliance_v2_relaxed"
MIN_SATISFIED_EVIDENCE_ITEMS = 1
MIN_SATISFIED_SOURCE_DOCUMENTS = 1
MIN_STRONG_CLAIM_EVIDENCE_ITEMS = 2
MIN_SUBSTANTIVE_SECTION_TEXT_LENGTH = 40
MIN_SUBSTANTIVE_SECTION_TOKEN_COUNT = 6
STATUS_COMPLETION_SCORES = {
    "satisfied": 100,
    "partial": 33,
    "not_satisfied": 0,
}


def execute_compliance_method(
    *,
    case_id: str,
    case_payload: dict[str, object],
    request: ComplianceRequest,
    method: ComplianceMethod | str,
    prompt: str,
) -> ComplianceResponse:
    llm_service = get_llm_service(request.provider, request.model)
    allowed_record_documents = extract_allowed_record_documents(case_payload)
    deliverables = annotate_deliverable_structure(
        list(case_payload.get("deliverables", [])) if isinstance(case_payload, dict) else []
    )
    analysis = parse_compliance_analysis_response(
        raw_analysis=llm_service.generate(prompt, temperature=0.0),
        method=method,
        expected_count=len(deliverables),
        allowed_record_documents=allowed_record_documents,
    )
    analysis = enrich_analysis_for_scoring(
        analysis,
        requirement_weights=_extract_deliverable_weights(case_payload),
        deliverable_metadata=deliverables,
    )
    analysis = assemble_compliance_analysis(
        findings=analysis.procedure_to_record or analysis.findings,
        linked_rows=analysis.linked_rows,
    )
    saved_path = build_compliance_result_path(case_id)
    created_at = current_timestamp()
    response = ComplianceResponse(
        case_id=case_id,
        compliance_provider=request.provider,
        compliance_model=request.model,
        extraction_provider=case_payload.get("extraction_provider") if isinstance(case_payload, dict) else None,
        extraction_model=case_payload.get("extraction_model") if isinstance(case_payload, dict) else None,
        method=method,
        reference_stored_filenames=[
            item.get("stored_filename")
            for item in case_payload.get("references", [])
            if isinstance(item, dict) and item.get("stored_filename")
        ] if isinstance(case_payload, dict) else [],
        created_at=created_at,
        saved_at=saved_path.as_posix(),
        analysis=analysis,
        section_matches=[],
    )
    saved_path.write_text(
        response.model_dump_json(
            indent=2,
            exclude_none=True,
        ),
        encoding="utf-8",
    )
    return response


def _extract_deliverable_weights(case_payload: dict[str, object]) -> list[float] | None:
    if not isinstance(case_payload, dict):
        return None
    deliverables = case_payload.get("deliverables", [])
    if not isinstance(deliverables, list):
        return None

    weights: list[float] = []
    for item in deliverables:
        if isinstance(item, dict):
            raw_weight = item.get("weight")
            if isinstance(raw_weight, (int, float)) and raw_weight > 0:
                weights.append(float(raw_weight))
                continue
        weights.append(1.0)
    return weights


def build_shared_output_instructions(
    *,
    single_requirement: bool = False,
    include_feedback: bool = False,
) -> list[str]:
    linked_row_shape = '"linked_rows":[{"requirement_ref":"REQ-1","rationale":"short traceability rationale"}],'
    output_structure = [
        "{",
        linked_row_shape,
        '"procedure_to_record":[{"requirement":"...","evidence":["verbatim or near-verbatim short quotes"],"source_document":"record-file-name"}]',
        "}",
    ]
    return [
        "You are a compliance adjudicator.",
        "Evaluate whether the record documents satisfy the explicit requirements stated in the extracted procedure deliverables.",
        "Task rules:",
        "- Evaluate each requirement independently.",
        "- Use only admissible evidence from the input payload for this method.",
        "- Treat extracted deliverables as the canonical requirement source.",
        "- Evaluate only requirement-to-record content match. Do not assume any preferred record template, section set, field set, workflow, or document style unless the requirement itself states it.",
        "- Provide evidence as short quote-based plain text grounded in admissible record evidence.",
        "- Do not use evidence fields for commentary, absence statements, reasoning, conclusions, or recommendations.",
        "- Do not invent requirements, evidence, sections, systems, roles, or context.",
        "Assessment guidance:",
        "- Break each requirement into the concrete elements it requires.",
        "- Check each element against admissible evidence only.",
        "- The backend will derive compliance status from the grounded evidence you return.",
        "- Your job is to return grounded evidence, the source document, and a short rationale describing how well the record supports the requirement.",
        "- Prefer substance over exact wording, but require the record evidence to establish the required point clearly. Treat equivalent wording or structured evidence conservatively unless it directly covers the required element.",
        "- Do not require a specific record layout, heading name, table shape, or wording pattern unless the requirement explicitly requires it.",
        "- Do not penalize a record merely because evidence appears in a different section, format, or phrasing than the procedure text.",
        "- A matching heading, title, or section number alone is never sufficient evidence. Use the section body text or table content, not structure alone, to justify compliance.",
        "- Be careful with indirect or summarized support. Do not treat indirect or inferred support as satisfied unless it clearly establishes the required element.",
        "- If a deliverable indicates broader subsection coverage expectations, use satisfied only when the evidence spans enough distinct relevant subsection areas rather than one local mention.",
        "- Reference context may clarify requirement meaning, but it does not create additional required fields or record structure beyond the requirement itself.",
        "- For conditional requirements such as 'where necessary' or 'in cases where', first assess whether the triggering condition is evidenced in the admissible record.",
        "- If the trigger is not evidenced, do not assume the condition occurred.",
        "- For near-complete records, do not ignore missing material elements. If a material required point is not evidenced, the rationale should make that clear.",
        "Output requirements:",
        "- Return one JSON object only.",
        "- Do not return markdown or code fences.",
        (
            "- Include exactly one procedure_to_record item and one linked_rows item for the single input requirement."
            if single_requirement
            else "- Include exactly one procedure_to_record item per input requirement in the same order."
        ),
        (
            "- Use requirement_ref REQ-1 for the single input requirement."
            if single_requirement
            else "- Include one linked_rows item per procedure_to_record item in the same order."
        ),
        (
            "- Do not emit any requirement_ref other than REQ-1."
            if single_requirement
            else "- Use requirement_ref values REQ-1, REQ-2, and so on, matching the input order."
        ),
        "- Every procedure_to_record.source_document value must refer only to a record document filename from this case.",
        "- For every requirement, set linked_rows.rationale to one short plain-text sentence grounded in the admissible record evidence.",
        "- Do not generate gaps, recommendations, or recommended_actions.",
        "- Do not generate status fields in linked_rows.",
        "- Do not include findings in the model output. The backend will derive findings from procedure_to_record.",
        "Return JSON with exactly this structure:",
        *output_structure,
    ]


def build_method_specific_constraints(method: ComplianceMethod | str) -> list[str]:
    shared = [f"Method metadata: {method}."]
    if method == "non_rag":
        return [
            *shared,
            "- Use full records as admissible evidence.",
            "- You may search anywhere inside records.",
            "- Do not use retrieval results because none are provided for this method.",
        ]
    if method == "record_retrieval_stage":
        return [
            *shared,
            "- Use ONLY retrieved_record_sections as admissible evidence.",
            "- Do not rely on any full record document outside retrieved_record_sections.",
            "- If a claim is supported only outside retrieved_record_sections, treat it as not admissible and return partial or not_satisfied accordingly.",
        ]
    return [
        *shared,
        "- Use ONLY retrieved_record_sections as admissible compliance evidence.",
        "- Use retrieved_requirement_context only to interpret the meaning of a requirement.",
        "- Never use retrieved_requirement_context as compliance evidence and never cite reference documents in source_document.",
        "- Evaluate each requirement from scratch using the provided deliverable, retrieved_record_sections, and retrieved_requirement_context.",
        "- Do not rely on any full record document outside retrieved_record_sections.",
        "- If a claim is supported only outside retrieved_record_sections, treat it as not admissible and return partial or not_satisfied accordingly.",
    ]


def build_compliance_prompt(
    *,
    method: ComplianceMethod | str,
    payload: dict[str, Any],
    instructions: str | None,
    single_requirement: bool = False,
    include_feedback: bool = False,
) -> str:
    prompt_parts = [
        *build_shared_output_instructions(
            single_requirement=single_requirement,
            include_feedback=include_feedback,
        ),
        "Input payload notes:",
        "- The payload contains the requirement list plus method-specific context blocks.",
        *build_method_specific_constraints(method),
        f"Prompt family: {PROMPT_FAMILY}",
        f"Compliance payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
    ]
    if instructions:
        prompt_parts.append(f"Additional user instructions:\n{instructions}")
    return "\n\n".join(prompt_parts)


def serialize_baseline_analysis(analysis: ComplianceAnalysis) -> list[dict[str, Any]]:
    findings = analysis.procedure_to_record or analysis.findings
    return [
        {
            "requirement_ref": f"REQ-{index + 1}",
            "requirement": finding.requirement,
            "status": finding.status,
            "evidence": finding.evidence,
            "source_document": finding.source_document,
        }
        for index, finding in enumerate(findings)
    ]


def extract_allowed_record_documents(case_payload: dict[str, Any] | dict[str, object]) -> set[str]:
    if not isinstance(case_payload, dict):
        return set()
    return {
        normalize_whitespace(document.get("source_filename") or document.get("stored_filename"))
        for document in case_payload.get("records", [])
        if isinstance(document, dict)
        and normalize_whitespace(document.get("source_filename") or document.get("stored_filename"))
    }


def simplify_document_for_prompt(document: dict[str, Any]) -> dict[str, Any]:
    parsed_json = document.get("parsed_json") or {}
    return {
        "document_type": document.get("document_type"),
        "source_filename": document.get("source_filename"),
        "stored_filename": document.get("stored_filename"),
        "group_id": document.get("group_id"),
        "language": document.get("language"),
        "parsed_json": {
            "source_filename": parsed_json.get("source_filename"),
            "stored_filename": parsed_json.get("stored_filename"),
            "parser_used": parsed_json.get("parser_used"),
            "metadata": parsed_json.get("metadata"),
            "sections": parsed_json.get("sections", []),
        },
    }


def build_compliance_result_path(case_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"case_{case_id}_compliance_{timestamp}_{uuid4().hex}.json"
    return get_case_compliance_dir(case_id) / filename


def build_requirement_query_text(deliverable: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            deliverable.get("requirement_text", ""),
            deliverable.get("source_quote", ""),
            deliverable.get("heading_title", ""),
        )
        if normalize_whitespace(part)
    )


def serialize_retrieved_section(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_document": chunk.get("source_document"),
        "section_label": chunk.get("section_label"),
        "heading_title": chunk.get("heading_title"),
        "text": chunk.get("text"),
        "table_markdown": chunk.get("table_markdown"),
        "faiss_score": chunk.get("faiss_score"),
        "reranker_score": chunk.get("reranker_score"),
        "retrieval_score": chunk.get("retrieval_score"),
    }


def serialize_deliverable_for_prompt(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_document": item.get("source_document"),
        "section_label": item.get("section_label"),
        "heading_title": item.get("heading_title"),
        "requirement_text": item.get("requirement_text"),
        "source_quote": item.get("source_quote"),
        "weight": item.get("weight"),
        "expected_evidence_breadth": item.get("expected_evidence_breadth"),
        "retrieval_score": item.get("retrieval_score"),
    }


def annotate_deliverable_structure(deliverables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [dict(item) for item in deliverables]
    section_labels = [
        normalize_whitespace(item.get("section_label"))
        for item in normalized
        if normalize_whitespace(item.get("section_label"))
    ]

    annotated: list[dict[str, Any]] = []
    for item in normalized:
        section_label = normalize_whitespace(item.get("section_label"))
        descendants = _count_descendant_sections(section_label, section_labels)
        expected_breadth = min(3, max(1, descendants if descendants else 1))
        annotated.append(
            {
                **item,
                "expected_evidence_breadth": expected_breadth,
            }
        )
    return annotated


def _count_descendant_sections(section_label: str, section_labels: list[str]) -> int:
    if not section_label:
        return 0
    descendants: set[str] = set()
    prefix = f"{section_label}."
    for candidate in section_labels:
        if not candidate.startswith(prefix):
            continue
        suffix = candidate[len(prefix):]
        immediate = suffix.split(".", 1)[0].strip()
        if immediate:
            descendants.add(immediate)
    return len(descendants)


def build_single_requirement_prompt(
    *,
    method: ComplianceMethod | str,
    requirement_payload: dict[str, Any],
    instructions: str | None,
    include_feedback: bool = False,
) -> str:
    return build_compliance_prompt(
        method=method,
        payload=requirement_payload,
        instructions=instructions,
        single_requirement=True,
        include_feedback=include_feedback,
    )


def parse_compliance_analysis_response(
    *,
    raw_analysis: str,
    method: str,
    expected_count: int,
    allowed_record_documents: set[str] | None = None,
) -> ComplianceAnalysis:
    try:
        analysis = ComplianceAnalysis(**extract_json_object(raw_analysis))
    except (ValidationError, ValueError) as exc:
        raise LLMGenerationError(f"Invalid compliance response from model: {exc}") from exc
    return normalize_compliance_analysis(
        analysis=analysis,
        method=method,
        expected_count=expected_count,
        allowed_record_documents=allowed_record_documents,
    )


def normalize_compliance_analysis(
    *,
    analysis: ComplianceAnalysis,
    method: str,
    expected_count: int,
    allowed_record_documents: set[str] | None = None,
) -> ComplianceAnalysis:
    validate_analysis_cardinality(
        analysis=analysis,
        expected_count=expected_count,
        method=method,
    )
    if analysis.procedure_to_record and not analysis.findings:
        analysis = analysis.model_copy(update={"findings": analysis.procedure_to_record})
    if allowed_record_documents is not None:
        analysis = _normalize_analysis_source_document(
            analysis=analysis,
            allowed_record_documents=allowed_record_documents,
        )
    if analysis.linked_rows:
        analysis = analysis.model_copy(
            update={
                "linked_rows": [
                    _sanitize_linked_row_recommendation(row)
                    for row in analysis.linked_rows
                ]
            }
        )
    if not analysis.linked_rows and analysis.procedure_to_record:
        analysis = analysis.model_copy(update={"linked_rows": _build_linked_rows(analysis)})
    elif analysis.linked_rows and analysis.procedure_to_record:
        analysis = analysis.model_copy(update={"linked_rows": _resolve_linked_row_requirements(analysis)})
    return analysis


def evaluate_single_requirement(
    *,
    llm_service: Any,
    method: ComplianceMethod | str,
    requirement_payload: dict[str, Any],
    instructions: str | None,
    temperature: float = 0.0,
    include_feedback: bool = False,
) -> ComplianceAnalysis:
    prompt = build_single_requirement_prompt(
        method=method,
        requirement_payload=requirement_payload,
        instructions=instructions,
        include_feedback=include_feedback,
    )
    return parse_compliance_analysis_response(
        raw_analysis=llm_service.generate(prompt, temperature=temperature),
        method=f"{method} requirement",
        expected_count=1,
    )


def normalize_requirement_finding(
    *,
    finding: ComplianceFinding,
    retrieved_record_sections: list[dict[str, Any]],
    allowed_record_documents: set[str],
) -> ComplianceFinding:
    normalized_evidence = normalize_string_list(finding.evidence)
    normalized = finding.model_copy(
        update={
            "source_document": sanitize_source_document(
                finding.source_document,
                allowed=allowed_record_documents,
                fallback=[item.get("source_document") for item in retrieved_record_sections],
            ),
            "evidence": normalized_evidence,
            "evidence_items": [
                ComplianceEvidenceItem(
                    text=evidence,
                    source_document=sanitize_source_document(
                        finding.source_document,
                        allowed=allowed_record_documents,
                        fallback=[item.get("source_document") for item in retrieved_record_sections],
                    ),
                )
                for evidence in normalized_evidence
            ],
        }
    )
    return verify_finding_against_retrieved_sections(
        finding=normalized,
        retrieved_sections=retrieved_record_sections,
    )


def normalize_requirement_linked_row(
    *,
    finding: ComplianceFinding,
    row: ComplianceLinkedRow | None,
) -> ComplianceLinkedRow:
    return ComplianceLinkedRow(
        requirement_ref=row.requirement_ref if row and row.requirement_ref else "",
        requirement=row.requirement if row and row.requirement else finding.requirement,
        status=finding.status,
        rationale=row.rationale if row and row.rationale else "",
        gap="",
        recommendation="",
    )


def compute_completion_percent(findings: list[ComplianceFinding]) -> int:
    if not findings:
        return 0
    total_score = sum(STATUS_COMPLETION_SCORES.get(finding.status, 0) for finding in findings)
    return round(total_score / len(findings))


def compute_weighted_completion_percent(findings: list[ComplianceFinding]) -> int:
    if not findings:
        return 0
    total_weight = sum(float(finding.weight or 0.0) for finding in findings)
    if total_weight <= 0:
        return compute_completion_percent(findings)
    total_score = sum(
        STATUS_COMPLETION_SCORES.get(finding.status, 0) * float(finding.weight or 0.0)
        for finding in findings
    )
    return round(total_score / total_weight)


def compute_overall_coverage_percent(findings: list[ComplianceFinding]) -> int:
    if not findings:
        return 0
    return round(
        sum(int(finding.requirement_coverage_percent or 0) for finding in findings) / len(findings)
    )


def compute_weighted_coverage_percent(findings: list[ComplianceFinding]) -> int:
    if not findings:
        return 0
    total_weight = sum(float(finding.weight or 0.0) for finding in findings)
    if total_weight <= 0:
        return compute_overall_coverage_percent(findings)
    total_score = sum(
        int(finding.requirement_coverage_percent or 0) * float(finding.weight or 0.0)
        for finding in findings
    )
    return round(total_score / total_weight)


def compute_average_evidence_strength(findings: list[ComplianceFinding]) -> float:
    if not findings:
        return 0.0
    return round(
        sum(float(finding.evidence_strength or 0.0) for finding in findings) / len(findings),
        4,
    )


def compute_weighted_average_evidence_strength(findings: list[ComplianceFinding]) -> float:
    if not findings:
        return 0.0
    total_weight = sum(float(finding.weight or 0.0) for finding in findings)
    if total_weight <= 0:
        return compute_average_evidence_strength(findings)
    total_score = sum(
        float(finding.evidence_strength or 0.0) * float(finding.weight or 0.0)
        for finding in findings
    )
    return round(total_score / total_weight, 4)


def apply_computed_analysis_metrics(analysis: ComplianceAnalysis) -> ComplianceAnalysis:
    findings = analysis.procedure_to_record or analysis.findings
    completion_percent = compute_completion_percent(findings)
    weighted_completion_percent = compute_weighted_completion_percent(findings)
    overall_coverage_percent = compute_overall_coverage_percent(findings)
    weighted_coverage_percent = compute_weighted_coverage_percent(findings)
    average_evidence_strength = compute_average_evidence_strength(findings)
    weighted_average_evidence_strength = compute_weighted_average_evidence_strength(findings)
    return analysis.model_copy(
        update={
            "completion_percent": completion_percent,
            "weighted_completion_percent": weighted_completion_percent,
            "overall_coverage_percent": overall_coverage_percent,
            "weighted_coverage_percent": weighted_coverage_percent,
            "average_evidence_strength": average_evidence_strength,
            "weighted_average_evidence_strength": weighted_average_evidence_strength,
        }
    )


def build_linked_rows_from_findings(
    findings: list[ComplianceFinding],
    existing_rows: list[ComplianceLinkedRow] | None = None,
) -> list[ComplianceLinkedRow]:
    rows = existing_rows or []
    resolved_rows: list[ComplianceLinkedRow] = []
    for index, finding in enumerate(findings):
        existing_row = rows[index] if index < len(rows) else None
        resolved_rows.append(
            ComplianceLinkedRow(
                requirement_ref=f"REQ-{index + 1}",
                requirement=(existing_row.requirement if existing_row and existing_row.requirement else finding.requirement),
                status=finding.status,
                rationale=(existing_row.rationale if existing_row and existing_row.rationale else ""),
                record_recall_at_k=(existing_row.record_recall_at_k if existing_row else None),
                gap="",
                recommendation="",
            )
        )
    return resolved_rows


def assemble_compliance_analysis(
    *,
    findings: list[ComplianceFinding],
    linked_rows: list[ComplianceLinkedRow] | None = None,
) -> ComplianceAnalysis:
    resolved_rows = build_linked_rows_from_findings(findings, existing_rows=linked_rows)
    analysis = ComplianceAnalysis(
        completion_percent=0,
        weighted_completion_percent=0,
        overall_coverage_percent=0,
        weighted_coverage_percent=0,
        average_evidence_strength=0.0,
        weighted_average_evidence_strength=0.0,
        linked_rows=resolved_rows,
        findings=findings,
        procedure_to_record=findings,
        gaps=[],
        recommended_actions=[],
    )
    return apply_computed_analysis_metrics(analysis)


def recommended_action_for_requirement(requirement: str) -> str:
    text = " ".join(str(requirement or "").split())
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
            return sanitize_compliance_recommendation(rewritten[0].upper() + rewritten[1:])
    return sanitize_compliance_recommendation(f"Address requirement: {text}")


def validate_analysis_cardinality(
    *,
    analysis: ComplianceAnalysis,
    expected_count: int,
    method: str,
) -> None:
    if expected_count <= 0:
        return
    actual_count = len(analysis.procedure_to_record or analysis.findings)
    if actual_count != expected_count:
        raise LLMGenerationError(
            f"{method} returned {actual_count} requirement evaluations; expected {expected_count}."
        )


def verify_finding_against_retrieved_sections(
    *,
    finding: ComplianceFinding,
    retrieved_sections: list[dict[str, Any]],
) -> ComplianceFinding:
    supported_evidence, evidence_breadth = _collect_supported_evidence(
        evidence_list=finding.evidence,
        sections=retrieved_sections,
    )
    next_source = finding.source_document
    if not supported_evidence:
        next_source = ""
    verified = finding.model_copy(
        update={
            "evidence": supported_evidence,
            "source_document": next_source,
            "evidence_breadth": evidence_breadth,
            "evidence_items": [
                ComplianceEvidenceItem(
                    text=evidence,
                    source_document=next_source,
                )
                for evidence in supported_evidence
            ],
        }
    )
    return apply_evidence_contradiction_verification(_apply_evidence_thresholds(verified))


def verify_finding_against_full_record_sections(
    *,
    finding: ComplianceFinding,
    record_sections: list[dict[str, Any]],
) -> ComplianceFinding:
    supported_evidence, evidence_breadth = _collect_supported_evidence(
        evidence_list=finding.evidence,
        sections=record_sections,
    )
    next_source = finding.source_document
    if not supported_evidence:
        next_source = ""
    verified = finding.model_copy(
        update={
            "evidence": supported_evidence,
            "source_document": next_source,
            "evidence_breadth": evidence_breadth,
            "evidence_items": [
                ComplianceEvidenceItem(
                    text=evidence,
                    source_document=next_source,
                )
                for evidence in supported_evidence
            ],
        }
    )
    return apply_evidence_contradiction_verification(_apply_evidence_thresholds(verified))


def evidence_supported_by_sections(evidence: str, sections: list[dict[str, Any]]) -> bool:
    return bool(_matching_sections_for_evidence(evidence, sections))


def _collect_supported_evidence(
    *,
    evidence_list: list[str],
    sections: list[dict[str, Any]],
) -> tuple[list[str], int]:
    supported_evidence: list[str] = []
    section_keys: set[str] = set()
    for evidence in evidence_list:
        matches = _matching_sections_for_evidence(evidence, sections)
        if not matches:
            continue
        supported_evidence.append(evidence)
        section_keys.update(_section_support_key(section) for section in matches)
    section_keys.discard("")
    return supported_evidence, len(section_keys)


def _matching_sections_for_evidence(evidence: str, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_evidence = normalize_whitespace(evidence).lower()
    if not normalized_evidence:
        return []
    evidence_tokens = re.findall(r"[a-z0-9]+", normalized_evidence)
    if not evidence_tokens:
        return []

    matches: list[dict[str, Any]] = []
    for section in sections:
        section_text = _substantive_section_text(section)
        if not section_text:
            continue
        if normalized_evidence in section_text:
            matches.append(section)
            continue
        section_tokens = set(re.findall(r"[a-z0-9]+", section_text))
        overlap = sum(1 for token in evidence_tokens if token in section_tokens)
        min_overlap = min(len(evidence_tokens), 4)
        if overlap >= min_overlap and (overlap / len(evidence_tokens)) >= 0.7:
            matches.append(section)
    return matches


def normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        normalized = normalize_whitespace(value)
        return [normalized] if normalized else []
    if not isinstance(value, list):
        return []
    return [normalize_whitespace(str(item)) for item in value if normalize_whitespace(str(item))]


def _substantive_section_text(section: dict[str, Any]) -> str:
    body_text = normalize_whitespace(section.get("text") or "")
    table_text = normalize_whitespace(section.get("table_markdown") or "")
    if table_text:
        return normalize_whitespace(" ".join(part for part in (body_text, table_text) if part)).lower()
    if not body_text:
        return ""
    body_tokens = re.findall(r"[a-z0-9]+", body_text.lower())
    if (
        len(body_text) < MIN_SUBSTANTIVE_SECTION_TEXT_LENGTH
        and len(body_tokens) < MIN_SUBSTANTIVE_SECTION_TOKEN_COUNT
    ):
        return ""
    return body_text.lower()


def _section_support_key(section: dict[str, Any]) -> str:
    source_document = normalize_whitespace(section.get("source_document") or "")
    section_label = normalize_whitespace(section.get("section_label") or "")
    heading_title = normalize_whitespace(section.get("heading_title") or "")
    for value in (section_label, heading_title):
        if value:
            return f"{source_document}::{value}" if source_document else value
    return source_document


def sanitize_source_document(
    value: Any,
    *,
    allowed: set[str],
    fallback: list[Any],
) -> str:
    requested = normalize_string_list(value)
    sanitized = [item for item in requested if item in allowed]
    if sanitized:
        return sanitized[0]
    fallback_sanitized = [
        normalize_whitespace(str(item))
        for item in fallback
        if normalize_whitespace(str(item)) in allowed
    ]
    deduped = list(dict.fromkeys(fallback_sanitized))
    return deduped[0] if deduped else ""


def sanitize_compliance_recommendation(value: str) -> str:
    text = normalize_whitespace(value)
    if not text:
        return ""
    replacements = [
        ("procedure deliverables", "record documentation"),
        ("procedure deliverable", "record document"),
        ("procedure documents", "record documents"),
        ("procedure document", "record document"),
        ("procedure text", "record documentation"),
    ]
    sanitized = text
    for old, new in replacements:
        sanitized = re.sub(old, new, sanitized, flags=re.IGNORECASE)
    return sanitized


def _normalize_analysis_source_document(
    *,
    analysis: ComplianceAnalysis,
    allowed_record_documents: set[str],
) -> ComplianceAnalysis:
    normalized_procedure_to_record = [
        _normalize_finding_source_document(
            finding=finding,
            allowed_record_documents=allowed_record_documents,
        )
        for finding in analysis.procedure_to_record
    ]
    normalized_findings = [
        _normalize_finding_source_document(
            finding=finding,
            allowed_record_documents=allowed_record_documents,
        )
        for finding in analysis.findings
    ]
    return analysis.model_copy(
        update={
            "procedure_to_record": normalized_procedure_to_record,
            "findings": normalized_findings,
        }
    )


def _normalize_finding_source_document(
    *,
    finding: ComplianceFinding,
    allowed_record_documents: set[str],
) -> ComplianceFinding:
    normalized_evidence = normalize_string_list(finding.evidence)
    if not normalized_evidence:
        normalized = finding.model_copy(
            update={
                "evidence": [],
                "source_document": "",
                "evidence_items": [],
            }
        )
        return _apply_evidence_thresholds(normalized)
    normalized = finding.model_copy(
        update={
            "evidence": normalized_evidence,
            "source_document": sanitize_source_document(
                finding.source_document,
                allowed=allowed_record_documents,
                fallback=[],
            ),
            "evidence_items": [
                ComplianceEvidenceItem(
                    text=evidence,
                    source_document=sanitize_source_document(
                        finding.source_document,
                        allowed=allowed_record_documents,
                        fallback=[],
                    ),
                )
                for evidence in normalized_evidence
            ],
        }
    )
    return apply_evidence_contradiction_verification(_apply_evidence_thresholds(normalized))


def _apply_evidence_thresholds(finding: ComplianceFinding) -> ComplianceFinding:
    evidence = normalize_string_list(finding.evidence)
    source_document = normalize_whitespace(finding.source_document)
    evidence_items = (
        [
            item
            for item in finding.evidence_items
            if normalize_whitespace(item.text)
        ]
        if finding.evidence_items
        else [
            ComplianceEvidenceItem(
                text=text,
                source_document=source_document,
            )
            for text in evidence
        ]
    )
    grounded_count = sum(
        1
        for item in evidence_items
        if (
            normalize_whitespace(item.text)
            and normalize_whitespace(item.source_document)
            and item.supports_requirement
        )
    )

    next_status = _derive_status_from_grounded_evidence(
        requirement=finding.requirement,
        grounded_count=grounded_count,
        has_source_document=bool(source_document),
    )

    return finding.model_copy(
        update={
            "llm_status": finding.llm_status or next_status,
            "nli_status": finding.nli_status or next_status,
            "final_metric_status": finding.final_metric_status or next_status,
            "pre_verification_status": finding.pre_verification_status or next_status,
            "status": next_status,
            "evidence": evidence,
            "source_document": source_document,
            "evidence_items": evidence_items,
        }
    )


def _derive_status_from_grounded_evidence(
    *,
    requirement: str,
    grounded_count: int,
    has_source_document: bool,
) -> str:
    if grounded_count <= 0 or not has_source_document:
        return "not_satisfied"
    required_evidence_items = (
        MIN_STRONG_CLAIM_EVIDENCE_ITEMS
        if _requires_multiple_citations(requirement)
        else MIN_SATISFIED_EVIDENCE_ITEMS
    )
    if grounded_count >= required_evidence_items:
        return "satisfied"
    return "partial"


def _requires_multiple_citations(requirement: str) -> bool:
    text = normalize_whitespace(requirement).lower()
    if not text:
        return False
    markers = (
        " both ",
        " each ",
        " all ",
        " including ",
        " include ",
        " as well as ",
    )
    padded = f" {text} "
    return any(marker in padded for marker in markers)


def flatten_record_sections(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        parsed_json = record.get("parsed_json") or {}
        source_document = (
            record.get("source_filename")
            or record.get("stored_filename")
            or parsed_json.get("source_filename")
            or parsed_json.get("stored_filename")
            or ""
        )
        flattened.extend(
            _flatten_record_sections(
                parsed_json.get("sections", []),
                source_document=str(source_document),
            )
        )
    return flattened


def _flatten_record_sections(
    sections: list[dict[str, Any]],
    *,
    source_document: str,
    parent_headings: list[str] | None = None,
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    lineage = parent_headings or []
    for section in sections:
        if not isinstance(section, dict):
            continue
        heading_title = normalize_whitespace(section.get("heading_title") or "")
        headings = [*lineage, heading_title] if heading_title else [*lineage]
        combined_heading = " / ".join(part for part in headings if part)
        flattened.append(
            {
                "source_document": source_document,
                "section_label": section.get("section_label"),
                "heading_title": combined_heading or section.get("heading_title") or "",
                "text": section.get("text"),
                "table_markdown": section.get("table_markdown"),
            }
        )
        tables = section.get("tables", [])
        if isinstance(tables, list):
            for table in tables:
                if isinstance(table, dict) and table.get("markdown"):
                    flattened.append(
                        {
                            "source_document": source_document,
                            "section_label": section.get("section_label"),
                            "heading_title": combined_heading or section.get("heading_title") or "",
                            "text": "",
                            "table_markdown": table.get("markdown"),
                        }
                    )
        flattened.extend(
            _flatten_record_sections(
                section.get("subsections", []),
                source_document=source_document,
                parent_headings=headings,
            )
        )
    return flattened


def _sanitize_linked_row_recommendation(row: ComplianceLinkedRow) -> ComplianceLinkedRow:
    return row.model_copy(
        update={
            "recommendation": sanitize_compliance_recommendation(row.recommendation),
        }
    )


def _build_linked_rows(analysis: ComplianceAnalysis) -> list[ComplianceLinkedRow]:
    linked_rows: list[ComplianceLinkedRow] = []
    for index, finding in enumerate(analysis.procedure_to_record):
        linked_rows.append(
            ComplianceLinkedRow(
                requirement_ref=f"REQ-{index + 1}",
                requirement=finding.requirement,
                status=finding.status,
                rationale="",
                gap="",
                recommendation="",
            )
        )
    return linked_rows


def _resolve_linked_row_requirements(analysis: ComplianceAnalysis) -> list[ComplianceLinkedRow]:
    requirement_map = {
        f"REQ-{index + 1}": finding.requirement
        for index, finding in enumerate(analysis.procedure_to_record)
    }
    resolved_rows: list[ComplianceLinkedRow] = []
    for row in analysis.linked_rows:
        requirement = row.requirement
        if not requirement and row.requirement_ref:
            requirement = requirement_map.get(row.requirement_ref, "")
        resolved_rows.append(
            row.model_copy(
                update={
                    "requirement": requirement,
                }
            )
        )
    return resolved_rows
