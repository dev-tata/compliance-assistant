from __future__ import annotations

from typing import Any

from app.schemas.compliance import (
    ComplianceAnalysis,
    ComplianceEvidenceItem,
    ComplianceFinding,
    ComplianceLinkedRow,
    ComplianceRequest,
    ComplianceResponse,
    ComplianceStageResult,
)
from app.services.compliance_methods.compliance_method_common import (
    apply_computed_overall_assessment,
    assemble_compliance_analysis,
    build_compliance_prompt,
    build_compliance_result_path,
    normalize_requirement_finding,
    normalize_requirement_linked_row,
    parse_compliance_analysis_response,
    serialize_deliverable_for_prompt,
    serialize_retrieved_section,
    simplify_document_for_prompt,
)
from app.services.compliance_methods.record_retrieval_stage_service import (
    RECORD_FINAL_TOP_K,
    apply_row_level_record_recall,
    build_record_retrieval_payload,
    compute_record_recall_at_k,
)
from app.services.compliance_scoring_service import compute_scores, enrich_analysis_for_scoring
from app.services.document_service import current_timestamp
from app.services.llm.errors import LLMGenerationError
from app.services.llm.factory import get_llm_service
from app.services.retrieval.faiss_retrieval import FAISS_TOP_K, RERANK_TOP_K, normalize_whitespace
from app.services.retrieval.record_index_service import prepare_record_indexes
from app.services.retrieval.reference_index_service import (
    prepare_reference_indexes,
    search_prepared_reference_indexes,
)

CONTEXT_TOP_K = FAISS_TOP_K
CONTEXT_FINAL_TOP_K = RERANK_TOP_K
STAGE_1_KEY = "stage_1_non_rag"
STAGE_2_KEY = "stage_2_record_retrieval"
STAGE_3_KEY = "stage_3_reference_retrieval"
STAGE_LABELS = {
    STAGE_1_KEY: "Stage 1 - Non-RAG",
    STAGE_2_KEY: "Stage 2 - Record Retrieval",
    STAGE_3_KEY: "Stage 3 - Reference Retrieval",
}


def run_two_stage_rag_compliance(
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
            "two_stage_rag requires extracted deliverables. Generate or select deliverables first."
        )
    if not case_payload.get("references", []):
        raise LLMGenerationError("two_stage_rag requires reference documents for requirement context.")

    llm_service = get_llm_service(request.provider, request.model)
    prepared_record_indexes = prepare_record_indexes(case_payload.get("records", []))
    if not prepared_record_indexes:
        raise LLMGenerationError("two_stage_rag requires retrievable record sections.")
    reference_context_indexes = prepare_reference_indexes(case_payload.get("references", []))
    if not reference_context_indexes:
        raise LLMGenerationError("two_stage_rag requires retrievable reference sections.")

    allowed_record_documents = {
        normalize_whitespace(document.get("source_filename") or document.get("stored_filename"))
        for document in case_payload.get("records", [])
    }

    retrieved_payload = build_record_retrieval_payload(
        deliverables=deliverables,
        prepared_record_indexes=prepared_record_indexes,
    )

    stage_1 = _run_stage_1_non_rag(
        llm_service=llm_service,
        case_payload=case_payload,
        deliverables=deliverables,
        instructions=request.instructions,
        allowed_record_documents=allowed_record_documents,
    )
    stage_2 = _run_stage_2_record_retrieval(
        llm_service=llm_service,
        deliverables=deliverables,
        previous_analysis=stage_1.analysis,
        instructions=request.instructions,
        allowed_record_documents=allowed_record_documents,
        retrieved_payload=retrieved_payload,
    )
    retrieved_payload = _attach_stage_2_reference_context(
        deliverables=deliverables,
        stage_2_analysis=stage_2.analysis,
        retrieved_payload=retrieved_payload,
        reference_context_indexes=reference_context_indexes,
    )
    stage_3 = _run_stage_3_reference_retrieval(
        llm_service=llm_service,
        deliverables=deliverables,
        previous_analysis=stage_2.analysis,
        instructions=request.instructions,
        allowed_record_documents=allowed_record_documents,
        retrieved_payload=retrieved_payload,
    )

    saved_path = build_compliance_result_path(case_id)
    response = ComplianceResponse(
        case_id=case_id,
        compliance_provider=request.provider,
        compliance_model=request.model,
        extraction_provider=case_payload.get("extraction_provider"),
        extraction_model=case_payload.get("extraction_model"),
        method="two_stage_rag",
        reference_stored_filenames=[
            item.get("stored_filename")
            for item in case_payload.get("references", [])
            if item.get("stored_filename")
        ],
        created_at=current_timestamp(),
        saved_at=saved_path.as_posix(),
        analysis=stage_3.analysis,
        scores=stage_3.scores,
        section_matches=[],
        retrieval_metrics=stage_3.retrieval_metrics,
        stages=[stage_1, stage_2, stage_3],
        baseline_method=stage_1.method,
        baseline_analysis=stage_1.analysis,
        baseline_scores=stage_1.scores,
        baseline_retrieval_metrics=stage_1.retrieval_metrics,
    )
    saved_path.write_text(
        response.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )
    return response


def _extract_deliverable_weights(deliverables: list[dict[str, Any]]) -> list[float]:
    weights: list[float] = []
    for item in deliverables:
        raw_weight = item.get("weight")
        if isinstance(raw_weight, (int, float)) and raw_weight > 0:
            weights.append(float(raw_weight))
        else:
            weights.append(1.0)
    return weights


def _run_stage_1_non_rag(
    *,
    llm_service: Any,
    case_payload: dict[str, Any],
    deliverables: list[dict[str, Any]],
    instructions: str | None,
    allowed_record_documents: set[str],
) -> ComplianceStageResult:
    prompt = build_compliance_prompt(
        method="non_rag",
        payload={
            "method": "non_rag",
            "requirement_source": "deliverables",
            "case_id": case_payload.get("case_id"),
            "title": case_payload.get("title"),
            "notes": case_payload.get("notes"),
            "deliverables": [serialize_deliverable_for_prompt(item) for item in deliverables],
            "records": [
                simplify_document_for_prompt(document) for document in case_payload.get("records", [])
            ],
        },
        instructions=instructions,
        include_feedback=False,
    )
    analysis = parse_compliance_analysis_response(
        raw_analysis=llm_service.generate(prompt, temperature=0.0),
        method="non_rag",
        expected_count=len(deliverables),
        allowed_record_documents=allowed_record_documents,
    )
    analysis = enrich_analysis_for_scoring(
        analysis,
        requirement_weights=_extract_deliverable_weights(deliverables),
    )
    analysis = apply_computed_overall_assessment(analysis)
    analysis = _tag_analysis_evidence(
        analysis=analysis,
        stage_key=STAGE_1_KEY,
        stage_label=STAGE_LABELS[STAGE_1_KEY],
    )
    analysis = _strip_stage_feedback(analysis)
    return ComplianceStageResult(
        stage_key=STAGE_1_KEY,
        stage_label=STAGE_LABELS[STAGE_1_KEY],
        method="non_rag",
        analysis=analysis,
        scores=compute_scores(analysis),
        retrieval_metrics=None,
    )


def _run_stage_2_record_retrieval(
    *,
    llm_service: Any,
    deliverables: list[dict[str, Any]],
    previous_analysis: ComplianceAnalysis,
    instructions: str | None,
    allowed_record_documents: set[str],
    retrieved_payload: list[dict[str, Any]],
) -> ComplianceStageResult:
    prompt = build_compliance_prompt(
        method="record_retrieval_stage",
        payload={
            "method": "record_retrieval_stage",
            "requirement_source": "deliverables",
            "requirement_evaluations": _build_stage_evaluations(
                deliverables=deliverables,
                baseline_analysis=previous_analysis,
                retrieved_payload=retrieved_payload,
                include_reference_context=False,
                include_feedback=False,
            ),
            "workflow_rules": [
                "Start from the provided baseline assessment for each requirement.",
                "Treat the original deliverable as the canonical requirement source.",
                "Use retrieved_record_sections only as admissible evidence.",
                "Keep or improve each baseline assessment, but do not downgrade it.",
            ],
        },
        instructions=instructions,
        include_feedback=False,
    )
    raw_analysis = parse_compliance_analysis_response(
        raw_analysis=llm_service.generate(prompt, temperature=0.0),
        method="record_retrieval_stage",
        expected_count=len(deliverables),
    )
    analysis = _merge_stage_analysis(
        candidate_analysis=raw_analysis,
        previous_analysis=previous_analysis,
        retrieved_payload=retrieved_payload,
        allowed_record_documents=allowed_record_documents,
        stage_key=STAGE_2_KEY,
        stage_label=STAGE_LABELS[STAGE_2_KEY],
    )
    analysis = _strip_stage_feedback(analysis)
    retrieval_metrics = compute_record_recall_at_k(
        findings=analysis.procedure_to_record or analysis.findings,
        retrieved_payload=retrieved_payload,
        k=RECORD_FINAL_TOP_K,
    )
    return ComplianceStageResult(
        stage_key=STAGE_2_KEY,
        stage_label=STAGE_LABELS[STAGE_2_KEY],
        method="record_retrieval_stage",
        analysis=analysis,
        scores=compute_scores(analysis),
        retrieval_metrics=retrieval_metrics,
    )


def _run_stage_3_reference_retrieval(
    *,
    llm_service: Any,
    deliverables: list[dict[str, Any]],
    previous_analysis: ComplianceAnalysis,
    instructions: str | None,
    allowed_record_documents: set[str],
    retrieved_payload: list[dict[str, Any]],
) -> ComplianceStageResult:
    prompt = build_compliance_prompt(
        method="two_stage_rag",
        payload={
            "method": "two_stage_rag",
            "requirement_source": "deliverables",
            "requirement_evaluations": _build_stage_evaluations(
                deliverables=deliverables,
                baseline_analysis=previous_analysis,
                retrieved_payload=retrieved_payload,
                include_reference_context=True,
                include_feedback=False,
            ),
            "workflow_rules": [
                "Start from the provided baseline assessment for each requirement.",
                "Treat the original deliverable as the canonical requirement source.",
                "Use retrieved_record_sections only as admissible compliance evidence.",
                "Use retrieved_requirement_context only to interpret the requirement, never as compliance evidence.",
                "Keep or improve each baseline assessment, but do not downgrade it.",
            ],
        },
        instructions=instructions,
    )
    raw_analysis = parse_compliance_analysis_response(
        raw_analysis=llm_service.generate(prompt, temperature=0.0),
        method="two_stage_rag",
        expected_count=len(deliverables),
    )
    analysis = _merge_stage_analysis(
        candidate_analysis=raw_analysis,
        previous_analysis=previous_analysis,
        retrieved_payload=retrieved_payload,
        allowed_record_documents=allowed_record_documents,
        stage_key=STAGE_3_KEY,
        stage_label=STAGE_LABELS[STAGE_3_KEY],
    )
    retrieval_metrics = compute_record_recall_at_k(
        findings=analysis.procedure_to_record or analysis.findings,
        retrieved_payload=retrieved_payload,
        k=RECORD_FINAL_TOP_K,
    )
    return ComplianceStageResult(
        stage_key=STAGE_3_KEY,
        stage_label=STAGE_LABELS[STAGE_3_KEY],
        method="two_stage_rag",
        analysis=analysis,
        scores=compute_scores(analysis),
        retrieval_metrics=retrieval_metrics,
    )


def _build_stage_evaluations(
    *,
    deliverables: list[dict[str, Any]],
    baseline_analysis: ComplianceAnalysis,
    retrieved_payload: list[dict[str, Any]],
    include_reference_context: bool,
    include_feedback: bool,
) -> list[dict[str, Any]]:
    findings = baseline_analysis.procedure_to_record or baseline_analysis.findings
    rows = baseline_analysis.linked_rows
    evaluations: list[dict[str, Any]] = []
    for index, deliverable in enumerate(deliverables):
        finding = findings[index] if index < len(findings) else None
        row = rows[index] if index < len(rows) else None
        payload = retrieved_payload[index] if index < len(retrieved_payload) else {}
        item = {
            "requirement_ref": f"REQ-{index + 1}",
            "deliverable": serialize_deliverable_for_prompt(deliverable),
            "baseline_assessment": {
                "status": finding.status if finding else "not_satisfied",
                "evidence": finding.evidence if finding else [],
                "source_documents": finding.source_documents if finding else [],
                "gap": row.gap if include_feedback and row else "",
                "recommendation": row.recommendation if include_feedback and row else "",
            },
            "retrieved_record_sections": payload.get("retrieved_record_sections", []),
        }
        if include_reference_context:
            item["retrieved_requirement_context"] = payload.get("retrieved_requirement_context", [])
        evaluations.append(item)
    return evaluations


def _merge_stage_analysis(
    *,
    candidate_analysis: ComplianceAnalysis,
    previous_analysis: ComplianceAnalysis,
    retrieved_payload: list[dict[str, Any]],
    allowed_record_documents: set[str],
    stage_key: str,
    stage_label: str,
) -> ComplianceAnalysis:
    previous_findings = previous_analysis.procedure_to_record or previous_analysis.findings
    previous_rows = previous_analysis.linked_rows
    candidate_findings = candidate_analysis.procedure_to_record or candidate_analysis.findings
    candidate_rows = candidate_analysis.linked_rows

    merged_findings: list[ComplianceFinding] = []
    merged_rows: list[ComplianceLinkedRow] = []
    for index, previous_finding in enumerate(previous_findings):
        candidate_finding = candidate_findings[index] if index < len(candidate_findings) else previous_finding
        previous_row = previous_rows[index] if index < len(previous_rows) else None
        candidate_row = candidate_rows[index] if index < len(candidate_rows) else None
        retrieved_sections = retrieved_payload[index].get("retrieved_record_sections", [])

        normalized_candidate = normalize_requirement_finding(
            finding=candidate_finding,
            retrieved_record_sections=retrieved_sections,
            allowed_record_documents=allowed_record_documents,
        )
        if not _is_requirement_improved(
            previous_finding=previous_finding,
            candidate_finding=normalized_candidate,
        ):
            merged_finding = previous_finding
            merged_row = normalize_requirement_linked_row(
                finding=merged_finding,
                row=_select_merged_row(
                    previous_row=previous_row,
                    candidate_row=candidate_row,
                    stage_key=stage_key,
                ),
            )
        else:
            merged_finding = _merge_finding_evidence(
                candidate_finding=normalized_candidate,
                previous_finding=previous_finding,
                stage_key=stage_key,
                stage_label=stage_label,
            )
            merged_row = normalize_requirement_linked_row(
                finding=merged_finding,
                row=candidate_row if candidate_row is not None else previous_row,
            )
        merged_findings.append(merged_finding)
        merged_rows.append(merged_row)

    analysis = assemble_compliance_analysis(
        findings=merged_findings,
        linked_rows=apply_row_level_record_recall(
            linked_rows=merged_rows,
            findings=merged_findings,
            retrieved_payload=retrieved_payload,
        ),
    )
    analysis = enrich_analysis_for_scoring(
        analysis,
        requirement_weights=_extract_deliverable_weights(deliverables),
    )
    return apply_computed_overall_assessment(analysis)


def _attach_stage_2_reference_context(
    *,
    deliverables: list[dict[str, Any]],
    stage_2_analysis: ComplianceAnalysis,
    retrieved_payload: list[dict[str, Any]],
    reference_context_indexes: list[tuple[Any, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    findings = stage_2_analysis.procedure_to_record or stage_2_analysis.findings
    rows = stage_2_analysis.linked_rows
    enriched_payload: list[dict[str, Any]] = []
    for index, payload in enumerate(retrieved_payload):
        deliverable = deliverables[index] if index < len(deliverables) else {}
        finding = findings[index] if index < len(findings) else None
        row = rows[index] if index < len(rows) else None
        retrieved_record_sections = payload.get("retrieved_record_sections", [])
        retrieved_requirement_context = [
            serialize_retrieved_section(section)
            for section in _search_requirement_context(
                reference_context_indexes=reference_context_indexes,
                query_text=_build_stage_3_reference_query(
                    deliverable=deliverable,
                    stage_2_finding=finding,
                    retrieved_record_sections=retrieved_record_sections,
                ),
                top_k=CONTEXT_TOP_K,
                final_top_k=CONTEXT_FINAL_TOP_K,
            )
        ]
        enriched_payload.append(
            {
                **payload,
                "retrieved_requirement_context": retrieved_requirement_context,
            }
        )
    return enriched_payload


def _tag_analysis_evidence(
    *,
    analysis: ComplianceAnalysis,
    stage_key: str,
    stage_label: str,
) -> ComplianceAnalysis:
    findings = analysis.procedure_to_record or analysis.findings
    tagged = [
        finding.model_copy(
            update={
                "evidence_items": [
                    ComplianceEvidenceItem(
                        text=evidence,
                        source_documents=finding.source_documents,
                        stage_key=stage_key,
                        stage_label=stage_label,
                    )
                    for evidence in finding.evidence
                ]
            }
        )
        for finding in findings
    ]
    return analysis.model_copy(update={"procedure_to_record": tagged, "findings": tagged})


def _tag_finding_evidence(
    *,
    finding: ComplianceFinding,
    previous_finding: ComplianceFinding,
    stage_key: str,
    stage_label: str,
) -> ComplianceFinding:
    inherited = _build_evidence_item_lookup(previous_finding.evidence_items)
    evidence_items: list[ComplianceEvidenceItem] = []
    for evidence in finding.evidence:
        normalized = _normalize_evidence_key(evidence)
        existing = inherited.get(normalized)
        if existing is not None:
            evidence_items.append(
                existing.model_copy(
                    update={
                        "source_documents": finding.source_documents or existing.source_documents,
                    }
                )
            )
        else:
            evidence_items.append(
                ComplianceEvidenceItem(
                    text=evidence,
                    source_documents=finding.source_documents,
                    stage_key=stage_key,
                    stage_label=stage_label,
                )
            )
    return finding.model_copy(update={"evidence_items": evidence_items})


def _merge_finding_evidence(
    *,
    candidate_finding: ComplianceFinding,
    previous_finding: ComplianceFinding,
    stage_key: str,
    stage_label: str,
) -> ComplianceFinding:
    tagged_candidate = _tag_finding_evidence(
        finding=candidate_finding,
        previous_finding=previous_finding,
        stage_key=stage_key,
        stage_label=stage_label,
    )
    evidence_by_key = _build_evidence_item_lookup(previous_finding.evidence_items)
    ordered_keys = [
        _normalize_evidence_key(item.text)
        for item in previous_finding.evidence_items
        if _normalize_evidence_key(item.text)
    ]

    for item in tagged_candidate.evidence_items:
        key = _normalize_evidence_key(item.text)
        if not key:
            continue
        existing = evidence_by_key.get(key)
        if existing is None:
            ordered_keys.append(key)
            evidence_by_key[key] = item
            continue
        evidence_by_key[key] = existing.model_copy(
            update={
                "source_documents": _merge_string_lists(
                    existing.source_documents,
                    item.source_documents,
                ),
                "stage_key": existing.stage_key or item.stage_key,
                "stage_label": existing.stage_label or item.stage_label,
            }
        )

    merged_items = [evidence_by_key[key] for key in ordered_keys if key in evidence_by_key]
    merged_sources = _merge_string_lists(
        previous_finding.source_documents,
        tagged_candidate.source_documents,
    )
    for item in merged_items:
        merged_sources = _merge_string_lists(merged_sources, item.source_documents)

    return tagged_candidate.model_copy(
        update={
            "evidence": [item.text for item in merged_items],
            "source_documents": merged_sources,
            "evidence_items": merged_items,
        }
    )


def _search_requirement_context(
    *,
    reference_context_indexes: list[tuple[Any, list[dict[str, Any]]]],
    query_text: str,
    top_k: int,
    final_top_k: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if reference_context_indexes:
        results.extend(
            search_prepared_reference_indexes(
                prepared_indexes=reference_context_indexes,
                query_text=query_text,
                top_k=top_k,
                final_top_k=final_top_k,
            )
        )
    return results[: max(final_top_k, 1)]


def _build_reference_query(
    *,
    deliverable: dict[str, Any],
    retrieved_record_sections: list[dict[str, Any]],
) -> str:
    record_context = " ".join(
        normalize_whitespace(
            " ".join(
                part
                for part in (
                    section.get("heading_title"),
                    section.get("section_label"),
                    section.get("source_document"),
                )
                if part
            )
        )
        for section in retrieved_record_sections[: max(RECORD_FINAL_TOP_K, 1)]
    )
    return normalize_whitespace(
        " ".join(
            part
            for part in (
                deliverable.get("requirement_text", ""),
                deliverable.get("source_quote", ""),
                deliverable.get("heading_title", ""),
                record_context,
            )
            if normalize_whitespace(part)
        )
    )


def _build_stage_3_reference_query(
    *,
    deliverable: dict[str, Any],
    stage_2_finding: ComplianceFinding | None,
    retrieved_record_sections: list[dict[str, Any]],
) -> str:
    return normalize_whitespace(
        " ".join(
            part
            for part in (
                _build_reference_query(
                    deliverable=deliverable,
                    retrieved_record_sections=retrieved_record_sections,
                ),
                stage_2_finding.requirement if stage_2_finding else "",
                " ".join(stage_2_finding.evidence) if stage_2_finding else "",
            )
            if normalize_whitespace(part)
        )
    )


def _strip_stage_feedback(analysis: ComplianceAnalysis) -> ComplianceAnalysis:
    stripped_rows = [
        row.model_copy(
            update={
                "gap": "",
                "recommendation": "",
            }
        )
        for row in analysis.linked_rows
    ]
    return analysis.model_copy(
        update={
            "linked_rows": stripped_rows,
            "gaps": [],
            "recommended_actions": [],
        }
    )


def _status_rank(status: str) -> int:
    order = {
        "not_satisfied": 0,
        "partial": 1,
        "satisfied": 2,
    }
    return order.get(status, -1)


def _is_requirement_improved(
    *,
    previous_finding: ComplianceFinding,
    candidate_finding: ComplianceFinding,
) -> bool:
    previous_rank = _status_rank(previous_finding.status)
    candidate_rank = _status_rank(candidate_finding.status)
    if candidate_rank < previous_rank:
        return False

    previous_evidence_keys = {
        _normalize_evidence_key(item.text)
        for item in previous_finding.evidence_items
        if _normalize_evidence_key(item.text)
    } or {
        _normalize_evidence_key(text)
        for text in previous_finding.evidence
        if _normalize_evidence_key(text)
    }
    candidate_evidence_keys = {
        _normalize_evidence_key(text)
        for text in candidate_finding.evidence
        if _normalize_evidence_key(text)
    }

    has_new_citations = bool(candidate_evidence_keys - previous_evidence_keys)
    if candidate_rank > previous_rank:
        return has_new_citations or not previous_evidence_keys
    if not has_new_citations:
        return False

    previous_sources = {normalize_whitespace(item) for item in previous_finding.source_documents if normalize_whitespace(item)}
    candidate_sources = {normalize_whitespace(item) for item in candidate_finding.source_documents if normalize_whitespace(item)}
    if candidate_sources - previous_sources:
        return True

    return has_new_citations


def _select_merged_row(
    *,
    previous_row: ComplianceLinkedRow | None,
    candidate_row: ComplianceLinkedRow | None,
    stage_key: str,
) -> ComplianceLinkedRow | None:
    if stage_key == STAGE_3_KEY:
        return candidate_row if candidate_row is not None else previous_row
    return previous_row if previous_row is not None else candidate_row


def _normalize_evidence_key(text: str) -> str:
    return normalize_whitespace(text)


def _build_evidence_item_lookup(
    evidence_items: list[ComplianceEvidenceItem],
) -> dict[str, ComplianceEvidenceItem]:
    return {
        _normalize_evidence_key(item.text): item
        for item in evidence_items
        if _normalize_evidence_key(item.text)
    }


def _merge_string_lists(*values: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for items in values:
        for item in items:
            normalized = normalize_whitespace(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(item)
    return merged
