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
    annotate_deliverable_structure,
    apply_computed_analysis_metrics,
    assemble_compliance_analysis,
    build_compliance_prompt,
    build_compliance_result_path,
    flatten_record_sections,
    normalize_requirement_finding,
    normalize_requirement_linked_row,
    parse_compliance_analysis_response,
    serialize_deliverable_for_prompt,
    serialize_retrieved_section,
    simplify_document_for_prompt,
    verify_finding_against_full_record_sections,
)
from app.services.compliance_methods.record_retrieval_stage_service import (
    RECORD_FINAL_TOP_K,
    apply_row_level_record_recall,
    build_record_retrieval_payload,
    compute_record_recall_at_k,
)
from app.services.compliance_scoring_service import enrich_analysis_for_scoring
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
    deliverables = annotate_deliverable_structure(deliverables)
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
        instructions=request.instructions,
        allowed_record_documents=allowed_record_documents,
        retrieved_payload=retrieved_payload,
        stage_2_analysis=stage_2.analysis,
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
        section_matches=[],
        retrieval_metrics=stage_3.retrieval_metrics,
        stages=[stage_1, stage_2, stage_3],
        baseline_method=stage_1.method,
        baseline_analysis=stage_1.analysis,
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
    full_record_sections = flatten_record_sections(case_payload.get("records", []))
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
            "workflow_rules": [
                "Treat explicit contradictions inside the record as material non-compliance findings, not as satisfied support.",
                "If a table-derived highest or overall value conflicts with a recorded overall statement, reflect that contradiction directly in the status, rationale, gap, and recommendation.",
                "Do not mark a requirement as satisfied when the record contains grounded evidence that the required value is inconsistent or incorrectly recorded.",
            ],
        },
        instructions=instructions,
        include_feedback=True,
        llm_sets_status=True,
    )
    raw_analysis = llm_service.generate(prompt, temperature=0.0)
    model_analysis = parse_compliance_analysis_response(
        raw_analysis=raw_analysis,
        method="non_rag",
        expected_count=len(deliverables),
        allowed_record_documents=allowed_record_documents,
        preserve_status=True,
    )
    analysis = parse_compliance_analysis_response(
        raw_analysis=raw_analysis,
        method="non_rag",
        expected_count=len(deliverables),
        allowed_record_documents=allowed_record_documents,
    )
    validated_findings = [
        verify_finding_against_full_record_sections(
            finding=finding,
            record_sections=full_record_sections,
        )
        for finding in (analysis.procedure_to_record or analysis.findings)
    ]
    analysis = assemble_compliance_analysis(
        findings=validated_findings,
        linked_rows=analysis.linked_rows,
    )
    analysis = enrich_analysis_for_scoring(
        analysis,
        requirement_weights=_extract_deliverable_weights(deliverables),
        deliverable_metadata=deliverables,
    )
    analysis = _restore_model_judgement_floor(
        candidate_analysis=analysis,
        model_analysis=model_analysis,
    )
    analysis = assemble_compliance_analysis(
        findings=analysis.procedure_to_record or analysis.findings,
        linked_rows=analysis.linked_rows,
    )
    analysis = _tag_analysis_evidence(
        analysis=analysis,
        stage_key=STAGE_1_KEY,
        stage_label=STAGE_LABELS[STAGE_1_KEY],
    )
    return ComplianceStageResult(
        stage_key=STAGE_1_KEY,
        stage_label=STAGE_LABELS[STAGE_1_KEY],
        method="non_rag",
        analysis=analysis,
        retrieval_metrics=None,
    )


def _run_stage_2_record_retrieval(
    *,
    llm_service: Any,
    deliverables: list[dict[str, Any]],
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
                baseline_analysis=None,
                retrieved_payload=retrieved_payload,
                include_reference_context=False,
                include_baseline=False,
                include_feedback=True,
            ),
            "workflow_rules": [
                "Treat the original deliverable as the canonical requirement source.",
                "Use retrieved_record_sections only as admissible evidence.",
                "Evaluate each requirement from scratch using only the deliverable and retrieved_record_sections.",
            ],
        },
        instructions=instructions,
        include_feedback=True,
    )
    raw_analysis = parse_compliance_analysis_response(
        raw_analysis=llm_service.generate(prompt, temperature=0.0),
        method="record_retrieval_stage",
        expected_count=len(deliverables),
    )
    analysis = _normalize_retrieval_stage_analysis(
        candidate_analysis=raw_analysis,
        deliverables=deliverables,
        retrieved_payload=retrieved_payload,
        allowed_record_documents=allowed_record_documents,
        stage_key=STAGE_2_KEY,
        stage_label=STAGE_LABELS[STAGE_2_KEY],
        requirement_weights=_extract_deliverable_weights(deliverables),
    )
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
        retrieval_metrics=retrieval_metrics,
    )


def _run_stage_3_reference_retrieval(
    *,
    llm_service: Any,
    deliverables: list[dict[str, Any]],
    instructions: str | None,
    allowed_record_documents: set[str],
    retrieved_payload: list[dict[str, Any]],
    stage_2_analysis: ComplianceAnalysis,
) -> ComplianceStageResult:
    prompt = build_compliance_prompt(
        method="two_stage_rag",
        payload={
            "method": "two_stage_rag",
            "requirement_source": "deliverables",
            "requirement_evaluations": _build_stage_evaluations(
                deliverables=deliverables,
                baseline_analysis=None,
                retrieved_payload=retrieved_payload,
                include_reference_context=True,
                include_baseline=False,
                include_feedback=True,
            ),
            "workflow_rules": [
                "Treat the original deliverable as the canonical requirement source.",
                "Use retrieved_record_sections only as admissible compliance evidence.",
                "Use retrieved_requirement_context only to interpret the requirement, never as compliance evidence.",
                "Evaluate each requirement from scratch using the deliverable, retrieved_record_sections, and retrieved_requirement_context.",
                "You may revise the assessment upward or downward based on the admissible evidence.",
            ],
        },
        instructions=instructions,
        include_feedback=True,
    )
    raw_analysis = parse_compliance_analysis_response(
        raw_analysis=llm_service.generate(prompt, temperature=0.0),
        method="two_stage_rag",
        expected_count=len(deliverables),
    )
    candidate_analysis = _normalize_retrieval_stage_analysis(
        candidate_analysis=raw_analysis,
        deliverables=deliverables,
        retrieved_payload=retrieved_payload,
        allowed_record_documents=allowed_record_documents,
        stage_key=STAGE_3_KEY,
        stage_label=STAGE_LABELS[STAGE_3_KEY],
        requirement_weights=_extract_deliverable_weights(deliverables),
    )
    analysis = _merge_stage_analysis(
        candidate_analysis=candidate_analysis,
        previous_analysis=stage_2_analysis,
        retrieved_payload=retrieved_payload,
        allowed_record_documents=allowed_record_documents,
        stage_key=STAGE_3_KEY,
        stage_label=STAGE_LABELS[STAGE_3_KEY],
        requirement_weights=_extract_deliverable_weights(deliverables),
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
        retrieval_metrics=retrieval_metrics,
    )


def _build_stage_evaluations(
    *,
    deliverables: list[dict[str, Any]],
    baseline_analysis: ComplianceAnalysis | None,
    retrieved_payload: list[dict[str, Any]],
    include_reference_context: bool,
    include_baseline: bool,
    include_feedback: bool,
) -> list[dict[str, Any]]:
    findings = (baseline_analysis.procedure_to_record or baseline_analysis.findings) if baseline_analysis else []
    rows = baseline_analysis.linked_rows if baseline_analysis else []
    evaluations: list[dict[str, Any]] = []
    for index, deliverable in enumerate(deliverables):
        payload = retrieved_payload[index] if index < len(retrieved_payload) else {}
        item = {
            "requirement_ref": f"REQ-{index + 1}",
            "deliverable": serialize_deliverable_for_prompt(deliverable),
            "retrieved_record_sections": payload.get("retrieved_record_sections", []),
        }
        if include_baseline:
            finding = findings[index] if index < len(findings) else None
            row = rows[index] if index < len(rows) else None
            item["baseline_assessment"] = {
                "status": finding.status if finding else "not_satisfied",
                "evidence": finding.evidence if finding else [],
                "source_document": finding.source_document if finding else "",
                "gap": row.gap if include_feedback and row else "",
                "recommendation": row.recommendation if include_feedback and row else "",
            }
        if include_reference_context:
            item["retrieved_requirement_context"] = payload.get("retrieved_requirement_context", [])
        evaluations.append(item)
    return evaluations


def _normalize_retrieval_stage_analysis(
    *,
    candidate_analysis: ComplianceAnalysis,
    deliverables: list[dict[str, Any]],
    retrieved_payload: list[dict[str, Any]],
    allowed_record_documents: set[str],
    stage_key: str,
    stage_label: str,
    requirement_weights: list[float] | None = None,
) -> ComplianceAnalysis:
    candidate_findings = candidate_analysis.procedure_to_record or candidate_analysis.findings
    candidate_rows = candidate_analysis.linked_rows

    normalized_findings: list[ComplianceFinding] = []
    normalized_rows: list[ComplianceLinkedRow] = []
    for index, candidate_finding in enumerate(candidate_findings):
        candidate_row = candidate_rows[index] if index < len(candidate_rows) else None
        retrieved_sections = retrieved_payload[index].get("retrieved_record_sections", [])
        normalized_finding = normalize_requirement_finding(
            finding=candidate_finding,
            retrieved_record_sections=retrieved_sections,
            allowed_record_documents=allowed_record_documents,
        )
        tagged_finding = _tag_finding_evidence(
            finding=normalized_finding,
            previous_finding=normalized_finding.model_copy(update={"evidence_items": []}),
            stage_key=stage_key,
            stage_label=stage_label,
        )
        normalized_findings.append(tagged_finding)
        normalized_rows.append(
            normalize_requirement_linked_row(
                finding=tagged_finding,
                row=candidate_row,
            )
        )

    analysis = assemble_compliance_analysis(
        findings=normalized_findings,
        linked_rows=apply_row_level_record_recall(
            linked_rows=normalized_rows,
            findings=normalized_findings,
            retrieved_payload=retrieved_payload,
        ),
    )
    analysis = enrich_analysis_for_scoring(
        analysis,
        requirement_weights=requirement_weights,
        deliverable_metadata=deliverables,
    )
    return assemble_compliance_analysis(
        findings=analysis.procedure_to_record or analysis.findings,
        linked_rows=analysis.linked_rows,
    )


def _merge_stage_analysis(
    *,
    candidate_analysis: ComplianceAnalysis,
    previous_analysis: ComplianceAnalysis,
    retrieved_payload: list[dict[str, Any]],
    allowed_record_documents: set[str],
    stage_key: str,
    stage_label: str,
    requirement_weights: list[float] | None = None,
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
        merge_mode = _select_requirement_merge_mode(
            previous_finding=previous_finding,
            candidate_finding=normalized_candidate,
            stage_key=stage_key,
        )
        if merge_mode == "keep":
            merged_finding = previous_finding
            merged_row = normalize_requirement_linked_row(
                finding=merged_finding,
                row=_select_merged_row(
                    previous_row=previous_row,
                    candidate_row=candidate_row,
                    stage_key=stage_key,
                ),
            )
        elif merge_mode == "merge":
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
        else:
            merged_finding = _tag_finding_evidence(
                finding=normalized_candidate,
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
        requirement_weights=requirement_weights,
    )
    return assemble_compliance_analysis(
        findings=analysis.procedure_to_record or analysis.findings,
        linked_rows=analysis.linked_rows,
    )


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
                        source_document=finding.source_document,
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
                        "source_document": finding.source_document or existing.source_document,
                    }
                )
            )
        else:
            evidence_items.append(
                ComplianceEvidenceItem(
                    text=evidence,
                    source_document=finding.source_document,
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
                "source_document": existing.source_document or item.source_document,
                "stage_key": existing.stage_key or item.stage_key,
                "stage_label": existing.stage_label or item.stage_label,
            }
        )

    merged_items = [evidence_by_key[key] for key in ordered_keys if key in evidence_by_key]
    merged_source = previous_finding.source_document or tagged_candidate.source_document
    for item in merged_items:
        merged_source = merged_source or item.source_document

    return tagged_candidate.model_copy(
        update={
            "evidence": [item.text for item in merged_items],
            "source_document": merged_source,
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


def _status_rank(status: str) -> int:
    order = {
        "not_satisfied": 0,
        "partial": 1,
        "satisfied": 2,
    }
    return order.get(status, -1)


def _restore_model_judgement_floor(
    *,
    candidate_analysis: ComplianceAnalysis,
    model_analysis: ComplianceAnalysis,
) -> ComplianceAnalysis:
    candidate_findings = candidate_analysis.procedure_to_record or candidate_analysis.findings
    candidate_rows = candidate_analysis.linked_rows
    model_findings = model_analysis.procedure_to_record or model_analysis.findings
    model_rows = model_analysis.linked_rows

    restored_findings: list[ComplianceFinding] = []
    restored_rows: list[ComplianceLinkedRow] = []
    for index, candidate_finding in enumerate(candidate_findings):
        model_finding = model_findings[index] if index < len(model_findings) else None
        model_row = model_rows[index] if index < len(model_rows) else None
        candidate_row = candidate_rows[index] if index < len(candidate_rows) else None

        if (
            model_finding is None
            or _status_rank(model_finding.status) >= _status_rank(candidate_finding.status)
            or not _should_preserve_contradiction_judgement(model_row)
        ):
            restored_findings.append(candidate_finding)
            restored_rows.append(candidate_row if candidate_row is not None else model_row)
            continue

        restored_findings.append(
            _align_finding_scores_to_status(
                candidate_finding.model_copy(
                    update={
                        "status": model_finding.status,
                    }
                )
            )
        )
        restored_rows.append(
            (candidate_row if candidate_row is not None else model_row).model_copy(
                update={
                    "status": model_finding.status,
                    "rationale": model_row.rationale if model_row else (candidate_row.rationale if candidate_row else ""),
                    "gap": model_row.gap if model_row else (candidate_row.gap if candidate_row else ""),
                    "recommendation": (
                        model_row.recommendation
                        if model_row
                        else (candidate_row.recommendation if candidate_row else "")
                    ),
                }
            )
            if (candidate_row is not None or model_row is not None)
            else ComplianceLinkedRow(
                requirement_ref=f"REQ-{index + 1}",
                requirement=candidate_finding.requirement,
                status=model_finding.status,
                rationale="",
                gap="",
                recommendation="",
            )
        )

    return candidate_analysis.model_copy(
        update={
            "procedure_to_record": restored_findings,
            "findings": restored_findings,
            "linked_rows": restored_rows,
        }
    )


def _should_preserve_contradiction_judgement(row: ComplianceLinkedRow | None) -> bool:
    rationale = normalize_whitespace(row.rationale if row else "").lower()
    if not rationale:
        return False
    contradiction_markers = (
        "does not match",
        "do not match",
        "mismatch",
        "conflict",
        "conflicts with",
        "contradict",
        "contradiction",
        "inconsistent",
        "incorrectly states",
        "incorrectly records",
        "incorrectly classified",
        "highest criticality as",
        "highest observed function criticality",
        "highest function criticality",
        "but the",
    )
    return any(marker in rationale for marker in contradiction_markers)


def _align_finding_scores_to_status(finding: ComplianceFinding) -> ComplianceFinding:
    if finding.status == "satisfied":
        return finding
    if finding.status == "partial":
        return finding.model_copy(
            update={
                "requirement_coverage_percent": min(int(finding.requirement_coverage_percent or 0), 49),
                "evidence_strength": min(float(finding.evidence_strength or 0.0), 0.49),
            }
        )
    return finding.model_copy(
        update={
            "requirement_coverage_percent": 0,
            "evidence_strength": min(float(finding.evidence_strength or 0.0), 0.19),
        }
    )


def _select_requirement_merge_mode(
    *,
    previous_finding: ComplianceFinding,
    candidate_finding: ComplianceFinding,
    stage_key: str,
) -> str:
    previous_rank = _status_rank(previous_finding.status)
    candidate_rank = _status_rank(candidate_finding.status)
    previous_evidence_keys = _finding_evidence_keys(previous_finding)
    candidate_evidence_keys = _finding_evidence_keys(candidate_finding)
    has_new_citations = bool(candidate_evidence_keys - previous_evidence_keys)

    if candidate_rank > previous_rank:
        return "merge" if has_new_citations or not previous_evidence_keys else "keep"
    if candidate_rank == previous_rank:
        return "merge" if has_new_citations else "keep"
    if stage_key == STAGE_3_KEY:
        return "keep"
    if _should_accept_downgrade(
        previous_finding=previous_finding,
        candidate_finding=candidate_finding,
        previous_evidence_keys=previous_evidence_keys,
        candidate_evidence_keys=candidate_evidence_keys,
    ):
        return "replace"
    return "keep"


def _finding_evidence_keys(finding: ComplianceFinding) -> set[str]:
    item_keys = {
        _normalize_evidence_key(item.text)
        for item in finding.evidence_items
        if _normalize_evidence_key(item.text)
    }
    if item_keys:
        return item_keys
    return {
        _normalize_evidence_key(text)
        for text in finding.evidence
        if _normalize_evidence_key(text)
    }


def _should_accept_downgrade(
    *,
    previous_finding: ComplianceFinding,
    candidate_finding: ComplianceFinding,
    previous_evidence_keys: set[str],
    candidate_evidence_keys: set[str],
) -> bool:
    if candidate_evidence_keys:
        return True
    if previous_evidence_keys and not candidate_evidence_keys:
        return True
    return previous_finding.status != candidate_finding.status


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

