from __future__ import annotations

from typing import Any

from app.schemas.compliance import (
    ComplianceFinding,
    ComplianceLinkedRow,
    ComplianceRequest,
    ComplianceResponse,
    RetrievalMetrics,
)
from app.services.compliance_methods.compliance_method_common import (
    apply_computed_overall_assessment,
    assemble_compliance_analysis,
    build_compliance_result_path,
    build_requirement_query_text,
    evidence_supported_by_sections,
    evaluate_single_requirement,
    normalize_requirement_finding,
    normalize_requirement_linked_row,
    serialize_baseline_analysis,
    serialize_deliverable_for_prompt,
    serialize_retrieved_section,
)
from app.services.compliance_methods.non_rag_service import evaluate_non_rag_analysis
from app.services.compliance_scoring_service import compute_scores, enrich_analysis_for_scoring
from app.services.document_service import current_timestamp
from app.services.llm.errors import LLMGenerationError
from app.services.llm.factory import get_llm_service
from app.services.retrieval.faiss_retrieval import FAISS_TOP_K, RERANK_TOP_K, normalize_whitespace
from app.services.retrieval.record_index_service import prepare_record_indexes, search_prepared_record_indexes

RECORD_TOP_K = FAISS_TOP_K
RECORD_FINAL_TOP_K = RERANK_TOP_K


def run_single_source_rag_compliance(
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
            "single_source_rag requires extracted deliverables. Generate or select deliverables first."
        )

    llm_service = get_llm_service(request.provider, request.model)
    record_indexes = prepare_record_indexes(case_payload.get("records", []))
    if not record_indexes:
        raise LLMGenerationError("single_source_rag requires retrievable record sections.")

    allowed_record_documents = {
        normalize_whitespace(document.get("source_filename") or document.get("stored_filename"))
        for document in case_payload.get("records", [])
    }
    baseline_analysis = evaluate_non_rag_analysis(
        llm_service=llm_service,
        case_payload=case_payload,
        instructions=request.instructions,
    )
    baseline_items = serialize_baseline_analysis(baseline_analysis)

    normalized_findings: list[ComplianceFinding] = []
    linked_rows: list[ComplianceLinkedRow] = []
    retrieved_payload: list[dict[str, Any]] = []

    for index, deliverable in enumerate(deliverables):
        retrieved_record_sections = [
            serialize_retrieved_section(section)
            for section in search_prepared_record_indexes(
                prepared_indexes=record_indexes,
                query_text=build_requirement_query_text(deliverable),
                top_k=RECORD_TOP_K,
                final_top_k=RECORD_FINAL_TOP_K,
            )
        ]
        requirement_analysis = evaluate_single_requirement(
            llm_service=llm_service,
            method="single_source_rag",
            requirement_payload={
                "method": "single_source_rag",
                "requirement_source": "deliverables",
                "baseline_assessment": [baseline_items[index]] if index < len(baseline_items) else [],
                "requirement_evaluations": [
                    {
                        "requirement_ref": "REQ-1",
                        "deliverable": serialize_deliverable_for_prompt(deliverable),
                        "retrieved_record_sections": retrieved_record_sections,
                    }
                ],
                "workflow_rules": [
                    "Evaluate only the single input requirement.",
                    "Use retrieved_record_sections only as admissible evidence.",
                    "Use the matching baseline assessment only as a prior summary, never as evidence.",
                ],
            },
            instructions=request.instructions,
        )
        normalized_finding = normalize_requirement_finding(
            finding=requirement_analysis.procedure_to_record[0],
            retrieved_record_sections=retrieved_record_sections,
            allowed_record_documents=allowed_record_documents,
        )
        normalized_findings.append(normalized_finding)
        linked_rows.append(
            normalize_requirement_linked_row(
                finding=normalized_finding,
                row=requirement_analysis.linked_rows[0] if requirement_analysis.linked_rows else None,
            )
        )
        retrieved_payload.append(
            {
                "requirement_ref": f"REQ-{index + 1}",
                "deliverable": serialize_deliverable_for_prompt(deliverable),
                "retrieved_record_sections": retrieved_record_sections,
            }
        )

    analysis = assemble_compliance_analysis(
        findings=normalized_findings,
        linked_rows=apply_row_level_record_recall(
            linked_rows=linked_rows,
            findings=normalized_findings,
            retrieved_payload=retrieved_payload,
        ),
    )
    analysis = enrich_analysis_for_scoring(analysis)
    analysis = apply_computed_overall_assessment(analysis)
    scores = compute_scores(analysis)

    saved_path = build_compliance_result_path(case_id)
    response = ComplianceResponse(
        case_id=case_id,
        compliance_provider=request.provider,
        compliance_model=request.model,
        extraction_provider=case_payload.get("extraction_provider"),
        extraction_model=case_payload.get("extraction_model"),
        method="single_source_rag",
        reference_stored_filenames=[],
        created_at=current_timestamp(),
        saved_at=saved_path.as_posix(),
        analysis=analysis,
        scores=scores,
        section_matches=[],
        retrieval_metrics=compute_record_recall_at_k(
            findings=analysis.procedure_to_record or analysis.findings,
            retrieved_payload=retrieved_payload,
            k=RECORD_FINAL_TOP_K,
        ),
    )
    saved_path.write_text(
        response.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )
    return response


def compute_record_recall_at_k(
    *,
    findings: list[ComplianceFinding],
    retrieved_payload: list[dict[str, Any]],
    k: int,
) -> RetrievalMetrics:
    evaluated = min(len(findings), len(retrieved_payload))
    if evaluated <= 0:
        return RetrievalMetrics(
            record_recall_at_k=0.0,
            record_k=k,
            evaluated_requirements=0,
            hit_requirements=0,
        )

    hit_requirements = 0
    for index in range(evaluated):
        finding = findings[index]
        retrieved_sections = retrieved_payload[index].get("retrieved_record_sections", [])
        if any(evidence_supported_by_sections(evidence, retrieved_sections) for evidence in finding.evidence):
            hit_requirements += 1

    return RetrievalMetrics(
        record_recall_at_k=round(hit_requirements / evaluated, 4),
        record_k=k,
        evaluated_requirements=evaluated,
        hit_requirements=hit_requirements,
    )


def apply_row_level_record_recall(
    *,
    linked_rows: list[ComplianceLinkedRow],
    findings: list[ComplianceFinding],
    retrieved_payload: list[dict[str, Any]],
) -> list[ComplianceLinkedRow]:
    evaluated = min(len(linked_rows), len(findings), len(retrieved_payload))
    resolved_rows: list[ComplianceLinkedRow] = []
    for index, row in enumerate(linked_rows):
        row_recall_at_k: float | None = None
        if index < evaluated:
            retrieved_sections = retrieved_payload[index].get("retrieved_record_sections", [])
            row_recall_at_k = 1.0 if any(
                evidence_supported_by_sections(evidence, retrieved_sections)
                for evidence in findings[index].evidence
            ) else 0.0
        resolved_rows.append(
            row.model_copy(
                update={
                    "record_recall_at_k": row_recall_at_k,
                }
            )
        )
    return resolved_rows
