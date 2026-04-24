from __future__ import annotations

from typing import Any

from app.schemas.compliance import (
    ComplianceFinding,
    ComplianceLinkedRow,
    ComplianceRequest,
    ComplianceResponse,
)
from app.services.compliance_methods.compliance_method_common import (
    apply_computed_overall_assessment,
    assemble_compliance_analysis,
    build_compliance_result_path,
    build_requirement_query_text,
    evaluate_single_requirement,
    normalize_requirement_finding,
    normalize_requirement_linked_row,
    serialize_baseline_analysis,
    serialize_deliverable_for_prompt,
    serialize_retrieved_section,
)
from app.services.compliance_methods.non_rag_service import evaluate_non_rag_analysis
from app.services.compliance_methods.single_source_rag_service import (
    apply_row_level_record_recall,
    compute_record_recall_at_k,
)
from app.services.compliance_scoring_service import compute_scores, enrich_analysis_for_scoring
from app.services.document_service import current_timestamp
from app.services.llm.errors import LLMGenerationError
from app.services.llm.factory import get_llm_service
from app.services.retrieval.faiss_retrieval import FAISS_TOP_K, RERANK_TOP_K, normalize_whitespace
from app.services.retrieval.record_index_service import prepare_record_indexes, search_prepared_record_indexes
from app.services.retrieval.reference_index_service import (
    prepare_reference_indexes,
    search_prepared_reference_indexes,
)

CONTEXT_TOP_K = FAISS_TOP_K
CONTEXT_FINAL_TOP_K = RERANK_TOP_K
RECORD_TOP_K = FAISS_TOP_K
RECORD_FINAL_TOP_K = RERANK_TOP_K


def run_multi_source_rag_compliance(
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
            "multi_source_rag requires extracted deliverables. Generate or select deliverables first."
        )
    if not case_payload.get("references", []):
        raise LLMGenerationError("multi_source_rag requires reference documents for requirement context.")

    llm_service = get_llm_service(request.provider, request.model)
    record_indexes = prepare_record_indexes(case_payload.get("records", []))
    if not record_indexes:
        raise LLMGenerationError("multi_source_rag requires retrievable record sections.")
    reference_context_indexes = prepare_reference_indexes(case_payload.get("references", []))
    if not reference_context_indexes:
        raise LLMGenerationError("multi_source_rag requires retrievable reference sections.")

    allowed_record_documents = {
        normalize_whitespace(document.get("source_filename") or document.get("stored_filename"))
        for document in case_payload.get("records", [])
    }
    non_rag_baseline_items = serialize_baseline_analysis(
        evaluate_non_rag_analysis(
            llm_service=llm_service,
            case_payload=case_payload,
            instructions=request.instructions,
        )
    )

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
        reference_query = _build_reference_query(
            deliverable=deliverable,
            retrieved_record_sections=retrieved_record_sections,
        )
        single_source_requirement_analysis = evaluate_single_requirement(
            llm_service=llm_service,
            method="single_source_rag",
            requirement_payload={
                "method": "single_source_rag",
                "requirement_source": "deliverables",
                "baseline_assessment": [non_rag_baseline_items[index]] if index < len(non_rag_baseline_items) else [],
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
        single_source_baseline_finding = normalize_requirement_finding(
            finding=single_source_requirement_analysis.procedure_to_record[0],
            retrieved_record_sections=retrieved_record_sections,
            allowed_record_documents=allowed_record_documents,
        )
        retrieved_requirement_context = [
            serialize_retrieved_section(section)
            for section in _search_requirement_context(
                reference_context_indexes=reference_context_indexes,
                query_text=reference_query,
                top_k=CONTEXT_TOP_K,
                final_top_k=CONTEXT_FINAL_TOP_K,
            )
        ]
        requirement_analysis = evaluate_single_requirement(
            llm_service=llm_service,
            method="multi_source_rag",
            requirement_payload={
                "method": "multi_source_rag",
                "requirement_source": "deliverables",
                "baseline_assessment": [
                    {
                        "requirement_ref": f"REQ-{index + 1}",
                        "requirement": single_source_baseline_finding.requirement,
                        "status": single_source_baseline_finding.status,
                        "evidence": single_source_baseline_finding.evidence,
                        "source_documents": single_source_baseline_finding.source_documents,
                    }
                ],
                "requirement_evaluations": [
                    {
                        "requirement_ref": "REQ-1",
                        "deliverable": serialize_deliverable_for_prompt(deliverable),
                        "retrieval_flow": "records_first_then_references_conditioned_on_record_result",
                        "reference_query": reference_query,
                        "retrieved_record_sections": retrieved_record_sections,
                        "retrieved_requirement_context": retrieved_requirement_context,
                    }
                ],
                "workflow_rules": [
                    "Evaluate only the single input requirement.",
                    "Use retrieved_record_sections as admissible evidence.",
                    "Use retrieved_requirement_context only to interpret the requirement, never as evidence.",
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
                "retrieved_requirement_context": retrieved_requirement_context,
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
        method="multi_source_rag",
        reference_stored_filenames=[
            item.get("stored_filename")
            for item in case_payload.get("references", [])
            if item.get("stored_filename")
        ],
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
