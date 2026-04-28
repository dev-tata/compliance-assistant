from __future__ import annotations

from typing import Any

from app.schemas.compliance import ComplianceFinding, ComplianceLinkedRow, RetrievalMetrics
from app.services.compliance_methods.compliance_method_common import (
    build_requirement_query_text,
    evidence_supported_by_sections,
    serialize_deliverable_for_prompt,
    serialize_retrieved_section,
)
from app.services.retrieval.faiss_retrieval import FAISS_TOP_K, RERANK_TOP_K
from app.services.retrieval.record_index_service import search_prepared_record_indexes

RECORD_TOP_K = FAISS_TOP_K
RECORD_FINAL_TOP_K = RERANK_TOP_K


def build_record_retrieval_payload(
    *,
    deliverables: list[dict[str, Any]],
    prepared_record_indexes: list[tuple[Any, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for index, deliverable in enumerate(deliverables):
        retrieved_record_sections = [
            serialize_retrieved_section(section)
            for section in search_prepared_record_indexes(
                prepared_indexes=prepared_record_indexes,
                query_text=build_requirement_query_text(deliverable),
                top_k=RECORD_TOP_K,
                final_top_k=RECORD_FINAL_TOP_K,
            )
        ]
        payload.append(
            {
                "requirement_ref": f"REQ-{index + 1}",
                "deliverable": serialize_deliverable_for_prompt(deliverable),
                "retrieved_record_sections": retrieved_record_sections,
            }
        )
    return payload


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
            average_record_recall_at_k=0.0,
            record_k=k,
            evaluated_requirements=0,
            hit_requirements=0,
        )

    hit_requirements = 0
    row_recalls: list[float] = []
    for index in range(evaluated):
        finding = findings[index]
        retrieved_sections = retrieved_payload[index].get("retrieved_record_sections", [])
        hit = any(
            evidence_supported_by_sections(evidence, retrieved_sections)
            for evidence in finding.evidence
        )
        row_recalls.append(1.0 if hit else 0.0)
        if hit:
            hit_requirements += 1

    average_record_recall_at_k = round(sum(row_recalls) / len(row_recalls), 4) if row_recalls else 0.0
    return RetrievalMetrics(
        record_recall_at_k=average_record_recall_at_k,
        average_record_recall_at_k=average_record_recall_at_k,
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
