from __future__ import annotations

import csv
import re
from typing import Any, Iterable, Sequence
from .config import evaluation_v3_config
from .schemas import (
    ComplianceLabel,
    ContradictionType,
    DeliverableNode,
    EVALUATION_V3_ANALYSIS_METRICS,
    EvaluationV3Result,
    EvaluationV3ResultRow,
    EvaluationUnit,
    EvaluationV3Metrics,
    EvidenceNode,
    MiniKGLinks,
    RequirementType,
    ReferenceNode,
    StageJudgment,
)

STATUS_COMPLETION_SCORES = {
    "satisfied": 1.0,
    "partial": 0.33,
    "not_satisfied": 0.0,
}

STRONG_CLAIM_MARKERS = (
    "all",
    "each",
    "every",
    "both",
    "including",
    "include",
    "includes",
    "as well as",
    "together with",
)

DIRECT_CONFLICT_MARKERS = (
    "not performed",
    "absent",
    "missing",
    "failed",
    "not verified",
)

REFERENCE_CONFLICT_MARKERS = (
    "contradicts",
    "conflicts with",
    "inconsistent with",
    "cannot both be true",
    "opposite of",
    "does not align with",
)

MISSING_EVIDENCE_MARKERS = (
    "not explicitly",
    "not clearly",
    "does not show",
    "not documented",
    "not stated",
)

CONFLICT_MARKERS = DIRECT_CONFLICT_MARKERS

EVIDENCE_STATUS_SCORES = {
    "supported": 1.0,
    "partial": 0.5,
    "missing": 0.0,
    "conflicting": 0.0,
}

FINAL_LABEL_BY_EVIDENCE_STATUS = {
    "supported": "satisfied",
    "partial": "partial",
    "missing": "not_satisfied",
    "conflicting": "not_satisfied",
}


def build_evaluation_unit(
    *,
    frozen_deliverable: DeliverableNode | dict[str, Any],
    retrieved_record_evidence_chunks: Sequence[EvidenceNode | dict[str, Any]] | None = None,
    retrieved_reference_evidence_chunks: Sequence[ReferenceNode | dict[str, Any]] | None = None,
    stage_1_output: StageJudgment | dict[str, Any] | None = None,
    stage_2_output: StageJudgment | dict[str, Any] | None = None,
    stage_3_output: StageJudgment | dict[str, Any] | None = None,
    required_evidence_count: int | None = None,
    contradiction_type: ContradictionType = "none",
    verifier_input: dict[str, Any] | None = None,
) -> EvaluationUnit:
    deliverable = _coerce_deliverable_node(frozen_deliverable)
    record_nodes = _coerce_record_evidence_nodes(
        deliverable_id=deliverable.deliverable_id,
        items=retrieved_record_evidence_chunks or [],
    )
    reference_nodes = _coerce_reference_nodes(
        deliverable_id=deliverable.deliverable_id,
        items=retrieved_reference_evidence_chunks or [],
    )

    stage_1 = _coerce_stage_judgment(
        stage_key="stage_1",
        raw_output=stage_1_output,
        record_nodes=record_nodes,
        reference_nodes=reference_nodes,
    )
    stage_2 = _coerce_stage_judgment(
        stage_key="stage_2",
        raw_output=stage_2_output,
        record_nodes=record_nodes,
        reference_nodes=reference_nodes,
    )
    stage_3 = _coerce_stage_judgment(
        stage_key="stage_3",
        raw_output=stage_3_output,
        record_nodes=record_nodes,
        reference_nodes=reference_nodes,
    )
    stage_3 = _merge_stage_3_grounded_record_evidence(
        stage_2=stage_2,
        stage_3=stage_3,
    )

    (
        base_required_evidence_count,
        weight_modifier,
        required_evidence_count_reason,
        resolved_required_evidence_count,
    ) = _resolve_required_evidence_count(
        deliverable=deliverable,
        explicit_value=required_evidence_count,
    )
    requirement_type = _classify_requirement_type(deliverable.requirement_text)
    print(
        "[evaluation_v3_grounding_debug]",
        {
            "retrieved_chunk_count": len(record_nodes),
            "first_chunk_text": record_nodes[0].text if record_nodes else "",
            "requirement_text": deliverable.requirement_text,
        },
    )
    print(
        {
            "stage": "evaluation_v3.builder.record_node_scores",
            "deliverable_id": deliverable.deliverable_id,
            "scores": [
                {
                    "text": node.text[:80],
                    "retrieval_score": node.retrieval_score,
                    "raw_retrieval_score": node.raw_retrieval_score,
                }
                for node in record_nodes
            ],
        }
    )
    # Evaluation V3 metrics (ground truth)
    # Uses grounded evidence and retrieval.
    # IMPORTANT: final_label and evidence_status are resolved from grounded evidence,
    # conflict signals, and retrieval coverage, not from legacy compliance status fields.
    final_label, final_rationale = _resolve_final_judgment(stage_1=stage_1, stage_2=stage_2, stage_3=stage_3)
    grounded_record_nodes = _resolve_grounded_record_nodes(
        stage_judgments=(stage_1, stage_2, stage_3),
        record_nodes=record_nodes,
    )
    grounded_record_evidence_count = _count_grounded_record_evidence(
        stage_judgments=(stage_1, stage_2, stage_3),
        record_nodes=record_nodes,
    )
    _log_grounding_selection_debug(
        deliverable_id=deliverable.deliverable_id,
        record_nodes=record_nodes,
        grounded_nodes=grounded_record_nodes,
    )
    base_evidence_status = _resolve_base_evidence_status(
        grounded_record_evidence_count=grounded_record_evidence_count,
        required_evidence_count=resolved_required_evidence_count,
    )
    resolved_contradiction_type = _resolve_contradiction_type(
        explicit_contradiction_type=contradiction_type,
        base_evidence_status=base_evidence_status,
        final_label=final_label,
        stage_judgments=(stage_1, stage_2, stage_3),
        record_nodes=record_nodes,
        verifier_input=verifier_input,
    )
    conflict_detected = _detect_conflict(
        stage_judgments=(stage_1, stage_2, stage_3),
        verifier_input=verifier_input,
    )
    # Final label ordering is fixed:
    # 1. evidence_status
    # 2. required_evidence_count check (inside evidence_status resolution)
    # 3. subsection downgrade in _resolve_final_label
    evidence_status = _resolve_evidence_status(
        grounded_record_evidence_count=grounded_record_evidence_count,
        required_evidence_count=resolved_required_evidence_count,
        conflict_detected=conflict_detected,
    )
    final_label = _resolve_final_label(
        evidence_status=evidence_status,
        unit_context=EvaluationUnit(
            deliverable=deliverable,
            weight=deliverable.weight,
            requirement_type=requirement_type,
            base_required_evidence_count=base_required_evidence_count,
            weight_modifier=weight_modifier,
            required_evidence_count_reason=required_evidence_count_reason,
            required_evidence_count=resolved_required_evidence_count,
            evidence_status=evidence_status,
            contradiction_type=resolved_contradiction_type,
            evidence_score=None,
            record_evidence_chunks=record_nodes,
            reference_evidence_chunks=reference_nodes,
            stage_1_answer=stage_1,
            stage_2_answer=stage_2,
            stage_3_answer=stage_3,
            final_label=None,
            final_rationale=final_rationale,
            mini_kg_links=None,
            verifier_result=None,
            metrics=None,
        ),
    )
    evidence_status, final_label = _enforce_record_grounding_validation(
        grounded_record_evidence_count=grounded_record_evidence_count,
        evidence_status=evidence_status,
        final_label=final_label,
    )
    evidence_score = _compute_evidence_score(evidence_status=evidence_status)

    metrics = _build_metrics(
        evidence_status=evidence_status,
        final_label=final_label,
        required_evidence_count=resolved_required_evidence_count,
        stage_judgments=(stage_1, stage_2, stage_3),
        record_nodes=record_nodes,
    )
    mini_kg_links = _build_mini_kg_links(
        deliverable_id=deliverable.deliverable_id,
        stage_judgments=(stage_1, stage_2, stage_3),
        record_nodes=record_nodes,
        reference_nodes=reference_nodes,
    )

    return EvaluationUnit(
        deliverable=deliverable,
        weight=deliverable.weight,
        requirement_type=requirement_type,
        base_required_evidence_count=base_required_evidence_count,
        weight_modifier=weight_modifier,
        required_evidence_count_reason=required_evidence_count_reason,
        required_evidence_count=resolved_required_evidence_count,
        evidence_status=evidence_status,
        contradiction_type=resolved_contradiction_type,
        evidence_score=evidence_score,
        record_evidence_chunks=record_nodes,
        reference_evidence_chunks=reference_nodes,
        stage_1_answer=stage_1,
        stage_2_answer=stage_2,
        stage_3_answer=stage_3,
        final_label=final_label,
        final_rationale=final_rationale,
        mini_kg_links=mini_kg_links,
        verifier_result=None,
        metrics=metrics,
    )


def derive_deliverable_requirement_metadata(
    deliverable: DeliverableNode | dict[str, Any],
) -> dict[str, Any]:
    deliverable_node = _coerce_deliverable_node(deliverable)
    requirement_type = _classify_requirement_type(deliverable_node.requirement_text)
    (
        _base_required_evidence_count,
        _weight_modifier,
        _required_evidence_count_reason,
        required_evidence_count,
    ) = _resolve_required_evidence_count(
        deliverable=deliverable_node,
        explicit_value=None,
    )
    return {
        "required_evidence_count": required_evidence_count,
        "requirement_type": requirement_type,
        "weight": deliverable_node.weight,
    }


def calculate_aggregate_metrics(units: Sequence[EvaluationUnit]) -> EvaluationV3Metrics:
    if not units:
        return EvaluationV3Metrics(
            satisfied_count=0,
            partial_count=0,
            not_satisfied_count=0,
            supported_count=0,
            missing_count=0,
            conflicting_count=0,
            avg_grounded_evidence_count=0.0,
            avg_evidence_coverage_ratio=0.0,
        )

    return EvaluationV3Metrics(
        satisfied_count=sum(1 for unit in units if unit.final_label == "satisfied"),
        partial_count=sum(1 for unit in units if unit.final_label == "partial"),
        not_satisfied_count=sum(1 for unit in units if unit.final_label == "not_satisfied"),
        supported_count=sum(1 for unit in units if unit.evidence_status == "supported"),
        missing_count=sum(1 for unit in units if unit.evidence_status == "missing"),
        conflicting_count=sum(1 for unit in units if unit.evidence_status == "conflicting"),
        avg_grounded_evidence_count=round(
            sum(_resolve_debug_grounded_evidence_count(unit) for unit in units) / len(units),
            4,
        ),
        avg_evidence_coverage_ratio=round(
            sum(_resolve_debug_evidence_coverage_ratio(unit) for unit in units) / len(units),
            4,
        ),
    )


def build_evaluation_v3_result_row(unit: EvaluationUnit) -> EvaluationV3ResultRow:
    stage_1_grounded_evidence_count = _resolve_stage_grounded_evidence_count(
        stage_judgment=unit.stage_1_answer,
        record_nodes=unit.record_evidence_chunks,
    )
    stage_2_grounded_evidence_count = _resolve_stage_grounded_evidence_count(
        stage_judgment=unit.stage_2_answer,
        record_nodes=unit.record_evidence_chunks,
    )
    stage_3_grounded_evidence_count = _resolve_stage_grounded_evidence_count(
        stage_judgment=unit.stage_3_answer,
        record_nodes=unit.record_evidence_chunks,
    )
    return EvaluationV3ResultRow(
        deliverable_id=unit.deliverable.deliverable_id,
        final_label=unit.final_label,
        stage_1_label=unit.stage_1_answer.label,
        stage_2_label=unit.stage_2_answer.label,
        stage_3_label=unit.stage_3_answer.label,
        stage_1_evidence_status=_resolve_stage_evidence_status(
            stage_judgment=unit.stage_1_answer,
            grounded_evidence_count=stage_1_grounded_evidence_count,
            required_evidence_count=unit.required_evidence_count,
        ),
        stage_2_evidence_status=_resolve_stage_evidence_status(
            stage_judgment=unit.stage_2_answer,
            grounded_evidence_count=stage_2_grounded_evidence_count,
            required_evidence_count=unit.required_evidence_count,
        ),
        stage_3_evidence_status=_resolve_stage_evidence_status(
            stage_judgment=unit.stage_3_answer,
            grounded_evidence_count=stage_3_grounded_evidence_count,
            required_evidence_count=unit.required_evidence_count,
        ),
        stage_1_grounded_evidence_count=stage_1_grounded_evidence_count,
        stage_2_grounded_evidence_count=stage_2_grounded_evidence_count,
        stage_3_grounded_evidence_count=stage_3_grounded_evidence_count,
        stage_1_evidence_coverage_ratio=_compute_evidence_coverage_ratio(
            grounded_evidence_count=stage_1_grounded_evidence_count,
            required_evidence_count=int(unit.required_evidence_count or 0),
        ),
        stage_2_evidence_coverage_ratio=_compute_evidence_coverage_ratio(
            grounded_evidence_count=stage_2_grounded_evidence_count,
            required_evidence_count=int(unit.required_evidence_count or 0),
        ),
        stage_3_evidence_coverage_ratio=_compute_evidence_coverage_ratio(
            grounded_evidence_count=stage_3_grounded_evidence_count,
            required_evidence_count=int(unit.required_evidence_count or 0),
        ),
        evidence_status=unit.evidence_status,
        grounded_evidence_count=_resolve_debug_grounded_evidence_count(unit),
        grounded_chunk_count=_resolve_debug_grounded_chunk_count(unit),
        required_evidence_count=unit.required_evidence_count,
        evidence_coverage_ratio=_resolve_debug_evidence_coverage_ratio(unit),
        has_conflict=_resolve_debug_has_conflict(unit),
        contradiction_type=unit.contradiction_type,
    )


def build_evaluation_v3_result_rows(units: Sequence[EvaluationUnit]) -> list[EvaluationV3ResultRow]:
    return [
        build_evaluation_v3_result_row(unit)
        for unit in sorted(units, key=lambda item: item.deliverable.deliverable_id)
    ]


def build_evaluation_v3_summary(rows: Sequence[EvaluationV3ResultRow]) -> dict[str, Any]:
    total_units = len(rows)
    if total_units <= 0:
        return {
            "total_units": 0,
            "satisfied": 0,
            "partial": 0,
            "not_satisfied": 0,
            "supported": 0,
            "missing": 0,
            "conflicting": 0,
            "avg_grounded_evidence": 0.0,
            "avg_evidence_coverage": 0.0,
        }

    return {
        "total_units": total_units,
        "satisfied": sum(1 for row in rows if row.final_label == "satisfied"),
        "partial": sum(1 for row in rows if row.final_label == "partial"),
        "not_satisfied": sum(1 for row in rows if row.final_label == "not_satisfied"),
        "supported": sum(1 for row in rows if row.evidence_status == "supported"),
        "missing": sum(1 for row in rows if row.evidence_status == "missing"),
        "conflicting": sum(1 for row in rows if row.evidence_status == "conflicting"),
        "avg_grounded_evidence": round(
            sum(int(row.grounded_evidence_count or 0) for row in rows) / total_units,
            4,
        ),
        "avg_evidence_coverage": round(
            sum(float(row.evidence_coverage_ratio or 0.0) for row in rows) / total_units,
            4,
        ),
    }


def build_evaluation_v3_result(
    *,
    case_id: str,
    created_at: str,
    source_compliance_saved_at: str,
    compliance_provider: str,
    compliance_model: str,
    method: str,
    units: Sequence[EvaluationUnit],
    aggregate_metrics: EvaluationV3Metrics,
) -> EvaluationV3Result:
    return EvaluationV3Result(
        case_id=case_id,
        created_at=created_at,
        source_compliance_saved_at=source_compliance_saved_at,
        compliance_provider=compliance_provider,
        compliance_model=compliance_model,
        method=method,
        metrics={
            key: value
            for key, value in aggregate_metrics.model_dump(exclude_none=True).items()
            if key in EVALUATION_V3_ANALYSIS_METRICS
        },
        units=build_evaluation_v3_result_rows(units),
    )


def build_debug_report_rows(units: Sequence[EvaluationUnit]) -> list[dict[str, Any]]:
    return [
        {
            "deliverable_id": unit.deliverable.deliverable_id,
            "requirement_type": unit.requirement_type,
            "base_required_evidence_count": unit.base_required_evidence_count,
            "weight": unit.weight,
            "weight_modifier": unit.weight_modifier,
            "required_evidence_count_reason": unit.required_evidence_count_reason,
            "final_label": unit.final_label,
            "evidence_status": unit.evidence_status,
            "required_evidence_count": unit.required_evidence_count,
            "grounded_evidence_count": _resolve_debug_grounded_evidence_count(unit),
            "evidence_coverage_ratio": _resolve_debug_evidence_coverage_ratio(unit),
            "grounded_chunk_count": _resolve_debug_grounded_chunk_count(unit),
            "grounded_subsection_count": len(_resolve_debug_grounded_subsection_ids(unit)),
            "has_conflict": _resolve_debug_has_conflict(unit),
            "subsection_count": len(_resolve_debug_subsection_ids(unit)),
            "subsection_ids": _resolve_debug_subsection_ids(unit),
            "subsection_coverage_ratio": _resolve_debug_subsection_coverage_ratio(unit),
            "subsection_threshold": evaluation_v3_config["SUBSECTION_COVERAGE_THRESHOLD"],
            "subsection_downgrade_applied": _resolve_debug_subsection_downgrade_applied(unit),
            "contradiction_type": unit.contradiction_type,
            "evidence_score": unit.evidence_score,
            "record_evidence_section_count": len(unit.record_evidence_chunks),
            "reference_evidence_section_count": len(unit.reference_evidence_chunks),
            "stage_1_label": unit.stage_1_answer.label,
            "stage_2_label": unit.stage_2_answer.label,
            "stage_3_label": unit.stage_3_answer.label,
            "rationale": _resolve_debug_rationale(unit),
        }
        for unit in units
    ]


def build_debug_report_summary(units: Sequence[EvaluationUnit]) -> dict[str, Any]:
    rows = build_debug_report_rows(units)
    subsection_coverage_values = [
        float(row.get("subsection_coverage_ratio") or 0.0)
        for row in rows
    ]
    return {
        "total_units": len(units),
        "final_label_counts": {
            "satisfied": sum(1 for row in rows if row.get("final_label") == "satisfied"),
            "partial": sum(1 for row in rows if row.get("final_label") == "partial"),
            "not_satisfied": sum(1 for row in rows if row.get("final_label") == "not_satisfied"),
        },
        "evidence_status_counts": {
            "supported": sum(1 for row in rows if row.get("evidence_status") == "supported"),
            "partial": sum(1 for row in rows if row.get("evidence_status") == "partial"),
            "missing": sum(1 for row in rows if row.get("evidence_status") == "missing"),
            "conflicting": sum(1 for row in rows if row.get("evidence_status") == "conflicting"),
        },
        "grounded_evidence_count_distribution": _build_count_distribution(
            int(row.get("grounded_evidence_count") or 0)
            for row in rows
        ),
        "grounded_subsection_count_distribution": _build_count_distribution(
            int(row.get("grounded_subsection_count") or 0)
            for row in rows
        ),
        "subsection_coverage_ratio_min": round(min(subsection_coverage_values), 4) if subsection_coverage_values else 0.0,
        "subsection_coverage_ratio_max": round(max(subsection_coverage_values), 4) if subsection_coverage_values else 0.0,
        "subsection_coverage_ratio_average": round(
            sum(subsection_coverage_values) / len(subsection_coverage_values),
            4,
        ) if subsection_coverage_values else 0.0,
    }


def build_compact_summary(units: Sequence[EvaluationUnit]) -> dict[str, Any]:
    rows = build_debug_report_rows(units)
    total_units = len(units)
    if total_units <= 0:
        return {
            "total_units": 0,
            "satisfied": 0,
            "partial": 0,
            "not_satisfied": 0,
            "supported": 0,
            "missing": 0,
            "conflicting": 0,
            "avg_coverage": 0.0,
            "avg_grounded": 0.0,
        }

    return {
        "total_units": total_units,
        "satisfied": sum(1 for row in rows if row.get("final_label") == "satisfied"),
        "partial": sum(1 for row in rows if row.get("final_label") == "partial"),
        "not_satisfied": sum(1 for row in rows if row.get("final_label") == "not_satisfied"),
        "supported": sum(1 for row in rows if row.get("evidence_status") == "supported"),
        "missing": sum(1 for row in rows if row.get("evidence_status") == "missing"),
        "conflicting": sum(1 for row in rows if row.get("evidence_status") == "conflicting"),
        "avg_coverage": round(
            sum(float(row.get("evidence_coverage_ratio") or 0.0) for row in rows) / total_units,
            4,
        ),
        "avg_grounded": round(
            sum(int(row.get("grounded_evidence_count") or 0) for row in rows) / total_units,
            4,
        ),
    }


def build_edge_case_debug_rows(units: Sequence[EvaluationUnit]) -> list[dict[str, Any]]:
    rows = build_debug_report_rows(units)
    fieldnames = [
        "deliverable_id",
        "requirement_type",
        "base_required_evidence_count",
        "weight",
        "weight_modifier",
        "required_evidence_count_reason",
        "final_label",
        "evidence_status",
        "required_evidence_count",
        "grounded_evidence_count",
        "evidence_coverage_ratio",
        "grounded_subsection_count",
        "subsection_coverage_ratio",
        "has_conflict",
        "contradiction_type",
        "stage_1_label",
        "stage_2_label",
        "stage_3_label",
        "rationale",
    ]
    return [
        {field: row.get(field) for field in fieldnames}
        for row in rows
    ]


def build_suspicious_debug_rows(units: Sequence[EvaluationUnit]) -> list[dict[str, Any]]:
    rows = build_debug_report_rows(units)
    return [
        row
        for row in rows
        if _is_suspicious_debug_row(row)
    ]


def write_debug_report_json(
    *,
    units: Sequence[EvaluationUnit],
    output_path: str,
) -> None:
    rows = build_debug_report_rows(units)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(_serialize_json(rows))


def write_debug_report_csv(
    *,
    units: Sequence[EvaluationUnit],
    output_path: str,
) -> None:
    rows = build_debug_report_rows(units)
    fieldnames = [
        "deliverable_id",
        "requirement_type",
        "base_required_evidence_count",
        "weight",
        "weight_modifier",
        "required_evidence_count_reason",
        "final_label",
        "evidence_status",
        "contradiction_type",
        "evidence_score",
        "record_evidence_section_count",
        "reference_evidence_section_count",
        "stage_1_label",
        "stage_2_label",
        "stage_3_label",
        "rationale",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _coerce_deliverable_node(raw_deliverable: DeliverableNode | dict[str, Any]) -> DeliverableNode:
    if isinstance(raw_deliverable, DeliverableNode):
        return raw_deliverable
    data = dict(raw_deliverable)
    procedure_section_link = data.get("procedure_section_link") or {}
    return DeliverableNode(
        deliverable_id=data.get("deliverable_id") or data.get("id") or "DELIV-001",
        source_document=data.get("source_document") or procedure_section_link.get("source_document") or "",
        section_label=data.get("section_label") or procedure_section_link.get("section_label") or "",
        heading_title=data.get("heading_title") or procedure_section_link.get("heading_title") or "",
        requirement_text=data.get("requirement_text") or "",
        weight=data.get("weight") or 1.0,
        required_evidence_count=data.get("required_evidence_count"),
    )


def _coerce_record_evidence_nodes(
    *,
    deliverable_id: str,
    items: Sequence[EvidenceNode | dict[str, Any]],
) -> list[EvidenceNode]:
    nodes: list[EvidenceNode] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, EvidenceNode):
            node = item
            if not node.evidence_id:
                node = node.model_copy(update={"evidence_id": _build_evidence_id(deliverable_id, "record", index)})
        else:
            node = EvidenceNode(
                evidence_id=_resolve_record_evidence_id(item, deliverable_id, index),
                source_document=_pick_first(item, "source_document", "document", "stored_filename", "source_filename"),
                section_id=_pick_first(item, "section_id"),
                subsection_id=_pick_first(item, "subsection_id", "section_id"),
                section_label=_pick_first(item, "section_label"),
                heading_title=_pick_first(item, "heading_title"),
                text=_resolve_chunk_text(item),
                reranker_score=item.get("reranker_score"),
                raw_retrieval_score=_resolve_raw_retrieval_score(item),
                retrieval_score=_resolve_retrieval_score(item),
            )
        nodes.append(node)
    return nodes


def _coerce_reference_nodes(
    *,
    deliverable_id: str,
    items: Sequence[ReferenceNode | dict[str, Any]],
) -> list[ReferenceNode]:
    nodes: list[ReferenceNode] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, ReferenceNode):
            node = item
            if not node.reference_id:
                node = node.model_copy(update={"reference_id": _build_evidence_id(deliverable_id, "reference", index)})
        else:
            node = ReferenceNode(
                reference_id=_resolve_reference_id(item, deliverable_id, index),
                source_document=_pick_first(item, "source_document", "document", "stored_filename", "source_filename"),
                section_id=_pick_first(item, "section_id"),
                subsection_id=_pick_first(item, "subsection_id", "section_id"),
                section_label=_pick_first(item, "section_label"),
                heading_title=_pick_first(item, "heading_title"),
                text=_resolve_chunk_text(item),
                reranker_score=item.get("reranker_score"),
                raw_retrieval_score=_resolve_raw_retrieval_score(item),
                retrieval_score=_resolve_retrieval_score(item),
            )
        nodes.append(node)
    return nodes


def _coerce_stage_judgment(
    *,
    stage_key: str,
    raw_output: StageJudgment | dict[str, Any] | None,
    record_nodes: Sequence[EvidenceNode],
    reference_nodes: Sequence[ReferenceNode],
) -> StageJudgment:
    if isinstance(raw_output, StageJudgment):
        return raw_output.model_copy(update={"stage_key": stage_key})

    payload = dict(raw_output or {})
    record_ids = _resolve_stage_record_evidence_ids(payload, record_nodes)
    reference_ids = _resolve_stage_reference_ids(payload, reference_nodes)
    label = _normalize_label(payload.get("label"))
    rationale = payload.get("rationale") or payload.get("reasoning") or payload.get("summary") or ""
    conflict_flag = _extract_conflict_flag(payload)

    return StageJudgment(
        stage_key=stage_key,
        label=label,
        rationale=str(rationale or ""),
        conflict_flag=conflict_flag,
        supporting_record_evidence_ids=record_ids,
        supporting_record_evidence_items=_normalize_stage_record_evidence_items(payload.get("evidence_items")),
        supporting_reference_ids=reference_ids,
    )


def _resolve_stage_record_evidence_ids(payload: dict[str, Any], record_nodes: Sequence[EvidenceNode]) -> list[str]:
    explicit_ids = _normalize_id_list(payload.get("supporting_record_evidence_ids"))
    if explicit_ids:
        return explicit_ids

    evidence_items = payload.get("evidence_items")
    if isinstance(evidence_items, list):
        matched_ids = _match_items_to_record_nodes(evidence_items, record_nodes)
        if matched_ids:
            return matched_ids

    text_candidates = _extract_text_candidates(payload, keys=("evidence", "record_evidence", "record_quotes"))
    return _match_texts_to_record_nodes(text_candidates, record_nodes)


def _normalize_stage_record_evidence_items(raw_items: Any) -> list[dict[str, str]]:
    if not isinstance(raw_items, list):
        return []
    normalized_items: list[dict[str, str]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        normalized = {
            "evidence_id": str(item.get("evidence_id") or "").strip(),
            "section_id": str(item.get("section_id") or "").strip(),
            "subsection_id": str(item.get("subsection_id") or "").strip(),
            "section_label": str(item.get("section_label") or "").strip(),
            "heading_title": str(item.get("heading_title") or "").strip(),
            "source_document": str(item.get("source_document") or "").strip(),
            "source_stage": str(item.get("source_stage") or "").strip(),
            "text": str(item.get("text") or "").strip(),
        }
        if normalized["text"]:
            normalized_items.append(normalized)
    return normalized_items


def _resolve_stage_reference_ids(payload: dict[str, Any], reference_nodes: Sequence[ReferenceNode]) -> list[str]:
    explicit_ids = _normalize_id_list(payload.get("supporting_reference_ids"))
    if explicit_ids:
        return explicit_ids

    reference_items = payload.get("reference_items")
    if isinstance(reference_items, list):
        matched_ids = _match_items_to_reference_nodes(reference_items, reference_nodes)
        if matched_ids:
            return matched_ids

    text_candidates = _extract_text_candidates(
        payload,
        keys=("reference_evidence", "reference_quotes", "supporting_reference_texts"),
    )
    return _match_texts_to_reference_nodes(text_candidates, reference_nodes)


def _resolve_required_evidence_count(
    *,
    deliverable: DeliverableNode,
    explicit_value: int | None,
) -> tuple[int, int, str, int]:
    if explicit_value is not None:
        resolved = max(1, int(explicit_value))
        return resolved, 0, "explicit_value_override", resolved
    if deliverable.required_evidence_count is not None:
        resolved = max(1, int(deliverable.required_evidence_count))
        return resolved, 0, "deliverable_required_evidence_count_override", resolved
    requirement_type = _classify_requirement_type(deliverable.requirement_text)
    base_required_evidence_count = _base_required_evidence_count_for_requirement_type(
        requirement_type=requirement_type,
        requirement_text=deliverable.requirement_text,
    )
    weight_modifier = _weight_modifier_for_deliverable_weight(deliverable.weight)
    resolved_required_evidence_count = min(base_required_evidence_count + weight_modifier, 5)
    reason = (
        f"requirement_type={requirement_type}; "
        f"base_required_evidence_count={max(1, base_required_evidence_count)}; "
        f"weight_modifier={weight_modifier}; "
        f"weight={deliverable.weight}; "
        "cap=5"
    )
    return max(1, base_required_evidence_count), weight_modifier, reason, max(1, resolved_required_evidence_count)


def calculate_required_evidence_count(
    *,
    requirement_text: str,
    override_value: int | None = None,
) -> int:
    if override_value is not None:
        return max(0, override_value)
    normalized_text = _normalized_match_key(requirement_text)
    if _contains_strong_claim_marker(normalized_text):
        return 2
    return 1


def _base_required_evidence_count_for_requirement_type(
    *,
    requirement_type: RequirementType,
    requirement_text: str,
) -> int:
    if requirement_type in {"single_field", "conditional"}:
        return 1
    if requirement_type in {"relationship", "list_or_table", "control_measure"}:
        return 2
    if requirement_type == "per_function":
        return 3
    return calculate_required_evidence_count(requirement_text=requirement_text)


def _weight_modifier_for_deliverable_weight(weight: float) -> int:
    if weight >= 1.5:
        return 2
    if weight >= 1.2:
        return 1
    return 0


def _classify_requirement_type(requirement_text: str) -> RequirementType:
    normalized_text = _normalized_match_key(requirement_text)
    if any(
        marker in normalized_text
        for marker in ("residual risk", "deemed acceptable", "remaining after risk reduction")
    ):
        return "residual_risk_acceptability"
    if any(
        marker in normalized_text
        for marker in ("benefit-risk approach", "details and consequences", "in cases where a benefit-risk approach")
    ):
        return "benefit_risk_rationale"
    if any(marker in normalized_text for marker in ("each major system function", "for each", "per function")):
        return "per_function"
    if any(marker in normalized_text for marker in ("combination", "determined by", "criticality and complexity")):
        return "relationship"
    if any(marker in normalized_text for marker in ("risk control", "control measures", "risk reduction")):
        return "control_measure"
    if any(
        marker in normalized_text
        for marker in ("identified", "include", "including", "list", "major system functions")
    ):
        return "list_or_table"
    if (
        any(marker in normalized_text for marker in ("where necessary", "in cases where"))
        or bool(re.search(r"\bif\b", normalized_text))
    ):
        return "conditional"
    if any(
        marker in normalized_text
        for marker in ("shall be recorded", "shall be documented", "level shall be recorded")
    ):
        return "single_field"
    return "generic"


def _contains_strong_claim_marker(normalized_text: str) -> bool:
    if not normalized_text:
        return False
    for marker in STRONG_CLAIM_MARKERS:
        pattern = rf"\b{re.escape(marker)}\b"
        if re.search(pattern, normalized_text):
            return True
    return False


def _resolve_final_judgment(
    *,
    stage_1: StageJudgment,
    stage_2: StageJudgment,
    stage_3: StageJudgment,
) -> tuple[ComplianceLabel | None, str]:
    for stage in (stage_3, stage_2, stage_1):
        if stage.label or stage.rationale:
            return stage.label, stage.rationale
    return None, ""


def _compute_evidence_score(
    *,
    evidence_status: str,
) -> float | None:
    return EVIDENCE_STATUS_SCORES.get(evidence_status, 0.0)


def _resolve_final_label(
    *,
    evidence_status: str,
    unit_context: EvaluationUnit | None = None,
) -> ComplianceLabel:
    if evidence_status == "missing":
        return "not_satisfied"
    if evidence_status == "conflicting":
        return "partial"
    if evidence_status == "partial":
        return "partial"
    return "satisfied"


def _resolve_evidence_status(
    *,
    grounded_record_evidence_count: int,
    required_evidence_count: int,
    conflict_detected: bool,
) -> str:
    # Validation rule: only grounded record evidence can support a requirement.
    if grounded_record_evidence_count <= 0:
        return "missing"
    if conflict_detected:
        return "conflicting"
    if grounded_record_evidence_count >= required_evidence_count:
        return "supported"
    return "partial"


def _resolve_base_evidence_status(
    *,
    grounded_record_evidence_count: int,
    required_evidence_count: int,
) -> str:
    # Validation rule: reference context alone cannot satisfy support.
    if grounded_record_evidence_count <= 0:
        return "missing"
    if grounded_record_evidence_count >= required_evidence_count:
        return "supported"
    return "partial"


def _build_metrics(
    *,
    evidence_status: str,
    final_label: ComplianceLabel | None,
    required_evidence_count: int,
    stage_judgments: Sequence[StageJudgment],
    record_nodes: Sequence[EvidenceNode],
) -> EvaluationV3Metrics:
    grounded_nodes = _resolve_grounded_record_nodes(
        stage_judgments=stage_judgments,
        record_nodes=record_nodes,
    )
    return EvaluationV3Metrics(
        satisfied_count=1 if final_label == "satisfied" else 0,
        partial_count=1 if final_label == "partial" else 0,
        not_satisfied_count=1 if final_label == "not_satisfied" else 0,
        supported_count=1 if evidence_status == "supported" else 0,
        missing_count=1 if evidence_status == "missing" else 0,
        conflicting_count=1 if evidence_status == "conflicting" else 0,
        avg_grounded_evidence_count=float(
            _count_grounded_record_evidence(
                stage_judgments=stage_judgments,
                record_nodes=record_nodes,
            )
        ),
        avg_evidence_coverage_ratio=_compute_evidence_coverage_ratio(
            grounded_evidence_count=_count_grounded_record_evidence(
                stage_judgments=stage_judgments,
                record_nodes=record_nodes,
            ),
            required_evidence_count=required_evidence_count,
        ),
    )


def _enforce_record_grounding_validation(
    *,
    grounded_record_evidence_count: int,
    evidence_status: str,
    final_label: ComplianceLabel | None,
) -> tuple[str, ComplianceLabel]:
    if grounded_record_evidence_count <= 0:
        return "missing", "not_satisfied"
    return evidence_status, final_label or "not_satisfied"


def _resolve_unit_weight(unit: EvaluationUnit) -> float:
    weight = float(unit.weight or unit.deliverable.weight or 0.0)
    return weight if weight > 0 else 1.0


def _completion_percent_for_label(label: ComplianceLabel | None) -> int:
    if label == "satisfied":
        return 100
    if label == "partial":
        return 33
    return 0


def _has_llm_overclaim(unit: EvaluationUnit) -> bool:
    if unit.final_label != "not_satisfied":
        return False
    return any(
        stage.label == "satisfied"
        for stage in (unit.stage_1_answer, unit.stage_2_answer, unit.stage_3_answer)
    )


def _compute_unit_stage_alignment(unit: EvaluationUnit) -> float:
    labels = [
        stage.label
        for stage in (unit.stage_1_answer, unit.stage_2_answer, unit.stage_3_answer)
        if stage.label
    ]
    if unit.final_label:
        labels.append(unit.final_label)
    if len(labels) <= 1:
        return 1.0
    unique_labels = set(labels)
    if len(unique_labels) == 1:
        return 1.0
    final_label = unit.final_label
    if final_label and any(stage_label == final_label for stage_label in labels[:-1]):
        return 0.5
    if len(unique_labels) == 2:
        return 0.5
    return 0.0


def _resolve_debug_rationale(unit: EvaluationUnit) -> str:
    if unit.final_rationale:
        return unit.final_rationale
    for stage in (unit.stage_3_answer, unit.stage_2_answer, unit.stage_1_answer):
        if stage.rationale:
            return stage.rationale
    return ""


def _resolve_debug_grounded_evidence_count(unit: EvaluationUnit) -> int:
    return _count_grounded_record_evidence(
        stage_judgments=(unit.stage_1_answer, unit.stage_2_answer, unit.stage_3_answer),
        record_nodes=unit.record_evidence_chunks,
    )


def _resolve_stage_grounded_evidence_count(
    *,
    stage_judgment: StageJudgment,
    record_nodes: Sequence[EvidenceNode],
) -> int:
    return len(
        _resolve_stage_grounded_record_evidence_items(
            stage_judgment=stage_judgment,
            record_nodes=record_nodes,
        )
    )


def _resolve_debug_grounded_chunk_count(unit: EvaluationUnit) -> int:
    grounded_nodes = _resolve_grounded_record_nodes(
        stage_judgments=(unit.stage_1_answer, unit.stage_2_answer, unit.stage_3_answer),
        record_nodes=unit.record_evidence_chunks,
    )
    return len(grounded_nodes)


def _resolve_debug_has_conflict(unit: EvaluationUnit) -> bool:
    if unit.evidence_status == "conflicting":
        return True
    return _contradiction_type_implies_conflict(unit.contradiction_type)


def _resolve_debug_subsection_ids(unit: EvaluationUnit) -> list[str]:
    return sorted(
        {
            node.subsection_id
            for node in unit.record_evidence_chunks
            if node.subsection_id
        }
    )


def _resolve_debug_grounded_subsection_ids(unit: EvaluationUnit) -> list[str]:
    grounded_nodes = _resolve_grounded_record_nodes(
        stage_judgments=(unit.stage_1_answer, unit.stage_2_answer, unit.stage_3_answer),
        record_nodes=unit.record_evidence_chunks,
    )
    return sorted(
        {
            node.subsection_id
            for node in grounded_nodes
            if node.subsection_id
        }
    )


def _resolve_debug_subsection_coverage_ratio(unit: EvaluationUnit) -> float:
    return round(
        _compute_subsection_coverage_ratio_from_nodes(
            grounded_nodes=_resolve_grounded_record_nodes(
                stage_judgments=(unit.stage_1_answer, unit.stage_2_answer, unit.stage_3_answer),
                record_nodes=unit.record_evidence_chunks,
            ),
            record_nodes=unit.record_evidence_chunks,
        ),
        4,
    )


def _resolve_debug_evidence_coverage_ratio(unit: EvaluationUnit) -> float:
    return _compute_evidence_coverage_ratio(
        grounded_evidence_count=_resolve_debug_grounded_evidence_count(unit),
        required_evidence_count=int(unit.required_evidence_count or 0),
    )


def _resolve_stage_evidence_status(
    *,
    stage_judgment: StageJudgment,
    grounded_evidence_count: int,
    required_evidence_count: int,
) -> str:
    return _resolve_evidence_status(
        grounded_record_evidence_count=grounded_evidence_count,
        required_evidence_count=required_evidence_count,
        conflict_detected=bool(stage_judgment.conflict_flag),
    )


def _compute_evidence_coverage_ratio(
    *,
    grounded_evidence_count: int,
    required_evidence_count: int,
) -> float:
    if required_evidence_count <= 0:
        return 0.0
    return round(grounded_evidence_count / required_evidence_count, 4)


def _compute_subsection_coverage_ratio_from_nodes(
    *,
    grounded_nodes: Sequence[EvidenceNode],
    record_nodes: Sequence[EvidenceNode],
) -> float:
    subsection_ids = {
        node.subsection_id
        for node in record_nodes
        if node.subsection_id
    }
    if not subsection_ids:
        return 0.0
    grounded_subsection_ids = {
        node.subsection_id
        for node in grounded_nodes
        if node.subsection_id
    }
    return len(grounded_subsection_ids) / len(subsection_ids)


def _resolve_debug_subsection_downgrade_applied(unit: EvaluationUnit) -> bool:
    if unit.evidence_status != "supported":
        return False
    return unit.final_label != "satisfied"


def _build_count_distribution(values: Iterable[int]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def _contradiction_type_implies_conflict(contradiction_type: ContradictionType) -> bool:
    if contradiction_type in {"none", "missing_evidence", "reference_clarification"}:
        return False
    return True


def _is_suspicious_debug_row(row: dict[str, Any]) -> bool:
    final_label = row.get("final_label")
    evidence_status = row.get("evidence_status")
    contradiction_type = row.get("contradiction_type")
    grounded_evidence_count = int(row.get("grounded_evidence_count") or 0)
    required_evidence_count = int(row.get("required_evidence_count") or 0)
    has_conflict = bool(row.get("has_conflict"))

    if final_label == "satisfied" and grounded_evidence_count == 0:
        return True
    if evidence_status == "supported" and grounded_evidence_count < required_evidence_count:
        return True
    if evidence_status == "missing" and grounded_evidence_count > 0:
        return True
    if has_conflict and contradiction_type == "none":
        return True
    if (
        contradiction_type not in {"none", "missing_evidence", "reference_conflict", "reference_clarification"}
        and not has_conflict
    ):
        return True
    return False


def _build_mini_kg_links(
    *,
    deliverable_id: str,
    stage_judgments: Sequence[StageJudgment],
    record_nodes: Sequence[EvidenceNode],
    reference_nodes: Sequence[ReferenceNode],
) -> MiniKGLinks:
    stage_lookup = {stage.stage_key: stage for stage in stage_judgments}
    present_stage_keys = [
        stage.stage_key
        for stage in stage_judgments
        if stage.label or stage.rationale or stage.supporting_record_evidence_ids or stage.supporting_reference_ids
    ]
    return MiniKGLinks(
        deliverable_id=deliverable_id,
        stage_judgment_keys=present_stage_keys,
        record_evidence_ids=[item.evidence_id for item in record_nodes if item.evidence_id],
        reference_ids=[item.reference_id for item in reference_nodes if item.reference_id],
        stage_1_record_evidence_ids=stage_lookup.get("stage_1", StageJudgment(stage_key="stage_1")).supporting_record_evidence_ids,
        stage_2_record_evidence_ids=stage_lookup.get("stage_2", StageJudgment(stage_key="stage_2")).supporting_record_evidence_ids,
        stage_3_record_evidence_ids=stage_lookup.get("stage_3", StageJudgment(stage_key="stage_3")).supporting_record_evidence_ids,
        stage_3_reference_ids=stage_lookup.get("stage_3", StageJudgment(stage_key="stage_3")).supporting_reference_ids,
    )


def _compute_stage_alignment(stage_judgments: Sequence[StageJudgment]) -> float | None:
    labels = [stage.label for stage in stage_judgments if stage.label]
    if not labels:
        return None
    if len(labels) == 1:
        return 1.0
    aligned_pairs = 0
    total_pairs = 0
    for index, left in enumerate(labels):
        for right in labels[index + 1:]:
            total_pairs += 1
            if left == right:
                aligned_pairs += 1
    if total_pairs <= 0:
        return 1.0
    return round(aligned_pairs / total_pairs, 4)


def _compute_retrieval_support_rate(stage_judgments: Sequence[StageJudgment]) -> float:
    labeled_stages = [stage for stage in stage_judgments if stage.label]
    if not labeled_stages:
        return 0.0
    supported = sum(
        1
        for stage in labeled_stages
        if stage.supporting_record_evidence_ids or stage.supporting_reference_ids
    )
    return round(supported / len(labeled_stages), 4)


def _compute_llm_overclaim_rate(
    *,
    evidence_score: float | None,
    stage_judgments: Sequence[StageJudgment],
) -> float:
    positive_stages = [
        stage
        for stage in stage_judgments
        if stage.label in {"satisfied", "partial"}
    ]
    if not positive_stages:
        return 0.0
    unsupported_positive_stages = sum(
        1
        for stage in positive_stages
        if not stage.supporting_record_evidence_ids and not stage.supporting_reference_ids
    )
    if evidence_score is not None and evidence_score <= 0.0:
        unsupported_positive_stages = len(positive_stages)
    return round(unsupported_positive_stages / len(positive_stages), 4)


def _count_grounded_record_evidence(
    *,
    stage_judgments: Sequence[StageJudgment],
    record_nodes: Sequence[EvidenceNode],
) -> int:
    return len(
        _resolve_grounded_record_evidence_items(
            stage_judgments=stage_judgments,
            record_nodes=record_nodes,
        )
    )


def _log_grounding_selection_debug(
    *,
    deliverable_id: str,
    record_nodes: Sequence[EvidenceNode],
    grounded_nodes: Sequence[EvidenceNode],
) -> None:
    grounded_ids = {node.evidence_id for node in grounded_nodes if node.evidence_id}
    print(
        {
            "stage": "evaluation_v3.builder.grounding_selection",
            "deliverable_id": deliverable_id,
            "chunks": [
                {
                    "raw_retrieval_score": node.raw_retrieval_score,
                    "reranker_score": node.reranker_score,
                    "retrieval_score": node.retrieval_score,
                    "selected_as_grounded": node.evidence_id in grounded_ids,
                }
                for node in record_nodes
            ],
        }
    )


def _resolve_grounded_record_nodes(
    *,
    stage_judgments: Sequence[StageJudgment],
    record_nodes: Sequence[EvidenceNode],
    threshold: float | None = None,
) -> list[EvidenceNode]:
    resolved_threshold = (
        evaluation_v3_config["GROUNDING_SCORE_THRESHOLD"]
        if threshold is None
        else threshold
    )
    scored_nodes = [
        node
        for node in record_nodes
        if node.retrieval_score is not None
    ]
    max_score = max((float(node.retrieval_score) for node in scored_nodes), default=None)
    stage_lookup = {stage.stage_key: stage for stage in stage_judgments}
    stage_3 = stage_lookup.get("stage_3")
    stage_2 = stage_lookup.get("stage_2")
    accepted_record_ids: list[str] = []
    grounding_source = "none"
    if stage_3 and stage_3.supporting_record_evidence_ids:
        accepted_record_ids = list(stage_3.supporting_record_evidence_ids)
        grounding_source = "stage_3"
    elif stage_2 and stage_2.supporting_record_evidence_ids:
        accepted_record_ids = list(stage_2.supporting_record_evidence_ids)
        grounding_source = "stage_2"

    accepted_record_id_set = {
        evidence_id
        for evidence_id in accepted_record_ids
        if evidence_id
    }
    grounded_nodes = [
        node
        for node in record_nodes
        if node.evidence_id and node.evidence_id in accepted_record_id_set
    ]
    grounded_count_before_fallback = len(grounded_nodes)
    fallback_applied = False
    print(
        {
            "max_score": max_score,
            "threshold": float(resolved_threshold),
            "top_n": int(evaluation_v3_config["GROUNDING_TOP_N"]),
            "uses_top_n_grounding": False,
            "grounded_count_before_fallback": grounded_count_before_fallback,
            "fallback_applied": fallback_applied,
            "grounding_source": grounding_source,
            "accepted_record_evidence_ids": accepted_record_ids,
        }
    )
    return grounded_nodes


def _resolve_grounded_record_evidence_items(
    *,
    stage_judgments: Sequence[StageJudgment],
    record_nodes: Sequence[EvidenceNode],
) -> list[dict[str, str]]:
    stage_lookup = {stage.stage_key: stage for stage in stage_judgments}
    stage_3 = stage_lookup.get("stage_3")
    stage_2 = stage_lookup.get("stage_2")
    selected_stage = (
        stage_3
        if stage_3 and (stage_3.supporting_record_evidence_items or stage_3.supporting_record_evidence_ids)
        else stage_2
        if stage_2 and (stage_2.supporting_record_evidence_items or stage_2.supporting_record_evidence_ids)
        else None
    )
    if selected_stage is None:
        return []
    accepted_items = list(selected_stage.supporting_record_evidence_items)
    if not accepted_items:
        accepted_record_id_set = {
            evidence_id
            for evidence_id in selected_stage.supporting_record_evidence_ids
            if evidence_id
        }
        return [
            {
                "evidence_id": node.evidence_id,
                "section_id": node.section_id,
                "subsection_id": node.subsection_id,
                "section_label": node.section_label,
                "heading_title": node.heading_title,
                "source_document": node.source_document,
                "text": node.text,
            }
            for node in record_nodes
            if node.evidence_id and node.evidence_id in accepted_record_id_set
        ]
    grounded_items: list[dict[str, str]] = []
    for item in accepted_items:
        if _match_record_item_to_nodes(item, record_nodes):
            grounded_items.append(item)
    return grounded_items


def _merge_stage_3_grounded_record_evidence(
    *,
    stage_2: StageJudgment,
    stage_3: StageJudgment,
) -> StageJudgment:
    if stage_3.conflict_flag:
        return stage_3

    merged_ids = _dedupe(
        [
            *list(stage_2.supporting_record_evidence_ids),
            *list(stage_3.supporting_record_evidence_ids),
        ]
    )
    merged_items = _merge_stage_record_evidence_items(
        primary_items=list(stage_3.supporting_record_evidence_items),
        fallback_items=list(stage_2.supporting_record_evidence_items),
    )
    return stage_3.model_copy(
        update={
            "supporting_record_evidence_ids": merged_ids,
            "supporting_record_evidence_items": merged_items,
        }
    )


def _merge_stage_record_evidence_items(
    *,
    primary_items: list[dict[str, str]],
    fallback_items: list[dict[str, str]],
) -> list[dict[str, str]]:
    merged_items: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in [*primary_items, *fallback_items]:
        key = _stage_record_item_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        merged_items.append(item)
    return merged_items


def _stage_record_item_key(item: dict[str, str]) -> str:
    evidence_id = str(item.get("evidence_id") or "").strip()
    if evidence_id:
        return f"id:{evidence_id}"
    text = str(item.get("text") or "").strip().lower()
    source_document = str(item.get("source_document") or "").strip().lower()
    if not text:
        return ""
    return f"text:{source_document}|{text}"


def _resolve_stage_grounded_record_evidence_items(
    *,
    stage_judgment: StageJudgment,
    record_nodes: Sequence[EvidenceNode],
) -> list[dict[str, str]]:
    accepted_items = list(stage_judgment.supporting_record_evidence_items)
    if not accepted_items:
        accepted_record_id_set = {
            evidence_id
            for evidence_id in stage_judgment.supporting_record_evidence_ids
            if evidence_id
        }
        return [
            {
                "evidence_id": node.evidence_id,
                "section_id": node.section_id,
                "subsection_id": node.subsection_id,
                "section_label": node.section_label,
                "heading_title": node.heading_title,
                "source_document": node.source_document,
                "text": node.text,
            }
            for node in record_nodes
            if node.evidence_id and node.evidence_id in accepted_record_id_set
        ]
    grounded_items: list[dict[str, str]] = []
    for item in accepted_items:
        if _match_record_item_to_nodes(item, record_nodes):
            grounded_items.append(item)
    return grounded_items


def _detect_conflict(
    *,
    stage_judgments: Sequence[StageJudgment],
    verifier_input: dict[str, Any] | None,
) -> bool:
    if any(stage.conflict_flag for stage in stage_judgments):
        return True
    if not isinstance(verifier_input, dict):
        return False
    if _is_truthy_flag(verifier_input.get("conflict_flag")):
        return True
    if _is_truthy_flag(verifier_input.get("has_conflict")):
        return True
    if _is_truthy_flag(verifier_input.get("conflicting")):
        return True
    contradiction_value = str(verifier_input.get("contradiction_type") or "").strip().lower()
    return contradiction_value not in {"", "none", "missing_evidence", "reference_clarification"}


def _extract_conflict_flag(payload: dict[str, Any]) -> bool:
    if _is_truthy_flag(payload.get("conflict_flag")):
        return True
    if _is_truthy_flag(payload.get("has_conflict")):
        return True
    if _is_truthy_flag(payload.get("conflicting")):
        return True
    contradiction_value = str(payload.get("contradiction_type") or "").strip().lower()
    return contradiction_value not in {"", "none", "missing_evidence", "reference_clarification"}


def _resolve_contradiction_type(
    *,
    explicit_contradiction_type: ContradictionType,
    base_evidence_status: str,
    final_label: ComplianceLabel | None,
    stage_judgments: Sequence[StageJudgment],
    record_nodes: Sequence[EvidenceNode],
    verifier_input: dict[str, Any] | None,
) -> ContradictionType:
    if explicit_contradiction_type != "none":
        return explicit_contradiction_type
    if _has_direct_conflict_signal(
        stage_judgments=stage_judgments,
        record_nodes=record_nodes,
        verifier_input=verifier_input,
    ):
        return "direct_conflict"
    if _has_reference_conflict(stage_judgments):
        return "reference_conflict"
    if _has_reference_clarification(stage_judgments):
        return "reference_clarification"
    if _claims_satisfied_without_grounded_evidence(
        base_evidence_status=base_evidence_status,
        final_label=final_label,
        stage_judgments=stage_judgments,
    ):
        return "missing_evidence"
    return "none"


def _claims_satisfied_without_grounded_evidence(
    *,
    base_evidence_status: str,
    final_label: ComplianceLabel | None,
    stage_judgments: Sequence[StageJudgment],
) -> bool:
    if base_evidence_status != "missing":
        return False
    if final_label == "satisfied":
        return True
    return any(stage.label == "satisfied" for stage in stage_judgments)


def _has_direct_conflict_signal(
    *,
    stage_judgments: Sequence[StageJudgment],
    record_nodes: Sequence[EvidenceNode],
    verifier_input: dict[str, Any] | None,
) -> bool:
    for node in record_nodes:
        if _contains_direct_conflict_marker(node.text):
            return True
    for stage in stage_judgments:
        if _contains_direct_conflict_marker(stage.rationale):
            return True
    if not isinstance(verifier_input, dict):
        return False
    for key in ("notes", "rationale", "summary"):
        if _contains_direct_conflict_marker(verifier_input.get(key)):
            return True
    return False


def _has_reference_conflict(stage_judgments: Sequence[StageJudgment]) -> bool:
    stage_lookup = {stage.stage_key: stage for stage in stage_judgments}
    stage_3 = stage_lookup.get("stage_3")
    if stage_3 is None or not stage_3.supporting_reference_ids:
        return False
    return _contains_reference_conflict_marker(stage_3.rationale)


def _has_reference_clarification(stage_judgments: Sequence[StageJudgment]) -> bool:
    stage_lookup = {stage.stage_key: stage for stage in stage_judgments}
    stage_3 = stage_lookup.get("stage_3")
    if stage_3 is None or not stage_3.supporting_reference_ids:
        return False
    return not _contains_reference_conflict_marker(stage_3.rationale)


def _contains_direct_conflict_marker(value: object) -> bool:
    normalized = _normalized_match_key(value)
    if not normalized:
        return False
    return any(marker in normalized for marker in CONFLICT_MARKERS)


def _contains_reference_conflict_marker(value: object) -> bool:
    normalized = _normalized_match_key(value)
    if not normalized:
        return False
    return any(marker in normalized for marker in REFERENCE_CONFLICT_MARKERS)


def _match_items_to_record_nodes(items: list[Any], record_nodes: Sequence[EvidenceNode]) -> list[str]:
    matched_ids: list[str] = []
    for item in items:
        if isinstance(item, dict):
            explicit_id = str(item.get("evidence_id") or "").strip()
            if explicit_id:
                matched_ids.append(explicit_id)
                continue
            matched_ids.extend(_match_record_item_to_nodes(item, record_nodes))
    return _dedupe(matched_ids)


def _match_items_to_reference_nodes(items: list[Any], reference_nodes: Sequence[ReferenceNode]) -> list[str]:
    matched_ids: list[str] = []
    for item in items:
        if isinstance(item, dict):
            explicit_id = str(item.get("reference_id") or "").strip()
            if explicit_id:
                matched_ids.append(explicit_id)
                continue
            text = item.get("text")
            if text:
                matched_ids.extend(_match_texts_to_reference_nodes([str(text)], reference_nodes))
    return _dedupe(matched_ids)


def _match_texts_to_record_nodes(texts: Iterable[str], record_nodes: Sequence[EvidenceNode]) -> list[str]:
    matched_ids: list[str] = []
    for text in texts:
        matched_ids.extend(_match_record_text_to_nodes(text, record_nodes))
    return _dedupe(matched_ids)


def _match_texts_to_reference_nodes(texts: Iterable[str], reference_nodes: Sequence[ReferenceNode]) -> list[str]:
    matched_ids: list[str] = []
    for text in texts:
        normalized_text = _normalized_match_key(text)
        if not normalized_text:
            continue
        for node in reference_nodes:
            if _is_text_match(normalized_text, _normalized_match_key(node.text)):
                if node.reference_id:
                    matched_ids.append(node.reference_id)
    return _dedupe(matched_ids)


def _extract_text_candidates(payload: dict[str, Any], *, keys: Sequence[str]) -> list[str]:
    collected: list[str] = []
    for key in keys:
        raw_value = payload.get(key)
        if isinstance(raw_value, str) and raw_value.strip():
            collected.append(raw_value)
        elif isinstance(raw_value, list):
            for item in raw_value:
                if isinstance(item, str) and item.strip():
                    collected.append(item)
    return collected


def _normalized_match_key(value: object) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def _is_text_match(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left == right or left in right or right in left


def _match_record_item_to_nodes(item: dict[str, Any], record_nodes: Sequence[EvidenceNode]) -> list[str]:
    text = item.get("text")
    if not text:
        return []
    metadata = {
        "source_document": str(item.get("source_document") or "").strip(),
        "section_id": str(item.get("section_id") or "").strip(),
        "subsection_id": str(item.get("subsection_id") or "").strip(),
        "section_label": str(item.get("section_label") or "").strip(),
        "heading_title": str(item.get("heading_title") or "").strip(),
    }
    return _match_record_text_to_nodes(str(text), record_nodes, metadata=metadata)


def _match_record_text_to_nodes(
    text: str,
    record_nodes: Sequence[EvidenceNode],
    *,
    metadata: dict[str, str] | None = None,
) -> list[str]:
    normalized_text = _normalized_match_key(text)
    if not normalized_text:
        return []

    exact_matches: list[str] = []
    candidate_scores: list[tuple[int, int, str]] = []
    for node in record_nodes:
        if not node.evidence_id:
            continue
        normalized_node_text = _normalized_match_key(node.text)
        if _is_text_match(normalized_text, normalized_node_text):
            exact_matches.append(node.evidence_id)
            continue

        overlap_score = _token_overlap_match_score(text, node.text)
        metadata_score = _record_metadata_match_score(metadata, node)
        if overlap_score <= 0 and metadata_score <= 0:
            continue
        candidate_scores.append((metadata_score, overlap_score, node.evidence_id))

    if exact_matches:
        return _dedupe(exact_matches)

    if not candidate_scores:
        return []

    candidate_scores.sort(reverse=True)
    best_metadata_score, best_overlap_score, _ = candidate_scores[0]
    if best_overlap_score <= 0:
        return []

    return _dedupe(
        evidence_id
        for metadata_score, overlap_score, evidence_id in candidate_scores
        if metadata_score == best_metadata_score and overlap_score == best_overlap_score
    )


def _token_overlap_match_score(left: str, right: str) -> int:
    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)
    if not left_tokens or not right_tokens:
        return 0

    overlapping_tokens = set(left_tokens) & set(right_tokens)
    overlap_count = len(overlapping_tokens)
    required_overlap_count = min(5, len(set(left_tokens)))
    overlap_ratio = overlap_count / max(1, len(set(left_tokens)))
    if overlap_ratio < 0.65:
        return 0
    if overlap_count < required_overlap_count:
        return 0
    return overlap_count


def _meaningful_tokens(value: object) -> list[str]:
    normalized = _normalized_match_key(value)
    if not normalized:
        return []
    punctuation_removed = re.sub(r"[^\w\s]", " ", normalized)
    return [
        token
        for token in punctuation_removed.split()
        if len(token) >= 3
    ]


def _record_metadata_match_score(metadata: dict[str, str] | None, node: EvidenceNode) -> int:
    if not metadata:
        return 0
    score = 0
    field_pairs = (
        ("source_document", node.source_document),
        ("section_id", node.section_id),
        ("subsection_id", node.subsection_id),
        ("section_label", node.section_label),
        ("heading_title", node.heading_title),
    )
    for key, node_value in field_pairs:
        left = _normalized_match_key(metadata.get(key))
        right = _normalized_match_key(node_value)
        if left and right and left == right:
            score += 1
    return score


def _resolve_chunk_text(item: dict[str, Any]) -> str:
    text = _pick_first(item, "text", "quote", "content", "excerpt")
    table_markdown = _pick_first(item, "table_markdown")
    if text and table_markdown:
        return f"{text}\n{table_markdown}".strip()
    if text:
        return text
    if table_markdown:
        return table_markdown
    return ""


def _resolve_retrieval_score(item: dict[str, Any]) -> float | None:
    retrieval_score = item.get("retrieval_score")
    if isinstance(retrieval_score, (int, float)):
        return min(1.0, max(0.0, float(retrieval_score)))
    return None


def _resolve_raw_retrieval_score(item: dict[str, Any]) -> float | None:
    for key in ("raw_retrieval_score", "reranker_score", "faiss_score"):
        raw_value = item.get(key)
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
    return None


def _resolve_record_evidence_id(item: dict[str, Any], deliverable_id: str, index: int) -> str:
    explicit_id = _pick_first(item, "evidence_id", "id", "chunk_id")
    if explicit_id:
        return explicit_id
    return _build_evidence_id(deliverable_id, "record", index)


def _resolve_reference_id(item: dict[str, Any], deliverable_id: str, index: int) -> str:
    explicit_id = _pick_first(item, "reference_id", "id", "chunk_id")
    if explicit_id:
        return explicit_id
    return _build_evidence_id(deliverable_id, "reference", index)


def _build_evidence_id(deliverable_id: str, prefix: str, index: int) -> str:
    return f"{deliverable_id}:{prefix}:{index}"


def _pick_first(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalize_label(value: object) -> ComplianceLabel | None:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"", "none"}:
        return None
    alias_map = {
        "matched": "satisfied",
        "unmatched": "not_satisfied",
        "unsatisfied": "not_satisfied",
        "not_matched": "not_satisfied",
        "no_match": "not_satisfied",
    }
    normalized = alias_map.get(normalized, normalized)
    if normalized in {"satisfied", "partial", "not_satisfied"}:
        return normalized
    return None


def _normalize_id_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return _dedupe([str(item).strip() for item in value if str(item).strip()])


def _is_truthy_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value or "").strip().lower()
    return normalized in {"true", "1", "yes", "y"}




def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _serialize_json(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, ensure_ascii=False)
