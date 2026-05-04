from __future__ import annotations

import csv
import hashlib
import re
from typing import Any, Iterable, Sequence
from .config import evaluation_v3_config
from .schemas import (
    ComplianceLabel,
    ContradictionType,
    DeliverableNode,
    EvidencePipelineCounters,
    EvidenceRef,
    QuoteElementMappingDebug,
    EVALUATION_V3_ANALYSIS_METRICS,
    EvaluationV3Result,
    EvaluationV3ResultRow,
    RequirementElement,
    RequirementElementSupport,
    StageElementAssessment,
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
    requirement_elements = _build_placeholder_requirement_elements(
        deliverable=deliverable,
        requirement_type=requirement_type,
    )
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
        deliverable_id=deliverable.deliverable_id,
        stage_judgments=(stage_1, stage_2, stage_3),
        record_nodes=record_nodes,
    )
    stage_1_grounded_items = _resolve_stage_grounded_record_evidence_items(
        deliverable_id=deliverable.deliverable_id,
        stage_judgment=stage_1,
        record_nodes=record_nodes,
    )
    stage_2_grounded_items = _resolve_stage_grounded_record_evidence_items(
        deliverable_id=deliverable.deliverable_id,
        stage_judgment=stage_2,
        record_nodes=record_nodes,
    )
    stage_3_grounded_items = _resolve_stage_grounded_record_evidence_items(
        deliverable_id=deliverable.deliverable_id,
        stage_judgment=stage_3,
        record_nodes=record_nodes,
    )
    final_grounded_items = _resolve_grounded_record_evidence_items(
        deliverable_id=deliverable.deliverable_id,
        stage_judgments=(stage_1, stage_2, stage_3),
        record_nodes=record_nodes,
    )
    stage_1_grounded_evidence_count = _resolve_stage_grounded_evidence_count(
        deliverable_id=deliverable.deliverable_id,
        stage_judgment=stage_1,
        record_nodes=record_nodes,
    )
    stage_2_grounded_evidence_count = _resolve_stage_grounded_evidence_count(
        deliverable_id=deliverable.deliverable_id,
        stage_judgment=stage_2,
        record_nodes=record_nodes,
    )
    stage_3_grounded_evidence_count = _resolve_stage_grounded_evidence_count(
        deliverable_id=deliverable.deliverable_id,
        stage_judgment=stage_3,
        record_nodes=record_nodes,
    )
    stage_1_element_assessment = _build_placeholder_stage_element_assessment(
        stage_key="stage_1",
        requirement_elements=requirement_elements,
        grounded_record_evidence_items=stage_1_grounded_items,
        grounded_record_nodes=_resolve_stage_grounded_record_nodes(
            stage_judgment=stage_1,
            record_nodes=record_nodes,
        ),
        conflict_flag=stage_1.conflict_flag,
    )
    stage_2_element_assessment = _build_placeholder_stage_element_assessment(
        stage_key="stage_2",
        requirement_elements=requirement_elements,
        grounded_record_evidence_items=stage_2_grounded_items,
        grounded_record_nodes=_resolve_stage_grounded_record_nodes(
            stage_judgment=stage_2,
            record_nodes=record_nodes,
        ),
        conflict_flag=stage_2.conflict_flag,
    )
    stage_3_element_assessment = _build_placeholder_stage_element_assessment(
        stage_key="stage_3",
        requirement_elements=requirement_elements,
        grounded_record_evidence_items=stage_3_grounded_items,
        grounded_record_nodes=_resolve_stage_grounded_record_nodes(
            stage_judgment=stage_3,
            record_nodes=record_nodes,
        ),
        conflict_flag=stage_3.conflict_flag,
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
    final_element_assessment = _build_placeholder_stage_element_assessment(
        stage_key="final",
        requirement_elements=requirement_elements,
        grounded_record_evidence_items=final_grounded_items,
        grounded_record_nodes=grounded_record_nodes,
        conflict_flag=conflict_detected,
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
            requirement_elements=requirement_elements,
            stage_1_element_assessment=stage_1_element_assessment,
            stage_2_element_assessment=stage_2_element_assessment,
            stage_3_element_assessment=stage_3_element_assessment,
            final_element_assessment=final_element_assessment,
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
        deliverable_id=deliverable.deliverable_id,
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
        requirement_elements=requirement_elements,
        stage_1_element_assessment=stage_1_element_assessment,
        stage_2_element_assessment=stage_2_element_assessment,
        stage_3_element_assessment=stage_3_element_assessment,
        final_element_assessment=final_element_assessment,
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


def _build_placeholder_requirement_elements(
    *,
    deliverable: DeliverableNode,
    requirement_type: RequirementType,
) -> list[RequirementElement]:
    requirement_text = " ".join(str(deliverable.requirement_text or "").split()).strip()
    element_texts = _decompose_requirement_text(
        requirement_text=requirement_text,
        requirement_type=requirement_type,
    )
    return [
        RequirementElement(
            element_id=f"{deliverable.deliverable_id}:element:{index}",
            element_text=element_text,
            element_type=requirement_type,
            required=True,
        )
        for index, element_text in enumerate(element_texts, start=1)
    ]


def _decompose_requirement_text(
    *,
    requirement_text: str,
    requirement_type: RequirementType,
) -> list[str]:
    text = " ".join(str(requirement_text or "").split()).strip()
    if not text:
        return [""]

    if requirement_type in {"single_field", "generic"}:
        clauses = _split_requirement_clauses(text)
        return clauses[:3] if clauses else [text]

    if requirement_type == "relationship":
        concept_elements = _build_relationship_elements(text)
        return concept_elements[:3] if concept_elements else [text]

    if requirement_type == "list_or_table":
        return [
            f"Presence of the required list/table structure: {text}",
            f"Required listed content is included in that list/table: {text}",
        ]

    if requirement_type == "per_function":
        attribute_label = _infer_per_function_attribute_label(text)
        return [
            f"Relevant functions/categories are explicitly identified: {text}",
            f"The required {attribute_label} is documented for each identified function/category: {text}",
        ]

    if requirement_type == "control_measure":
        return [
            f"Required control measure is present or specified: {text}",
            f"Implementation, effect, or risk-reduction evidence is documented for that control measure: {text}",
        ]

    if requirement_type == "benefit_risk_rationale":
        return [
            f"Benefit-risk trigger/context is stated: {text}",
            f"Benefit-risk rationale, details, or consequences are documented: {text}",
        ]

    if requirement_type == "residual_risk_acceptability":
        return [
            f"Residual risk statement is documented: {text}",
            f"Residual risk acceptability conclusion is documented: {text}",
        ]

    if requirement_type == "conditional":
        condition, outcome = _split_conditional_requirement(text)
        if condition and outcome:
            return [
                f"Condition or trigger is defined: {condition}",
                f"Required action or outcome is documented: {outcome}",
            ]
        return [
            f"Condition or trigger is defined: {text}",
            f"Required action or outcome is documented: {text}",
        ]

    return [text]


def _split_requirement_clauses(text: str) -> list[str]:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return []

    explicit_parts = [
        part.strip(" ,.;:")
        for part in re.split(r"\s*;\s*|\s+\b(?:as well as|together with)\b\s+", normalized, flags=re.IGNORECASE)
        if part.strip(" ,.;:")
    ]
    if len(explicit_parts) > 1:
        return explicit_parts

    lowered = normalized.lower()
    if any(marker in lowered for marker in (" and ", " including ", " include ", " includes ")):
        parts = [
            part.strip(" ,.;:")
            for part in re.split(r"\s+\b(?:and|including|include|includes)\b\s+", normalized, maxsplit=2, flags=re.IGNORECASE)
            if part.strip(" ,.;:")
        ]
        if 1 < len(parts) <= 3:
            return parts

    return [normalized]


def _build_relationship_elements(text: str) -> list[str]:
    normalized = " ".join(text.split()).strip()
    concept_parts = _split_relationship_concepts(normalized)
    elements: list[str] = []
    for concept in concept_parts[:2]:
        elements.append(f"Concept is explicitly identified: {concept}")
    elements.append(f"Required relationship between the identified concepts is documented: {normalized}")
    return _dedupe(elements)[:3]


def _split_relationship_concepts(text: str) -> list[str]:
    patterns = (
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[.,;:]|$)",
        r"\bfrom\s+(.+?)\s+and\s+(.+?)(?:[.,;:]|$)",
        r"\bof\s+(.+?)\s+and\s+(.+?)(?:[.,;:]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return [match.group(1).strip(" ,.;:"), match.group(2).strip(" ,.;:")]

    clauses = _split_requirement_clauses(text)
    if len(clauses) >= 2:
        return clauses[:2]
    return [text]


def _split_conditional_requirement(text: str) -> tuple[str, str]:
    normalized = " ".join(text.split()).strip()
    patterns = (
        r"^(in cases where .+?),(?:\s*)(.+)$",
        r"^(if .+?),(?:\s*)(.+)$",
        r"^(when .+?),(?:\s*)(.+)$",
        r"^(where .+?),(?:\s*)(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" ,.;:"), match.group(2).strip(" ,.;:")
    return "", ""


def _build_placeholder_stage_element_assessment(
    *,
    stage_key: str,
    requirement_elements: Sequence[RequirementElement],
    grounded_record_evidence_items: Sequence[dict[str, str]],
    grounded_record_nodes: Sequence[EvidenceNode],
    conflict_flag: bool,
) -> StageElementAssessment:
    required_elements = [element for element in requirement_elements if element.required]
    elements: list[RequirementElementSupport] = []
    quote_mapping_debug = _build_quote_element_mapping_debug(
        requirement_elements=required_elements,
        grounded_record_evidence_items=grounded_record_evidence_items,
    )
    for element in requirement_elements:
        mapped_items = _map_grounded_quotes_to_requirement_element(
            element=element,
            grounded_record_evidence_items=grounded_record_evidence_items,
        ) if element.required else []
        conflict_details = _detect_requirement_element_conflicts(
            element=element,
            requirement_elements=requirement_elements,
            grounded_record_evidence_items=grounded_record_evidence_items,
            grounded_record_nodes=grounded_record_nodes,
        ) if element.required else None
        
        supporting_refs = _build_evidence_refs(items=mapped_items, element_id=element.element_id)
        conflicting_refs = _build_evidence_refs(
            items=(conflict_details or {}).get("items", []),
            element_id=element.element_id,
            conflict_type=(conflict_details or {}).get("conflict_type"),
            conflict_reason=(conflict_details or {}).get("conflict_reason"),
        )
        # Derive arrays from refs for status determination
        supporting_quotes = [ref.quote for ref in supporting_refs]
        supporting_ids = [ref.evidence_id for ref in supporting_refs]
        conflicting_quotes = [ref.quote for ref in conflicting_refs]
        conflicting_ids = [ref.evidence_id for ref in conflicting_refs]
        
        has_detected_conflict = bool(conflicting_ids or conflicting_quotes)
        if conflict_flag or has_detected_conflict:
            element_status = "contradicted"
        elif supporting_quotes:
            element_status = "supported"
        elif grounded_record_evidence_items:
            element_status = "weak_match"
        else:
            element_status = "missing"
        
        # For weak_match, include all grounded items that didn't support or conflict
        weak_match_items = []
        if element.required and element_status == "weak_match":
            # Find items that are grounded but not in supporting or conflicting
            supporting_quote_set = set(supporting_quotes)
            conflicting_quote_set = set(conflicting_quotes)
            for item in grounded_record_evidence_items:
                quote = str(item.get("text") or "").strip()
                if quote and quote not in supporting_quote_set and quote not in conflicting_quote_set:
                    weak_match_items.append(item)
        weak_match_refs = _build_evidence_refs(
            items=weak_match_items,
            element_id=element.element_id,
            weak_match_reason="available_evidence_did_not_support_element"
        )
        
        elements.append(
            RequirementElementSupport(
                element_id=element.element_id,
                element_text=element.element_text,
                element_type=element.element_type,
                required=element.required,
                supporting_evidence_refs=supporting_refs,
                conflicting_evidence_refs=conflicting_refs,
                weak_match_evidence_refs=weak_match_refs,
                element_status=element_status if element.required else "missing",
                has_conflict=conflict_flag or has_detected_conflict,
                conflict_types=[(conflict_details or {}).get("conflict_type")] if (conflict_details or {}).get("conflict_type") else [],
                conflict_reasons=[(conflict_details or {}).get("conflict_reason")] if (conflict_details or {}).get("conflict_reason") else [],
            )
        )
    supported_required_elements = sum(
        1
        for element_support, element in zip(elements, requirement_elements, strict=False)
        if element.required and element_support.element_status == "supported"
    )
    total_required_elements = len(required_elements)
    conflict_types = _dedupe(
        [
            str(ct or "").strip()
            for element in elements
            if element.conflict_types
            for ct in element.conflict_types
        ]
    )
    conflict_count = sum(1 for element in elements if bool(element.has_conflict))
    conflicting_element_ids = _dedupe(
        [
            str(element.element_id or "").strip()
            for element in elements
            if bool(element.has_conflict) and str(element.element_id or "").strip()
        ]
    )
    conflicting_evidence_ids = _dedupe(
        [
            evidence_id
            for element in elements
            for evidence_id in element.conflicting_evidence_ids
            if str(evidence_id or "").strip()
        ]
    )
    conflicting_quotes = _dedupe(
        [
            quote
            for element in elements
            for quote in element.conflicting_quotes
            if str(quote or "").strip()
        ]
    )
    conflict_reason = next(
        (
            str(element.conflict_reasons[0] if element.conflict_reasons else "").strip()
            for element in elements
            if bool(element.has_conflict) and element.conflict_reasons
        ),
        None,
    )
    return StageElementAssessment(
        stage_key=stage_key,
        supported_required_elements=supported_required_elements,
        total_required_elements=total_required_elements,
        element_coverage_ratio=(
            round(supported_required_elements / total_required_elements, 4)
            if total_required_elements > 0 else 0.0
        ),
        elements=elements,
        quote_mapping_debug=quote_mapping_debug,
        conflict_count=conflict_count,
        has_conflict=bool(conflict_count or conflict_flag),
        conflict_type=conflict_types[0] if conflict_types else None,
        conflict_types=conflict_types,
        conflicting_element_ids=conflicting_element_ids,
        conflicting_evidence_ids=conflicting_evidence_ids,
        conflicting_quotes=conflicting_quotes,
        conflict_reason=conflict_reason,
    )


def _detect_requirement_element_conflicts(
    *,
    element: RequirementElement,
    requirement_elements: Sequence[RequirementElement],
    grounded_record_evidence_items: Sequence[dict[str, str]],
    grounded_record_nodes: Sequence[EvidenceNode],
) -> dict[str, Any] | None:
    element_type = str(element.element_type or "").strip()
    if not element.required:
        return None
    if element_type == "relationship":
        return _detect_relationship_element_conflict(
            element=element,
            grounded_record_evidence_items=grounded_record_evidence_items,
            grounded_record_nodes=grounded_record_nodes,
        )
    if element_type == "residual_risk_acceptability":
        return _detect_residual_risk_element_conflict(
            element=element,
            grounded_record_evidence_items=grounded_record_evidence_items,
        )
    if element_type == "benefit_risk_rationale":
        return _detect_benefit_risk_element_conflict(
            grounded_record_evidence_items=grounded_record_evidence_items,
        )
    if element_type == "per_function":
        return _detect_per_function_element_conflict(
            element=element,
            grounded_record_evidence_items=grounded_record_evidence_items,
            grounded_record_nodes=grounded_record_nodes,
        )
    if element_type == "control_measure":
        return _detect_control_measure_element_conflict(
            grounded_record_evidence_items=grounded_record_evidence_items,
        )
    return None


def _detect_relationship_element_conflict(
    *,
    element: RequirementElement,
    grounded_record_evidence_items: Sequence[dict[str, str]],
    grounded_record_nodes: Sequence[EvidenceNode],
) -> dict[str, Any] | None:
    normalized_element = _normalized_match_key(element.element_text)
    if "required relationship between" not in normalized_element:
        return None
    context_conflict = _detect_synthetic_criticality_summary_mismatch(
        grounded_record_evidence_items=grounded_record_evidence_items,
        grounded_record_nodes=grounded_record_nodes,
    )
    if context_conflict is not None:
        return context_conflict
    relevant_items = [
        item
        for item in grounded_record_evidence_items
        if _relationship_quote_has_risk_components(str(item.get("text") or ""))
    ]
    if not relevant_items:
        return None
    combined_text = " ".join(str(item.get("text") or "") for item in relevant_items)
    criticality = _extract_named_level(combined_text, "criticality")
    complexity = _extract_named_level(combined_text, "complexity")
    overall = _extract_named_level(combined_text, "overall")
    if criticality == "high" and complexity == "high" and overall == "low":
        return {
            "conflict_type": "relationship_value_mismatch",
            "conflict_reason": "overall risk level low conflicts with criticality high and complexity high",
            "items": relevant_items,
        }
    return None


def _detect_residual_risk_element_conflict(
    *,
    element: RequirementElement,
    grounded_record_evidence_items: Sequence[dict[str, str]],
) -> dict[str, Any] | None:
    normalized_element = _normalized_match_key(element.element_text)
    if "acceptability" not in normalized_element and "acceptable" not in normalized_element:
        return None
    conflicting_items = [
        item
        for item in grounded_record_evidence_items
        if _contains_residual_risk_negation(str(item.get("text") or ""))
    ]
    if not conflicting_items:
        return None
    return {
        "conflict_type": "residual_risk_unacceptable",
        "conflict_reason": "explicit residual risk unacceptable language",
        "items": conflicting_items,
    }


def _detect_benefit_risk_element_conflict(
    *,
    grounded_record_evidence_items: Sequence[dict[str, str]],
) -> dict[str, Any] | None:
    conflicting_items = [
        item
        for item in grounded_record_evidence_items
        if _contains_benefit_risk_explicit_absence(str(item.get("text") or ""))
    ]
    if not conflicting_items:
        return None
    return {
        "conflict_type": "benefit_risk_not_performed",
        "conflict_reason": "quote explicitly states no benefit-risk rationale or assessment was performed",
        "items": conflicting_items,
    }


def _detect_per_function_element_conflict(
    *,
    element: RequirementElement,
    grounded_record_evidence_items: Sequence[dict[str, str]],
    grounded_record_nodes: Sequence[EvidenceNode],
) -> dict[str, Any] | None:
    context_conflict = _detect_synthetic_criticality_summary_mismatch(
        grounded_record_evidence_items=grounded_record_evidence_items,
        grounded_record_nodes=grounded_record_nodes,
    )
    if context_conflict is not None and _infer_per_function_attribute_label(element.element_text) == "criticality":
        return {
            **context_conflict,
            "conflict_type": "per_function_attribute_mismatch",
        }
    attribute_label = _infer_per_function_attribute_label(element.element_text)
    conflicting_items = [
        item
        for item in grounded_record_evidence_items
        if _quote_has_concrete_function_reference(str(item.get("text") or ""))
        and _contains_per_function_explicit_conflict(
            quote_text=str(item.get("text") or ""),
            attribute_label=attribute_label,
        )
    ]
    if not conflicting_items:
        return None
    return {
        "conflict_type": "per_function_incompatible_attribute",
        "conflict_reason": f"quote explicitly negates or rejects required per-function {attribute_label}",
        "items": conflicting_items,
    }


def _detect_control_measure_element_conflict(
    *,
    grounded_record_evidence_items: Sequence[dict[str, str]],
) -> dict[str, Any] | None:
    conflicting_items = [
        item
        for item in grounded_record_evidence_items
        if _contains_control_measure_explicit_absence(str(item.get("text") or ""))
    ]
    if not conflicting_items:
        return None
    return {
        "conflict_type": "control_not_implemented",
        "conflict_reason": "quote explicitly states controls were absent or not implemented",
        "items": conflicting_items,
    }


def _map_grounded_quotes_to_requirement_element(
    *,
    element: RequirementElement,
    grounded_record_evidence_items: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    matched_items: list[dict[str, str]] = []
    for item in grounded_record_evidence_items:
        quote_text = str(item.get("text") or "").strip()
        if not quote_text:
            continue
        if _does_grounded_quote_support_element(element=element, quote_text=quote_text):
            matched_items.append(item)
    return matched_items


def _build_evidence_refs(
    *,
    items: Sequence[dict[str, Any]],
    element_id: str,
    conflict_type: str | None = None,
    conflict_reason: str | None = None,
    weak_match_reason: str | None = None,
) -> list[EvidenceRef]:
    refs = []
    seen_keys = set()
    for item in items:
        quote_text = str(item.get("text") or "").strip()
        evidence_id = str(item.get("evidence_id") or "").strip()
        if not quote_text:
            continue
        # Avoid duplicates based on evidence_id + quote + element_id
        key = (evidence_id, quote_text, element_id)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        refs.append(EvidenceRef(
            evidence_id=evidence_id,
            quote=quote_text,
            source_stage=str(item.get("source_stage") or "").strip(),
            section_id=str(item.get("section_id") or "").strip(),
            subsection_id=str(item.get("subsection_id") or "").strip(),
            heading_title=str(item.get("heading_title") or "").strip(),
            source_document=str(item.get("source_document") or "").strip(),
            element_id=element_id,
            conflict_type=conflict_type,
            conflict_reason=conflict_reason,
            weak_match_reason=weak_match_reason,
        ))
    return refs
    linked_pairs: list[tuple[str, str]] = []
    seen_quotes: set[str] = set()
    for item in items:
        quote_text = str(item.get("text") or "").strip()
        if not quote_text or quote_text in seen_quotes:
            continue
        seen_quotes.add(quote_text)
        linked_pairs.append((quote_text, str(item.get("evidence_id") or "").strip()))
    quotes = [quote_text for quote_text, _ in linked_pairs]
    evidence_ids = [evidence_id for _, evidence_id in linked_pairs if evidence_id]
    return quotes, evidence_ids


def _does_grounded_quote_support_element(
    *,
    element: RequirementElement,
    quote_text: str,
) -> bool:
    if str(element.element_type or "").strip() == "per_function":
        return _does_per_function_quote_support_element(
            element_text=element.element_text,
            quote_text=quote_text,
        )

    normalized_element = _normalized_match_key(element.element_text)
    normalized_quote = _normalized_match_key(quote_text)
    if not normalized_element or not normalized_quote:
        return False
    if _is_text_match(normalized_element, normalized_quote):
        return True

    overlap_score = _token_overlap_match_score(element.element_text, quote_text)
    if overlap_score > 0:
        return True

    element_phrases = _extract_key_phrases(element.element_text)
    if element_phrases and any(phrase in normalized_quote for phrase in element_phrases):
        return True

    keyword_hits = _requirement_type_keyword_hits(
        element_type=element.element_type,
        element_text=element.element_text,
        quote_text=quote_text,
    )
    return keyword_hits > 0


def _does_per_function_quote_support_element(
    *,
    element_text: str,
    quote_text: str,
) -> bool:
    normalized_element = _normalized_match_key(element_text)
    normalized_quote = _normalized_match_key(quote_text)
    if not normalized_element or not normalized_quote:
        return False

    attribute_label = _infer_per_function_attribute_label(element_text)
    rows = _extract_table_rows(quote_text)
    if rows:
        if "identified" in normalized_element or "functions/categories are explicitly identified" in normalized_element:
            return any(
                _row_has_function_identifier(cells)
                and not _is_table_header_cells(cells)
                and not _is_table_separator_cells(cells)
                for cells in rows
            )

        if attribute_label == "criticality":
            return any(
                _row_supports_per_function_attribute(
                    cells=cells,
                    column_index=1,
                    attribute_keywords=("criticality",),
                )
                for cells in rows
            )
        if attribute_label == "complexity":
            return any(
                _row_supports_per_function_attribute(
                    cells=cells,
                    column_index=2,
                    attribute_keywords=("complexity",),
                )
                for cells in rows
            )
        if attribute_label == "risk class":
            return any(
                _row_supports_per_function_attribute(
                    cells=cells,
                    column_index=3,
                    attribute_keywords=("risk class", "risk classification", "risk"),
                )
                for cells in rows
            )

        return any(
            _row_has_function_identifier(cells)
            and not _is_table_header_cells(cells)
            and not _is_table_separator_cells(cells)
            and _contains_classification_value(_normalized_match_key(" | ".join(cells)))
            for cells in rows
        )

    return _does_narrative_quote_support_per_function_element(
        element_text=element_text,
        quote_text=quote_text,
    )


def _does_narrative_quote_support_per_function_element(
    *,
    element_text: str,
    quote_text: str,
) -> bool:
    normalized_element = _normalized_match_key(element_text)
    normalized_quote = _normalized_match_key(quote_text)
    if not normalized_element or not normalized_quote:
        return False
    if _is_generic_per_function_narrative(quote_text):
        return False
    if not _quote_has_concrete_function_reference(quote_text):
        return False

    attribute_label = _infer_per_function_attribute_label(element_text)
    if "identified" in normalized_element or "functions/categories are explicitly identified" in normalized_element:
        return True
    if attribute_label == "criticality":
        return _quote_supports_narrative_criticality(quote_text)
    if attribute_label == "complexity":
        return _quote_supports_narrative_complexity(quote_text)
    if attribute_label == "risk class":
        return _quote_supports_narrative_risk_class(quote_text)
    return _contains_classification_value(normalized_quote)


def _infer_per_function_attribute_label(text: str) -> str:
    normalized = _normalized_match_key(text)
    if "criticality" in normalized:
        return "criticality"
    if "complexity" in normalized:
        return "complexity"
    if "risk class" in normalized or "risk classification" in normalized:
        return "risk class"
    return "attribute"


def _extract_table_row_cells(text: str) -> list[str]:
    normalized = " ".join(str(text or "").split()).strip()
    normalized = re.sub(r"^(table row:|table:)\s*", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.strip("“”\"'")
    if "|" not in normalized:
        return []
    return [
        cell.strip()
        for cell in normalized.split("|")
        if cell.strip()
    ]


def _extract_table_rows(text: str) -> list[list[str]]:
    cells = _extract_table_row_cells(text)
    if not cells:
        return []
    row_width = _infer_table_row_width(cells)
    if row_width <= 0 or len(cells) <= row_width:
        return [cells]
    return [
        cells[index:index + row_width]
        for index in range(0, len(cells), row_width)
        if cells[index:index + row_width]
    ]


def _infer_table_row_width(cells: Sequence[str]) -> int:
    normalized_cells = [_normalized_match_key(cell).strip("“”\"'") for cell in cells]
    header_sequence = ("function", "criticality", "complexity", "risk class")
    if len(normalized_cells) >= 4 and tuple(normalized_cells[:4]) == header_sequence:
        return 4
    if len(normalized_cells) >= 4:
        return 4
    return len(normalized_cells)


def _is_table_header_cells(cells: Sequence[str]) -> bool:
    if not cells:
        return False
    normalized_cells = [_normalized_match_key(cell).strip("“”\"'") for cell in cells]
    header_markers = {"function", "criticality", "complexity", "risk class", "risk classification", "risk"}
    return all(cell in header_markers for cell in normalized_cells)


def _is_table_separator_cells(cells: Sequence[str]) -> bool:
    if not cells:
        return False
    return all(re.fullmatch(r"[:\-]+", _normalized_match_key(cell).strip("“”\"'")) for cell in cells)


def _row_has_function_identifier(cells: Sequence[str]) -> bool:
    if len(cells) < 2:
        return False
    first_cell = _normalized_match_key(cells[0]).strip("“”\"'")
    if not first_cell:
        return False
    if first_cell in {"function", "functions"}:
        return False
    return any(char.isalpha() for char in first_cell)


def _row_supports_per_function_attribute(
    *,
    cells: Sequence[str],
    column_index: int,
    attribute_keywords: Sequence[str],
) -> bool:
    if (
        not cells
        or _is_table_header_cells(cells)
        or _is_table_separator_cells(cells)
        or not _row_has_function_identifier(cells)
    ):
        return False
    if len(cells) > column_index and _contains_classification_value(_normalized_match_key(cells[column_index])):
        return True
    normalized_joined = _normalized_match_key(" | ".join(cells))
    return any(keyword in normalized_joined for keyword in attribute_keywords) and _contains_classification_value(normalized_joined)


def _contains_classification_value(text: str) -> bool:
    normalized = _normalized_match_key(text)
    if not normalized:
        return False
    return any(
        re.search(rf"\b{re.escape(value)}\b", normalized)
        for value in ("low", "medium", "high")
    )


def _extract_named_level(text: str, field_name: str) -> str:
    normalized = _normalized_match_key(text)
    if not normalized:
        return ""
    pattern_map = {
        "criticality": r"system criticality level[:\s]+(low|medium|high)",
        "complexity": r"system complexity level[:\s]+(low|medium|high)",
        "overall": r"overall system risk level[:\s]+(low|medium|high)",
    }
    pattern = pattern_map.get(field_name)
    if not pattern:
        return ""
    match = re.search(pattern, normalized)
    return str(match.group(1) if match else "")


def _relationship_quote_has_risk_components(text: str) -> bool:
    normalized = _normalized_match_key(text)
    return any(
        marker in normalized
        for marker in ("system criticality level", "system complexity level", "overall system risk level")
    )


def _contains_residual_risk_negation(text: str) -> bool:
    normalized = _normalized_match_key(text)
    if "residual risk" not in normalized:
        return False
    return any(
        marker in normalized
        for marker in ("not acceptable", "unacceptable", "not deemed acceptable")
    )


def _contains_benefit_risk_explicit_absence(text: str) -> bool:
    normalized = _normalized_match_key(text)
    markers = (
        "no benefit-risk assessment",
        "no benefit-risk rationale",
        "benefit-risk rationale was not performed",
        "benefit-risk assessment was not performed",
    )
    return any(marker in normalized for marker in markers)


def _contains_per_function_explicit_conflict(*, quote_text: str, attribute_label: str) -> bool:
    normalized = _normalized_match_key(quote_text)
    if not normalized or not _quote_has_concrete_function_reference(quote_text):
        return False
    if attribute_label == "criticality":
        markers = (
            "no criticality",
            "criticality not documented",
            "criticality not defined",
            "criticality missing",
        )
    elif attribute_label == "complexity":
        markers = (
            "no complexity",
            "complexity not documented",
            "complexity not defined",
            "complexity missing",
            "likelihood not documented",
        )
    elif attribute_label == "risk class":
        markers = (
            "no risk class",
            "risk class not documented",
            "risk classification not documented",
            "risk class missing",
        )
    else:
        markers = ("not documented", "not defined", "missing")
    return any(marker in normalized for marker in markers)


def _contains_control_measure_explicit_absence(text: str) -> bool:
    normalized = _normalized_match_key(text)
    markers = (
        "no risk controls implemented",
        "controls not implemented",
        "no control measures implemented",
        "risk controls were not implemented",
    )
    return any(marker in normalized for marker in markers)


def _detect_synthetic_criticality_summary_mismatch(
    *,
    grounded_record_evidence_items: Sequence[dict[str, str]],
    grounded_record_nodes: Sequence[EvidenceNode],
) -> dict[str, Any] | None:
    context_text = " ".join(
        str(node.text or "").strip()
        for node in grounded_record_nodes
        if str(node.text or "").strip()
    )
    if not context_text:
        return None
    normalized_context = _normalized_match_key(context_text)
    if "highest observed function criticality in this assessment is medium" not in normalized_context:
        return None

    high_row_quotes = [
        str(item.get("text") or "").strip()
        for item in grounded_record_evidence_items
        if _quote_contains_high_criticality_function(str(item.get("text") or ""))
    ]
    if not high_row_quotes and "system criticality level: high" not in normalized_context:
        return None

    conflict_quotes = _dedupe(
        [
            "The highest observed function criticality in this assessment is Medium.",
            *high_row_quotes,
            *(
                ["System Criticality Level: High"]
                if "system criticality level: high" in normalized_context
                else []
            ),
        ]
    )
    conflicting_evidence_ids = _dedupe(
        [
            str(node.evidence_id or "").strip()
            for node in grounded_record_nodes
            if str(node.evidence_id or "").strip()
        ]
    )
    return {
        "conflict_type": "risk_class_relationship_mismatch",
        "conflict_reason": "summary criticality medium conflicts with grounded high criticality function or overall criticality high",
        "items": [
            {
                "evidence_id": evidence_id,
                "text": quote,
            }
            for evidence_id in (conflicting_evidence_ids or [""])
            for quote in conflict_quotes
        ],
    }


def _quote_contains_high_criticality_function(text: str) -> bool:
    normalized = _normalized_match_key(text)
    if not normalized:
        return False
    if "|" in text:
        rows = _extract_table_rows(text)
        return any(
            _row_supports_per_function_attribute(
                cells=cells,
                column_index=1,
                attribute_keywords=("criticality",),
            ) and len(cells) > 1 and _normalized_match_key(cells[1]) == "high"
            for cells in rows
        )
    return bool(
        _quote_has_concrete_function_reference(text)
        and (
            " critical " in f" {normalized} "
            or re.search(r"\bhigh\s+on\s+[a-z]", normalized)
            or re.search(r"\bis high\b", normalized)
        )
    )


def _quote_has_concrete_function_reference(quote_text: str) -> bool:
    normalized = _normalized_match_key(quote_text)
    if not normalized:
        return False
    patterns = (
        r"\b[a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*){0,2}\s+function\b",
        r"\b(?:high|medium|low)\s+on\s+[a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*){0,2}\b",
        r"\b[a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*){0,2}\s+is\s+(?:critical|complex|high|medium|low)\b",
        r"\b(?:merge|merging|clone|cloning|export|loading|backup|scheduling|report generation|workflow triggering|data export|job scheduling|configuration loading)\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def _is_generic_per_function_narrative(quote_text: str) -> bool:
    normalized = _normalized_match_key(quote_text)
    if not normalized:
        return False
    generic_markers = (
        "major functions have been assessed",
        "functions have been assessed",
        "according to their criticality",
        "according to their complexity",
        "risks posed by",
    )
    return any(marker in normalized for marker in generic_markers) and not _quote_has_concrete_function_reference(quote_text)


def _quote_supports_narrative_criticality(quote_text: str) -> bool:
    normalized = _normalized_match_key(quote_text)
    if not normalized:
        return False
    if re.search(r"\b(?:critical|criticality)\b", normalized):
        return True
    return bool(
        _contains_classification_value(normalized)
        and re.search(r"\b(?:high|medium|low)\s+on\s+[a-z]", normalized)
    )


def _quote_supports_narrative_complexity(quote_text: str) -> bool:
    normalized = _normalized_match_key(quote_text)
    if not normalized:
        return False
    if re.search(r"\b(?:complexity|complex|likelihood)\b", normalized):
        return True
    return bool(
        _contains_classification_value(normalized)
        and re.search(r"\b(?:complexity|likelihood)\b", normalized)
    )


def _quote_supports_narrative_risk_class(quote_text: str) -> bool:
    normalized = _normalized_match_key(quote_text)
    if not normalized:
        return False
    return bool(
        _contains_classification_value(normalized)
        and re.search(r"\b(?:risk class|risk classification|risk level)\b", normalized)
    )


def _extract_key_phrases(text: str) -> list[str]:
    normalized = _normalized_match_key(text)
    if not normalized:
        return []
    phrases: list[str] = []
    parts = re.split(r"[,:;()]", normalized)
    for part in parts:
        candidate = " ".join(part.split()).strip()
        if len(candidate.split()) >= 2:
            phrases.append(candidate)
    return _dedupe(phrases)[:4]


def _requirement_type_keyword_hits(
    *,
    element_type: str,
    element_text: str,
    quote_text: str,
) -> int:
    normalized_quote = _normalized_match_key(quote_text)
    keyword_map: dict[str, list[str]] = {
        "relationship": ["relationship", "between", "linked", "associated"],
        "list_or_table": ["list", "table", "column", "row", "includes", "contains"],
        "per_function": ["function", "functions", "criticality", "complexity", "classification", "risk"],
        "control_measure": ["control", "measure", "mitigate", "reduction", "implemented", "reviewed"],
        "benefit_risk_rationale": ["benefit", "risk", "rationale", "consequence", "justification"],
        "residual_risk_acceptability": ["residual", "risk", "acceptable", "acceptability"],
        "conditional": ["if", "when", "where", "in cases where"],
    }
    keywords = keyword_map.get(str(element_type or "").strip(), [])
    text_tokens = set(_meaningful_tokens(element_text))
    hits = sum(1 for keyword in keywords if keyword in normalized_quote)
    hits += sum(1 for token in text_tokens if token and token in normalized_quote)
    return hits


def calculate_aggregate_metrics(units: Sequence[EvaluationUnit]) -> EvaluationV3Metrics:
    if not units:
        return EvaluationV3Metrics(
            satisfied_count=0,
            partial_count=0,
            not_satisfied_count=0,
            supported_count=0,
            missing_count=0,
            requirements_with_conflict=0,
            total_conflict_findings=0,
            requirements_by_conflict_type={},
            conflict_findings_by_type={},
            avg_grounded_evidence_count=0.0,
            avg_evidence_coverage_ratio=0.0,
        )

    # Calculate conflict metrics
    requirements_with_conflict = sum(
        1
        for unit in units
        if unit.final_element_assessment is not None and bool(unit.final_element_assessment.has_conflict)
    )
    total_conflict_findings = sum(
        int(unit.final_element_assessment.conflict_count or 0)
        for unit in units
        if unit.final_element_assessment is not None
    )
    requirements_by_conflict_type: dict[str, int] = {}
    conflict_findings_by_type: dict[str, int] = {}
    for unit in units:
        if unit.final_element_assessment is None:
            continue
        assessment_requirement_counts, assessment_finding_counts = _summarize_assessment_conflict_types(
            unit.final_element_assessment
        )
        for conflict_type, count in assessment_requirement_counts.items():
            requirements_by_conflict_type[conflict_type] = requirements_by_conflict_type.get(conflict_type, 0) + count
        for conflict_type, count in assessment_finding_counts.items():
            conflict_findings_by_type[conflict_type] = conflict_findings_by_type.get(conflict_type, 0) + count

    return EvaluationV3Metrics(
        satisfied_count=sum(1 for unit in units if unit.final_label == "satisfied"),
        partial_count=sum(1 for unit in units if unit.final_label == "partial"),
        not_satisfied_count=sum(1 for unit in units if unit.final_label == "not_satisfied"),
        supported_count=sum(1 for unit in units if unit.evidence_status == "supported"),
        missing_count=sum(1 for unit in units if unit.evidence_status == "missing"),
        requirements_with_conflict=requirements_with_conflict,
        total_conflict_findings=total_conflict_findings,
        requirements_by_conflict_type=requirements_by_conflict_type,
        conflict_findings_by_type=conflict_findings_by_type,
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
        deliverable_id=unit.deliverable.deliverable_id,
        stage_judgment=unit.stage_1_answer,
        record_nodes=unit.record_evidence_chunks,
    )
    stage_2_grounded_evidence_count = _resolve_stage_grounded_evidence_count(
        deliverable_id=unit.deliverable.deliverable_id,
        stage_judgment=unit.stage_2_answer,
        record_nodes=unit.record_evidence_chunks,
    )
    stage_3_grounded_evidence_count = _resolve_stage_grounded_evidence_count(
        deliverable_id=unit.deliverable.deliverable_id,
        stage_judgment=unit.stage_3_answer,
        record_nodes=unit.record_evidence_chunks,
    )
    stage_1_evidence_pipeline = _build_stage_evidence_pipeline_counters(
        deliverable_id=unit.deliverable.deliverable_id,
        stage_key="stage_1",
        stage_judgment=unit.stage_1_answer,
        record_nodes=unit.record_evidence_chunks,
        requirement_elements=unit.requirement_elements,
        stage_element_assessment=unit.stage_1_element_assessment,
    )
    stage_2_evidence_pipeline = _build_stage_evidence_pipeline_counters(
        deliverable_id=unit.deliverable.deliverable_id,
        stage_key="stage_2",
        stage_judgment=unit.stage_2_answer,
        record_nodes=unit.record_evidence_chunks,
        requirement_elements=unit.requirement_elements,
        stage_element_assessment=unit.stage_2_element_assessment,
    )
    stage_3_evidence_pipeline = _build_stage_evidence_pipeline_counters(
        deliverable_id=unit.deliverable.deliverable_id,
        stage_key="stage_3",
        stage_judgment=unit.stage_3_answer,
        record_nodes=unit.record_evidence_chunks,
        requirement_elements=unit.requirement_elements,
        stage_element_assessment=unit.stage_3_element_assessment,
    )
    elements = unit.final_element_assessment.elements if unit.final_element_assessment else []
    required_element_count = len([e for e in elements if e.required])
    supported_element_count = len([e for e in elements if e.element_status == "supported"])
    missing_element_count = len([e for e in elements if e.element_status == "missing"])
    contradicted_element_count = len([e for e in elements if e.element_status == "contradicted"])
    weak_match_element_count = len([e for e in elements if e.element_status == "weak_match"])
    total_conflict_findings = sum(len(e.conflict_types) for e in elements)
    conflicted_element_ids = [e.element_id for e in elements if e.has_conflict]

    return EvaluationV3ResultRow(
        deliverable_id=unit.deliverable.deliverable_id,
        quote_label=unit.final_label,
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
        stage_1_evidence_pipeline=stage_1_evidence_pipeline,
        stage_2_evidence_pipeline=stage_2_evidence_pipeline,
        stage_3_evidence_pipeline=stage_3_evidence_pipeline,
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
        required_element_count=required_element_count,
        supported_element_count=supported_element_count,
        missing_element_count=missing_element_count,
        contradicted_element_count=contradicted_element_count,
        weak_match_element_count=weak_match_element_count,
        total_conflict_findings=total_conflict_findings,
        conflicted_element_ids=conflicted_element_ids,
        final_element_coverage_ratio=_resolve_element_coverage_ratio(unit.final_element_assessment),
        stage_1_element_coverage_ratio=_resolve_element_coverage_ratio(unit.stage_1_element_assessment),
        stage_2_element_coverage_ratio=_resolve_element_coverage_ratio(unit.stage_2_element_assessment),
        stage_3_element_coverage_ratio=_resolve_element_coverage_ratio(unit.stage_3_element_assessment),
        evidence_coverage_ratio=_resolve_debug_evidence_coverage_ratio(unit),
        requirement_elements=unit.requirement_elements,
        stage_1_element_assessment=unit.stage_1_element_assessment,
        stage_2_element_assessment=unit.stage_2_element_assessment,
        stage_3_element_assessment=unit.stage_3_element_assessment,
        final_element_assessment=unit.final_element_assessment,
        conflict_count=_resolve_element_conflict_count(unit.final_element_assessment),
        conflict_type=_resolve_element_conflict_type(unit.final_element_assessment),
        conflict_types=_resolve_element_conflict_types(unit.final_element_assessment),
        conflicting_element_ids=_resolve_element_conflicting_element_ids(unit.final_element_assessment),
        conflicting_evidence_ids=_resolve_element_conflicting_evidence_ids(unit.final_element_assessment),
        conflicting_quotes=_resolve_element_conflicting_quotes(unit.final_element_assessment),
        conflict_reason=_resolve_element_conflict_reason(unit.final_element_assessment),
        has_conflict=_resolve_debug_has_conflict(unit),
        contradiction_type=unit.contradiction_type,
        evidence_audit_status=_resolve_evidence_audit_status(
            has_conflict=_resolve_debug_has_conflict(unit),
            final_element_coverage_ratio=_resolve_element_coverage_ratio(unit.final_element_assessment),
            grounded_evidence_count=_resolve_debug_grounded_evidence_count(unit),
        ),
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
            "requirements_with_conflict": 0,
            "total_conflict_findings": 0,
            "total_required_elements": 0,
            "total_supported_elements": 0,
            "total_missing_elements": 0,
            "total_contradicted_elements": 0,
            "total_weak_match_elements": 0,
            "requirements_by_conflict_type": {},
            "conflict_findings_by_type": {},
            "avg_grounded_evidence": 0.0,
            "avg_evidence_coverage": 0.0,
        }

    requirements_by_conflict_type: dict[str, int] = {}
    conflict_findings_by_type: dict[str, int] = {}
    for row in rows:
        assessment_requirement_counts, assessment_finding_counts = _summarize_assessment_conflict_types(
            row.final_element_assessment
        )
        for conflict_type, count in assessment_requirement_counts.items():
            requirements_by_conflict_type[conflict_type] = requirements_by_conflict_type.get(conflict_type, 0) + count
        for conflict_type, count in assessment_finding_counts.items():
            conflict_findings_by_type[conflict_type] = conflict_findings_by_type.get(conflict_type, 0) + count

    return {
        "total_units": total_units,
        "satisfied": sum(1 for row in rows if row.quote_label == "satisfied"),
        "partial": sum(1 for row in rows if row.quote_label == "partial"),
        "not_satisfied": sum(1 for row in rows if row.quote_label == "not_satisfied"),
        "supported": sum(1 for row in rows if row.evidence_status == "supported"),
        "missing": sum(1 for row in rows if row.evidence_status == "missing"),
        "requirements_with_conflict": sum(1 for row in rows if row.has_conflict),
        "total_conflict_findings": sum(row.total_conflict_findings for row in rows),
        "total_required_elements": sum(row.required_element_count for row in rows),
        "total_supported_elements": sum(row.supported_element_count for row in rows),
        "total_missing_elements": sum(row.missing_element_count for row in rows),
        "total_contradicted_elements": sum(row.contradicted_element_count for row in rows),
        "total_weak_match_elements": sum(row.weak_match_element_count for row in rows),
        "requirements_by_conflict_type": requirements_by_conflict_type,
        "conflict_findings_by_type": conflict_findings_by_type,
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
            "quote_label": unit.final_label,
            "evidence_audit_status": _resolve_evidence_audit_status(
                has_conflict=_resolve_debug_has_conflict(unit),
                final_element_coverage_ratio=_resolve_element_coverage_ratio(unit.final_element_assessment),
                grounded_evidence_count=_resolve_debug_grounded_evidence_count(unit),
            ),
            "evidence_status": unit.evidence_status,
            "required_evidence_count": unit.required_evidence_count,
            "grounded_evidence_count": _resolve_debug_grounded_evidence_count(unit),
            "evidence_coverage_ratio": _resolve_debug_evidence_coverage_ratio(unit),
            "required_element_count": _resolve_required_element_count(unit),
            "supported_element_count": _resolve_supported_required_element_count(unit),
            "final_element_coverage_ratio": _resolve_element_coverage_ratio(unit.final_element_assessment),
            "stage_1_element_coverage_ratio": _resolve_element_coverage_ratio(unit.stage_1_element_assessment),
            "stage_2_element_coverage_ratio": _resolve_element_coverage_ratio(unit.stage_2_element_assessment),
            "stage_3_element_coverage_ratio": _resolve_element_coverage_ratio(unit.stage_3_element_assessment),
            "grounded_chunk_count": _resolve_debug_grounded_chunk_count(unit),
            "grounded_subsection_count": len(_resolve_debug_grounded_subsection_ids(unit)),
            "has_conflict": _resolve_debug_has_conflict(unit),
            "total_conflict_findings": _resolve_element_conflict_count(unit.final_element_assessment),
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
            "stage_1_evidence_pipeline": _build_stage_evidence_pipeline_counters(
                deliverable_id=unit.deliverable.deliverable_id,
                stage_key="stage_1",
                stage_judgment=unit.stage_1_answer,
                record_nodes=unit.record_evidence_chunks,
                requirement_elements=unit.requirement_elements,
                stage_element_assessment=unit.stage_1_element_assessment,
            ).model_dump(),
            "stage_2_evidence_pipeline": _build_stage_evidence_pipeline_counters(
                deliverable_id=unit.deliverable.deliverable_id,
                stage_key="stage_2",
                stage_judgment=unit.stage_2_answer,
                record_nodes=unit.record_evidence_chunks,
                requirement_elements=unit.requirement_elements,
                stage_element_assessment=unit.stage_2_element_assessment,
            ).model_dump(),
            "stage_3_evidence_pipeline": _build_stage_evidence_pipeline_counters(
                deliverable_id=unit.deliverable.deliverable_id,
                stage_key="stage_3",
                stage_judgment=unit.stage_3_answer,
                record_nodes=unit.record_evidence_chunks,
                requirement_elements=unit.requirement_elements,
                stage_element_assessment=unit.stage_3_element_assessment,
            ).model_dump(),
            "requirement_elements": [element.model_dump(exclude_none=True) for element in unit.requirement_elements],
            "stage_1_element_assessment": (
                unit.stage_1_element_assessment.model_dump(exclude_none=True)
                if unit.stage_1_element_assessment is not None else None
            ),
            "stage_2_element_assessment": (
                unit.stage_2_element_assessment.model_dump(exclude_none=True)
                if unit.stage_2_element_assessment is not None else None
            ),
            "stage_3_element_assessment": (
                unit.stage_3_element_assessment.model_dump(exclude_none=True)
                if unit.stage_3_element_assessment is not None else None
            ),
            "final_element_assessment": (
                unit.final_element_assessment.model_dump(exclude_none=True)
                if unit.final_element_assessment is not None else None
            ),
            "conflict_count": _resolve_element_conflict_count(unit.final_element_assessment),
            "conflict_type": _resolve_element_conflict_type(unit.final_element_assessment),
            "conflict_types": _resolve_element_conflict_types(unit.final_element_assessment),
            "conflicting_element_ids": _resolve_element_conflicting_element_ids(unit.final_element_assessment),
            "conflicting_evidence_ids": _resolve_element_conflicting_evidence_ids(unit.final_element_assessment),
            "conflicting_quotes": _resolve_element_conflicting_quotes(unit.final_element_assessment),
            "conflict_reason": _resolve_element_conflict_reason(unit.final_element_assessment),
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
    requirements_by_conflict_type: dict[str, int] = {}
    conflict_findings_by_type: dict[str, int] = {}
    for unit in units:
        assessment_requirement_counts, assessment_finding_counts = _summarize_assessment_conflict_types(
            unit.final_element_assessment
        )
        for conflict_type, count in assessment_requirement_counts.items():
            requirements_by_conflict_type[conflict_type] = requirements_by_conflict_type.get(conflict_type, 0) + count
        for conflict_type, count in assessment_finding_counts.items():
            conflict_findings_by_type[conflict_type] = conflict_findings_by_type.get(conflict_type, 0) + count
    return {
        "total_units": len(units),
        "quote_label_counts": {
            "satisfied": sum(1 for row in rows if row.get("quote_label") == "satisfied"),
            "partial": sum(1 for row in rows if row.get("quote_label") == "partial"),
            "not_satisfied": sum(1 for row in rows if row.get("quote_label") == "not_satisfied"),
        },
        "requirements_with_conflict": sum(1 for row in rows if bool(row.get("has_conflict"))),
        "total_conflict_findings": sum(int(row.get("total_conflict_findings") or 0) for row in rows),
        "requirements_by_conflict_type": requirements_by_conflict_type,
        "conflict_findings_by_type": conflict_findings_by_type,
        "evidence_audit_status_counts": {
            "supported": sum(1 for row in rows if row.get("evidence_audit_status") == "supported"),
            "partial": sum(1 for row in rows if row.get("evidence_audit_status") == "partial"),
            "weak_match": sum(1 for row in rows if row.get("evidence_audit_status") == "weak_match"),
            "missing": sum(1 for row in rows if row.get("evidence_audit_status") == "missing"),
            "conflict": sum(1 for row in rows if row.get("evidence_audit_status") == "conflict"),
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
        "satisfied": sum(1 for row in rows if row.get("quote_label") == "satisfied"),
        "partial": sum(1 for row in rows if row.get("quote_label") == "partial"),
        "not_satisfied": sum(1 for row in rows if row.get("quote_label") == "not_satisfied"),
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
        "quote_label",
        "evidence_audit_status",
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
        "quote_label",
        "evidence_audit_status",
        "evidence_status",
        "contradiction_type",
        "evidence_score",
        "record_evidence_section_count",
        "reference_evidence_section_count",
        "stage_1_label",
        "stage_2_label",
        "stage_3_label",
        "stage_1_evidence_pipeline",
        "stage_2_evidence_pipeline",
        "stage_3_evidence_pipeline",
        "requirement_elements",
        "stage_1_element_assessment",
        "stage_2_element_assessment",
        "stage_3_element_assessment",
        "final_element_assessment",
        "conflict_count",
        "conflict_type",
        "conflict_types",
        "conflicting_element_ids",
        "conflicting_evidence_ids",
        "conflicting_quotes",
        "conflict_reason",
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
    deliverable_id: str,
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
        requirements_with_conflict=1 if any(stage.conflict_flag for stage in stage_judgments) else 0,
        total_conflict_findings=0,  # Not available at unit level
        requirements_by_conflict_type={},  # Not available at unit level
        conflict_findings_by_type={},  # Not available at unit level
        avg_grounded_evidence_count=float(
            _count_grounded_record_evidence(
                deliverable_id=deliverable_id,
                stage_judgments=stage_judgments,
                record_nodes=record_nodes,
            )
        ),
        avg_evidence_coverage_ratio=_compute_evidence_coverage_ratio(
            grounded_evidence_count=_count_grounded_record_evidence(
                deliverable_id=deliverable_id,
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
        deliverable_id=unit.deliverable.deliverable_id,
        stage_judgments=(unit.stage_1_answer, unit.stage_2_answer, unit.stage_3_answer),
        record_nodes=unit.record_evidence_chunks,
    )


def _resolve_required_element_count(unit: EvaluationUnit) -> int:
    return sum(1 for element in unit.requirement_elements if element.required)


def _resolve_supported_required_element_count(unit: EvaluationUnit) -> int:
    if unit.final_element_assessment is None:
        return 0
    return int(unit.final_element_assessment.supported_required_elements or 0)


def _resolve_element_coverage_ratio(assessment: StageElementAssessment | None) -> float:
    if assessment is None:
        return 0.0
    return float(assessment.element_coverage_ratio or 0.0)


def _resolve_stage_grounded_evidence_count(
    *,
    deliverable_id: str,
    stage_judgment: StageJudgment,
    record_nodes: Sequence[EvidenceNode],
) -> int:
    return len(
        _resolve_stage_grounded_record_evidence_items(
            deliverable_id=deliverable_id,
            stage_judgment=stage_judgment,
            record_nodes=record_nodes,
        )
    )


def _build_stage_evidence_pipeline_counters(
    *,
    deliverable_id: str,
    stage_key: str,
    stage_judgment: StageJudgment,
    record_nodes: Sequence[EvidenceNode],
    requirement_elements: Sequence[RequirementElement],
    stage_element_assessment: StageElementAssessment | None,
) -> EvidencePipelineCounters:
    grounded_items = _resolve_stage_grounded_record_evidence_items(
        deliverable_id=deliverable_id,
        stage_judgment=stage_judgment,
        record_nodes=record_nodes,
    )
    required_elements = [element for element in requirement_elements if element.required]
    element_supported_quote_count = _count_element_supported_quotes(
        grounded_record_evidence_items=grounded_items,
        requirement_elements=required_elements,
    )
    required_element_count = len(required_elements)
    element_coverage_ratio = (
        _resolve_element_coverage_ratio(stage_element_assessment)
        if stage_element_assessment is not None
        else 0.0
    )
    return EvidencePipelineCounters(
        retrieved_candidate_count=_resolve_stage_retrieved_candidate_count(
            stage_key=stage_key,
            record_nodes=record_nodes,
        ),
        accepted_quote_count=_resolve_stage_accepted_quote_count(stage_judgment=stage_judgment),
        grounded_quote_count=len(grounded_items),
        element_supported_quote_count=element_supported_quote_count,
        required_element_count=required_element_count,
        element_coverage_ratio=element_coverage_ratio,
        quote_element_disagreement=bool(grounded_items) and element_supported_quote_count == 0,
    )


def _resolve_stage_retrieved_candidate_count(
    *,
    stage_key: str,
    record_nodes: Sequence[EvidenceNode],
) -> int:
    if stage_key == "stage_1":
        return 0
    return len(record_nodes)


def _resolve_stage_accepted_quote_count(*, stage_judgment: StageJudgment) -> int:
    if stage_judgment.supporting_record_evidence_items:
        return len(stage_judgment.supporting_record_evidence_items)
    return len([evidence_id for evidence_id in stage_judgment.supporting_record_evidence_ids if evidence_id])


def _count_element_supported_quotes(
    *,
    grounded_record_evidence_items: Sequence[dict[str, str]],
    requirement_elements: Sequence[RequirementElement],
) -> int:
    count = 0
    for item in grounded_record_evidence_items:
        quote_text = str(item.get("text") or "").strip()
        if not quote_text:
            continue
        if any(
            _does_grounded_quote_support_element(element=element, quote_text=quote_text)
            for element in requirement_elements
        ):
            count += 1
    return count


def _build_quote_element_mapping_debug(
    *,
    requirement_elements: Sequence[RequirementElement],
    grounded_record_evidence_items: Sequence[dict[str, str]],
) -> list[QuoteElementMappingDebug]:
    return [
        _build_single_quote_element_mapping_debug(
            quote_text=str(item.get("text") or "").strip(),
            requirement_elements=requirement_elements,
        )
        for item in grounded_record_evidence_items
        if str(item.get("text") or "").strip()
    ]


def _build_single_quote_element_mapping_debug(
    *,
    quote_text: str,
    requirement_elements: Sequence[RequirementElement],
) -> QuoteElementMappingDebug:
    matched_element_ids = [
        element.element_id
        for element in requirement_elements
        if _does_grounded_quote_support_element(element=element, quote_text=quote_text)
    ]
    reason = (
        _explain_quote_element_match(
            quote_text=quote_text,
            requirement_elements=requirement_elements,
        )
        if matched_element_ids
        else _explain_quote_element_non_match(
            quote_text=quote_text,
            requirement_elements=requirement_elements,
        )
    )
    return QuoteElementMappingDebug(
        quote_text=quote_text,
        matched_element_ids=matched_element_ids,
        reason=reason,
    )


def _explain_quote_element_match(
    *,
    quote_text: str,
    requirement_elements: Sequence[RequirementElement],
) -> str:
    if any(str(element.element_type or "").strip() == "per_function" for element in requirement_elements):
        if _extract_table_rows(quote_text):
            return "matched_table_row"
        return "matched_narrative_function_attribute"
    return "matched"


def _explain_quote_element_non_match(
    *,
    quote_text: str,
    requirement_elements: Sequence[RequirementElement],
) -> str:
    if any(str(element.element_type or "").strip() == "per_function" for element in requirement_elements):
        return _explain_per_function_quote_non_match(
            quote_text=quote_text,
            requirement_elements=requirement_elements,
        )
    return "no_element_match"


def _explain_per_function_quote_non_match(
    *,
    quote_text: str,
    requirement_elements: Sequence[RequirementElement],
) -> str:
    rows = _extract_table_rows(quote_text)
    if not rows:
        if _is_generic_per_function_narrative(quote_text):
            return "generic_not_per_function_specific"
        return "not_table_or_function_content"
    data_rows = [
        cells
        for cells in rows
        if cells and not _is_table_header_cells(cells) and not _is_table_separator_cells(cells)
    ]
    if not data_rows:
        if any(_is_table_header_cells(cells) for cells in rows):
            return "header_only"
        return "no_data_rows"
    if not any(_row_has_function_identifier(cells) for cells in data_rows):
        return "no_function_row"
    attribute_labels = {
        _infer_per_function_attribute_label(element.element_text)
        for element in requirement_elements
    }
    if "criticality" in attribute_labels and not any(
        _row_supports_per_function_attribute(cells=cells, column_index=1, attribute_keywords=("criticality",))
        for cells in data_rows
    ):
        return "missing_criticality_value"
    if "complexity" in attribute_labels and not any(
        _row_supports_per_function_attribute(cells=cells, column_index=2, attribute_keywords=("complexity",))
        for cells in data_rows
    ):
        return "missing_complexity_value"
    if "risk class" in attribute_labels and not any(
        _row_supports_per_function_attribute(
            cells=cells,
            column_index=3,
            attribute_keywords=("risk class", "risk classification", "risk"),
        )
        for cells in data_rows
    ):
        return "missing_risk_class_value"
    return "no_per_function_element_match"


def _resolve_debug_grounded_chunk_count(unit: EvaluationUnit) -> int:
    grounded_nodes = _resolve_grounded_record_nodes(
        stage_judgments=(unit.stage_1_answer, unit.stage_2_answer, unit.stage_3_answer),
        record_nodes=unit.record_evidence_chunks,
    )
    return len(grounded_nodes)


def _resolve_debug_has_conflict(unit: EvaluationUnit) -> bool:
    if unit.final_element_assessment is not None and bool(unit.final_element_assessment.has_conflict):
        return True
    if unit.evidence_status == "conflicting":
        return True
    return _contradiction_type_implies_conflict(unit.contradiction_type)


def _resolve_evidence_audit_status(
    *,
    has_conflict: bool,
    final_element_coverage_ratio: float,
    grounded_evidence_count: int,
) -> str:
    if has_conflict:
        return "conflict"
    if final_element_coverage_ratio >= 1.0:
        return "supported"
    if final_element_coverage_ratio > 0:
        return "partial"
    if grounded_evidence_count > 0:
        return "weak_match"
    return "missing"


def _summarize_assessment_conflict_types(
    assessment: StageElementAssessment | None,
) -> tuple[dict[str, int], dict[str, int]]:
    if assessment is None:
        return {}, {}
    requirements_by_conflict_type: dict[str, int] = {}
    conflict_findings_by_type: dict[str, int] = {}
    assessment_conflict_types: set[str] = set()
    for element in assessment.elements:
        element_conflict_types = [
            str(conflict_type or "").strip()
            for conflict_type in element.conflict_types
            if str(conflict_type or "").strip()
        ]
        if not element_conflict_types:
            continue
        assessment_conflict_types.update(element_conflict_types)
        for conflict_type in element_conflict_types:
            conflict_findings_by_type[conflict_type] = conflict_findings_by_type.get(conflict_type, 0) + 1
    for conflict_type in assessment_conflict_types:
        requirements_by_conflict_type[conflict_type] = requirements_by_conflict_type.get(conflict_type, 0) + 1
    return requirements_by_conflict_type, conflict_findings_by_type


def _resolve_element_conflict_count(assessment: StageElementAssessment | None) -> int:
    if assessment is None:
        return 0
    return int(assessment.conflict_count or 0)


def _resolve_element_conflict_type(assessment: StageElementAssessment | None) -> str | None:
    if assessment is None:
        return None
    value = str(assessment.conflict_type or "").strip()
    return value or None


def _resolve_element_conflict_types(assessment: StageElementAssessment | None) -> list[str]:
    if assessment is None:
        return []
    return [str(item).strip() for item in assessment.conflict_types if str(item).strip()]


def _resolve_element_conflicting_element_ids(assessment: StageElementAssessment | None) -> list[str]:
    if assessment is None:
        return []
    return [str(item).strip() for item in assessment.conflicting_element_ids if str(item).strip()]


def _resolve_element_conflicting_evidence_ids(assessment: StageElementAssessment | None) -> list[str]:
    if assessment is None:
        return []
    return [str(item).strip() for item in assessment.conflicting_evidence_ids if str(item).strip()]


def _resolve_element_conflicting_quotes(assessment: StageElementAssessment | None) -> list[str]:
    if assessment is None:
        return []
    return [str(item).strip() for item in assessment.conflicting_quotes if str(item).strip()]


def _resolve_element_conflict_reason(assessment: StageElementAssessment | None) -> str | None:
    if assessment is None:
        return None
    value = str(assessment.conflict_reason or "").strip()
    return value or None


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
    final_label = row.get("quote_label")
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
    deliverable_id: str,
    stage_judgments: Sequence[StageJudgment],
    record_nodes: Sequence[EvidenceNode],
) -> int:
    return len(
        _resolve_grounded_record_evidence_items(
            deliverable_id=deliverable_id,
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
    deliverable_id: str,
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
                "source_stage": selected_stage.stage_key,
                "text": node.text,
            }
            for node in record_nodes
            if node.evidence_id and node.evidence_id in accepted_record_id_set
        ]
    return _normalize_grounded_record_evidence_items(
        deliverable_id=deliverable_id,
        stage_key=selected_stage.stage_key,
        grounded_items=accepted_items,
        record_nodes=record_nodes,
    )


def _resolve_grounded_record_evidence_ids(
    *,
    stage_judgments: Sequence[StageJudgment],
) -> list[str]:
    stage_lookup = {stage.stage_key: stage for stage in stage_judgments}
    stage_3 = stage_lookup.get("stage_3")
    stage_2 = stage_lookup.get("stage_2")
    selected_stage = (
        stage_3
        if stage_3 and stage_3.supporting_record_evidence_ids
        else stage_2
        if stage_2 and stage_2.supporting_record_evidence_ids
        else None
    )
    if selected_stage is None:
        return []
    return [evidence_id for evidence_id in selected_stage.supporting_record_evidence_ids if evidence_id]


def _resolve_stage_grounded_record_nodes(
    *,
    stage_judgment: StageJudgment,
    record_nodes: Sequence[EvidenceNode],
) -> list[EvidenceNode]:
    matched_ids: list[str] = [
        str(evidence_id or "").strip()
        for evidence_id in stage_judgment.supporting_record_evidence_ids
        if str(evidence_id or "").strip()
    ]
    if not matched_ids and stage_judgment.supporting_record_evidence_items:
        for item in stage_judgment.supporting_record_evidence_items:
            matched_ids.extend(_match_record_item_to_nodes(item, record_nodes))
    matched_id_set = {evidence_id for evidence_id in matched_ids if evidence_id}
    if not matched_id_set:
        return []
    return [
        node
        for node in record_nodes
        if node.evidence_id and node.evidence_id in matched_id_set
    ]


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
    deliverable_id: str,
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
                "source_stage": stage_judgment.stage_key,
                "text": node.text,
            }
            for node in record_nodes
            if node.evidence_id and node.evidence_id in accepted_record_id_set
        ]
    return _normalize_grounded_record_evidence_items(
        deliverable_id=deliverable_id,
        stage_key=stage_judgment.stage_key,
        grounded_items=accepted_items,
        record_nodes=record_nodes,
    )


def _normalize_grounded_record_evidence_items(
    *,
    deliverable_id: str,
    stage_key: str,
    grounded_items: Sequence[dict[str, str]],
    record_nodes: Sequence[EvidenceNode],
) -> list[dict[str, str]]:
    normalized_items: list[dict[str, str]] = []
    for item in grounded_items:
        if not _match_record_item_to_nodes(item, record_nodes):
            continue
        quote_text = str(item.get("text") or "").strip()
        if not quote_text:
            continue
        normalized_items.append(
            {
                "evidence_id": _resolve_grounded_record_item_evidence_id(
                    item=item,
                    record_nodes=record_nodes,
                    deliverable_id=deliverable_id,
                    stage_key=stage_key,
                ),
                "section_id": str(item.get("section_id") or "").strip(),
                "subsection_id": str(item.get("subsection_id") or "").strip(),
                "section_label": str(item.get("section_label") or "").strip(),
                "heading_title": str(item.get("heading_title") or "").strip(),
                "source_document": str(item.get("source_document") or "").strip(),
                "source_stage": str(item.get("source_stage") or "").strip() or stage_key,
                "text": quote_text,
            }
        )
    return normalized_items


def _resolve_grounded_record_item_evidence_id(
    *,
    item: dict[str, Any],
    record_nodes: Sequence[EvidenceNode],
    deliverable_id: str,
    stage_key: str,
) -> str:
    explicit_id = str(item.get("evidence_id") or "").strip()
    if explicit_id:
        return explicit_id
    matched_ids = _match_record_item_to_nodes(item, record_nodes)
    if matched_ids:
        return matched_ids[0]
    return _build_grounded_quote_fallback_evidence_id(
        deliverable_id=deliverable_id,
        stage_key=stage_key,
        quote_text=str(item.get("text") or ""),
    )


def _build_grounded_quote_fallback_evidence_id(
    *,
    deliverable_id: str,
    stage_key: str,
    quote_text: str,
) -> str:
    normalized_quote = _normalized_match_key(quote_text)
    quote_hash = hashlib.sha1(normalized_quote.encode("utf-8")).hexdigest()[:12]
    return f"{deliverable_id}:{stage_key}:{quote_hash}"


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
