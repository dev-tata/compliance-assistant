from __future__ import annotations

import re

from app.schemas.compliance import (
    ComplianceAnalysis,
    ComplianceEvidenceItem,
    ComplianceFinding,
)


def _valid_evidence_items(finding: ComplianceFinding) -> list[ComplianceEvidenceItem]:
    return [
        item for item in finding.evidence_items
        if item.source_document and item.is_valid_for_requirement
    ]
def enrich_analysis_for_scoring(
    analysis: ComplianceAnalysis,
    *,
    requirement_weights: list[float] | None = None,
    deliverable_metadata: list[dict[str, object]] | None = None,
) -> ComplianceAnalysis:
    enriched_findings = [
        _enrich_finding(
            finding,
            weight=requirement_weights[index] if requirement_weights and index < len(requirement_weights) else None,
            deliverable_meta=deliverable_metadata[index] if deliverable_metadata and index < len(deliverable_metadata) else None,
        )
        for index, finding in enumerate(analysis.findings)
    ]
    return analysis.model_copy(
        update={
            "findings": enriched_findings,
            "procedure_to_record": enriched_findings,
        }
    )


def _enrich_finding(
    finding: ComplianceFinding,
    *,
    weight: float | None = None,
    deliverable_meta: dict[str, object] | None = None,
) -> ComplianceFinding:
    effective_weight = weight if weight is not None else _compute_weight(finding.requirement)
    expected_evidence_breadth = _coerce_expected_evidence_breadth(deliverable_meta)
    material_element_count = _estimate_material_element_count(finding.requirement)
    adjusted_status = _apply_structural_breadth_threshold(
        status=finding.status,
        evidence_breadth=finding.evidence_breadth,
        expected_evidence_breadth=expected_evidence_breadth,
    )
    normalized_finding = finding.model_copy(
        update={
            "status": adjusted_status,
            "expected_evidence_breadth": expected_evidence_breadth,
        }
    )
    evidence_strength = _compute_evidence_strength(
        normalized_finding,
        material_element_count=material_element_count,
    )
    requirement_coverage_percent = _compute_requirement_coverage_percent(
        finding=normalized_finding,
        material_element_count=material_element_count,
    )
    final_status = _derive_status_from_requirement_coverage_percent(
        requirement_coverage_percent=requirement_coverage_percent,
    )
    return normalized_finding.model_copy(
        update={
            "status": final_status,
            "evidence_strength": evidence_strength,
            "weight": effective_weight,
            "material_element_count": material_element_count,
            "requirement_coverage_percent": requirement_coverage_percent,
        }
    )


def _coerce_expected_evidence_breadth(deliverable_meta: dict[str, object] | None) -> int:
    if not deliverable_meta:
        return 1
    raw = deliverable_meta.get("expected_evidence_breadth")
    if isinstance(raw, (int, float)) and raw >= 1:
        return int(raw)
    return 1


def _apply_structural_breadth_threshold(
    *,
    status: str,
    evidence_breadth: int,
    expected_evidence_breadth: int,
) -> str:
    if status != "satisfied":
        return status
    if evidence_breadth <= 0:
        return status
    if expected_evidence_breadth <= 1:
        return status
    return status if evidence_breadth >= expected_evidence_breadth else "partial"


def _compute_evidence_strength(
    finding: ComplianceFinding,
    *,
    material_element_count: int,
) -> float:
    valid_items = _valid_evidence_items(finding)
    evidence_count = len(valid_items)
    grounded_evidence_count = evidence_count if finding.evidence_items else (
        min(max(len(finding.evidence), 0), 1) if finding.evidence and finding.source_document else 0
    )
    if grounded_evidence_count <= 0:
        return 0.0

    expected_breadth = max(1, finding.expected_evidence_breadth)
    observed_breadth = (
        min(finding.evidence_breadth, expected_breadth)
        if finding.evidence_breadth > 0
        else min(grounded_evidence_count, expected_breadth)
    )
    breadth_ratio = min(1.0, observed_breadth / expected_breadth)
    density_ratio = min(1.0, grounded_evidence_count / max(1, material_element_count))

    return round((0.7 * breadth_ratio) + (0.3 * density_ratio), 4)

def _compute_weight(requirement: str) -> float:
    text = requirement.lower()

    if "risk control" in text or "residual risk" in text:
        return 1.5
    if "criticality" in text or "complexity" in text or "overall risk classification" in text:
        return 1.2
    if "major system functions" in text or "risk identification" in text:
        return 1.1
    if "scope" in text or "responsibilit" in text or "version history" in text:
        return 0.8

    return 1.0


def _estimate_material_element_count(requirement: str) -> int:
    text = " ".join(str(requirement or "").split())
    if not text:
        return 1
    segments = [
        segment.strip(" ,.:")
        for segment in re.split(r";|\n+|\b(?:as well as|together with|including)\b", text, flags=re.IGNORECASE)
        if segment.strip(" ,.:")
    ]
    if len(segments) == 1 and len(text) >= 100:
        and_parts = [
            part.strip(" ,.:")
            for part in re.split(r"\band\b", text, flags=re.IGNORECASE)
            if part.strip(" ,.:")
        ]
        if 1 < len(and_parts) <= 3 and all(len(part) >= 12 for part in and_parts):
            segments = and_parts
    return max(1, min(len(segments), 5))


def _compute_requirement_coverage_percent(
    *,
    finding: ComplianceFinding,
    material_element_count: int,
) -> int:
    valid_items = _valid_evidence_items(finding)
    evidence_count = len(valid_items)
    grounded_count = evidence_count if finding.evidence_items else (
        min(max(len(finding.evidence), 0), 1) if finding.evidence and finding.source_document else 0
    )
    if grounded_count <= 0:
        return 0
    element_ratio = min(1.0, grounded_count / max(1, material_element_count))
    breadth_ratio = min(
        1.0,
        (
            min(finding.evidence_breadth, finding.expected_evidence_breadth)
            if finding.evidence_breadth > 0
            else min(grounded_count, finding.expected_evidence_breadth)
        ) / max(1, finding.expected_evidence_breadth),
    )
    return round(100 * min(element_ratio, breadth_ratio))


def _derive_status_from_requirement_coverage_percent(
    *,
    requirement_coverage_percent: int,
) -> str:
    if requirement_coverage_percent >= 80:
        return "satisfied"
    if requirement_coverage_percent >= 20:
        return "partial"
    return "not_satisfied"
