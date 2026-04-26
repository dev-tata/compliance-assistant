from __future__ import annotations

from app.schemas.compliance import (
    ComplianceAnalysis,
    ComplianceFinding,
    ComplianceScores,
)


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def enrich_analysis_for_scoring(
    analysis: ComplianceAnalysis,
    *,
    requirement_weights: list[float] | None = None,
) -> ComplianceAnalysis:
    enriched_findings = [
        _enrich_finding(
            finding,
            weight=requirement_weights[index] if requirement_weights and index < len(requirement_weights) else None,
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
) -> ComplianceFinding:
    evidence_strength = _compute_evidence_strength(finding)
    confidence = _compute_confidence(finding.status, evidence_strength)
    effective_weight = weight if weight is not None else _compute_weight(finding.requirement)
    return finding.model_copy(
        update={
            "evidence_strength": evidence_strength,
            "confidence": confidence,
            "weight": effective_weight,
        }
    )
def _compute_evidence_strength(finding: ComplianceFinding) -> float:
    evidence_count = len(finding.evidence)
    source_count = len(set(finding.source_documents))

    if evidence_count == 0:
        return 0.0

    base = 0.35 + (0.2 * min(evidence_count, 3)) + (0.1 * min(source_count, 2))

    if finding.status == "partial":
        base *= 0.8
    elif finding.status == "not_satisfied":
        base *= 0.6

    return round(min(1.0, base), 4)


def _compute_confidence(status: str, evidence_strength: float) -> float:
    if status == "satisfied":
        confidence = 0.55 + (0.45 * evidence_strength)
    elif status == "partial":
        confidence = 0.45 + (0.35 * evidence_strength)
    else:
        confidence = 0.35 + (0.4 * evidence_strength)

    return round(min(1.0, max(0.0, confidence)), 4)
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


def compute_scores(
    analysis: ComplianceAnalysis,
    *,
    weighted_m2: bool = False,
) -> ComplianceScores:
    findings = analysis.findings
    if not findings:
        return ComplianceScores(
            m2_ordinal_score=0.0,
            m3_evidence_weighted_score=0.0,
            m5_grounding_score=0.0,
        )

    ordinal_map = {
        "satisfied": 1.0,
        "partial": 0.5,
        "not_satisfied": 0.0,
    }

    total_weight = sum(f.weight for f in findings)

    if weighted_m2:
        m2 = _safe_div(
            sum(ordinal_map.get(f.status, 0.0) * f.weight for f in findings),
            total_weight,
        )
    else:
        m2 = _safe_div(
            sum(ordinal_map.get(f.status, 0.0) for f in findings),
            len(findings),
        )

    m3 = _safe_div(
        sum(
            ordinal_map.get(f.status, 0.0) * f.evidence_strength * f.weight
            for f in findings
        ),
        total_weight,
    )

    evidence_presence = sum(1.0 if f.evidence else 0.0 for f in findings) / len(findings)
    avg_evidence_strength = _safe_div(
        sum(f.evidence_strength for f in findings if f.evidence),
        len([f for f in findings if f.evidence]),
    )
    grounded_items = sum(
        1.0 if f.evidence and f.source_documents else 0.0 for f in findings
    ) / len(findings)
    m5 = (evidence_presence + avg_evidence_strength + grounded_items) / 3.0

    return ComplianceScores(
        m2_ordinal_score=round(m2, 4),
        m3_evidence_weighted_score=round(m3, 4),
        m5_grounding_score=round(m5, 4),
    )
