from __future__ import annotations

from functools import lru_cache
import re

import numpy as np

from app.schemas.compliance import ComplianceEvidenceItem, ComplianceFinding
from app.services.llm.config import get_env
from app.services.retrieval.faiss_retrieval import normalize_whitespace

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
SUPPORTED_RELATIONS = {"entailment", "contradiction", "neutral"}
NEGATION_MARKERS = (
    " no ",
    " not ",
    " none ",
    " never ",
    " without ",
    " lacks ",
    " lack of ",
    " missing ",
    " does not ",
    " do not ",
    " did not ",
    " is not ",
    " are not ",
    " was not ",
    " were not ",
    " cannot ",
    " can't ",
    " wont ",
    " won't ",
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "its",
    "must",
    "of",
    "on",
    "or",
    "shall",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "to",
    "use",
    "used",
    "where",
}


def apply_evidence_contradiction_verification(
    finding: ComplianceFinding,
) -> ComplianceFinding:
    verifier = _get_verifier()
    if verifier is None:
        return finding.model_copy(
            update={
                "nli_status": finding.nli_status or finding.status,
            }
        )

    evidence_items = _materialize_evidence_items(finding)
    if not evidence_items or not normalize_whitespace(finding.requirement):
        return finding.model_copy(
            update={
                "nli_status": finding.nli_status or finding.status,
                "verification_applied": False,
                "verification_notes": [],
            }
        )

    assessments = verifier.assess(
        requirement=finding.requirement,
        evidence_items=evidence_items,
    )
    if not assessments:
        return finding.model_copy(
            update={
                "nli_status": finding.nli_status or finding.status,
                "verification_applied": False,
                "verification_notes": [],
            }
        )

    contradiction_count = sum(
        1
        for item in assessments
        if item.nli_relation == "contradiction" and not item.supports_requirement
    )
    neutral_count = sum(1 for item in assessments if item.nli_relation == "neutral")
    supportive_count = sum(1 for item in assessments if item.supports_requirement)
    next_status = "not_satisfied" if contradiction_count > 0 else finding.status

    notes = _build_verification_notes(
        supportive_count=supportive_count,
        contradiction_count=contradiction_count,
        neutral_count=neutral_count,
        original_status=finding.status,
        verified_status=next_status,
    )

    return finding.model_copy(
        update={
            "llm_status": finding.llm_status or finding.status,
            "nli_status": next_status,
            "pre_verification_status": finding.pre_verification_status or finding.status,
            "contradiction_status_before": finding.status if contradiction_count > 0 else finding.contradiction_status_before,
            "evidence_items": assessments,
            "contradiction_detected": contradiction_count > 0,
            "contradiction_evidence_count": contradiction_count,
            "supporting_evidence_count": supportive_count,
            "neutral_evidence_count": neutral_count,
            "verification_applied": True,
            "contradiction_override_applied": contradiction_count > 0 and finding.status != next_status,
            "verification_notes": notes,
        }
    )


class _EvidenceContradictionVerifier:
    def __init__(
        self,
        *,
        model_name: str,
        min_entailment_score: float,
        min_contradiction_score: float,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._cross_encoder = CrossEncoder(model_name)
        self._min_entailment_score = min_entailment_score
        self._min_contradiction_score = min_contradiction_score
        self._label_lookup = _build_label_lookup(
            getattr(self._cross_encoder.model.config, "id2label", None)
        )

    def assess(
        self,
        *,
        requirement: str,
        evidence_items: list[ComplianceEvidenceItem],
    ) -> list[ComplianceEvidenceItem]:
        pairs = [
            (normalize_whitespace(requirement), normalize_whitespace(item.text))
            for item in evidence_items
            if normalize_whitespace(item.text)
        ]
        if not pairs:
            return evidence_items

        logits = np.asarray(
            self._cross_encoder.predict(
                pairs,
                show_progress_bar=False,
            )
        )
        if logits.ndim == 1:
            logits = np.expand_dims(logits, axis=0)
        probabilities = _softmax(logits)

        assessed: list[ComplianceEvidenceItem] = []
        pair_index = 0
        for item in evidence_items:
            if not normalize_whitespace(item.text):
                assessed.append(item)
                continue
            probs = probabilities[pair_index]
            pair_index += 1
            relation, score = self._classify(probabilities=probs)
            explicit_contradiction = _is_explicit_requirement_contradiction(
                requirement=requirement,
                evidence=item.text,
                relation=relation,
            )
            supports_requirement = not explicit_contradiction
            assessed.append(
                item.model_copy(
                    update={
                        "nli_relation": "contradiction" if explicit_contradiction else relation,
                        "nli_score": round(float(score), 4),
                        "supports_requirement": supports_requirement,
                    }
                )
            )
        return assessed

    def _classify(self, *, probabilities: np.ndarray) -> tuple[str, float]:
        scored_labels: list[tuple[str, float]] = []
        for index, score in enumerate(probabilities):
            label = self._label_lookup.get(index)
            if label in SUPPORTED_RELATIONS:
                scored_labels.append((label, float(score)))

        if not scored_labels:
            return "neutral", float(np.max(probabilities))

        relation, score = max(scored_labels, key=lambda item: item[1])
        if relation == "contradiction" and score < self._min_contradiction_score:
            return "neutral", score
        if relation == "entailment" and score < self._min_entailment_score:
            return "neutral", score
        return relation, score


def _materialize_evidence_items(finding: ComplianceFinding) -> list[ComplianceEvidenceItem]:
    if finding.evidence_items:
        return [
            item
            for item in finding.evidence_items
            if normalize_whitespace(item.text)
        ]
    return [
        ComplianceEvidenceItem(
            text=text,
            source_document=finding.source_document,
        )
        for text in finding.evidence
        if normalize_whitespace(text)
    ]


def _build_verification_notes(
    *,
    supportive_count: int,
    contradiction_count: int,
    neutral_count: int,
    original_status: str,
    verified_status: str,
) -> list[str]:
    notes: list[str] = []
    if contradiction_count > 0:
        notes.append(
            f"Cross-encoder detected {contradiction_count} contradiction evidence item(s)."
        )
    if neutral_count > 0:
        notes.append(
            f"Cross-encoder marked {neutral_count} evidence item(s) as insufficient or weak."
        )
    if supportive_count > 0:
        notes.append(
            f"Cross-encoder retained {supportive_count} evidence item(s) as supportive."
        )
    if verified_status != original_status:
        notes.append(f"Status adjusted from {original_status} to {verified_status}.")
    return notes


def _is_explicit_requirement_contradiction(
    *,
    requirement: str,
    evidence: str,
    relation: str,
) -> bool:
    if relation != "contradiction":
        return False

    normalized_evidence = f" {normalize_whitespace(evidence).lower()} "
    if not normalized_evidence.strip():
        return False
    if not any(marker in normalized_evidence for marker in NEGATION_MARKERS):
        return False

    requirement_keywords = _extract_keywords(requirement)
    evidence_keywords = _extract_keywords(evidence)
    if not requirement_keywords or not evidence_keywords:
        return False

    overlap = requirement_keywords & evidence_keywords
    return len(overlap) >= 3


def _extract_keywords(text: str) -> set[str]:
    normalized = normalize_whitespace(text).lower()
    if not normalized:
        return set()
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) >= 4 and token not in STOPWORDS
    }


def _build_label_lookup(id2label: object) -> dict[int, str]:
    if not isinstance(id2label, dict):
        return {}
    lookup: dict[int, str] = {}
    for raw_index, raw_label in id2label.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        normalized = str(raw_label or "").strip().lower()
        if normalized.endswith("entailment"):
            lookup[index] = "entailment"
        elif normalized.endswith("contradiction"):
            lookup[index] = "contradiction"
        elif normalized.endswith("neutral"):
            lookup[index] = "neutral"
    return lookup


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _is_enabled() -> bool:
    raw = str(get_env("NLI_CONTRADICTION_ENABLED", "true") or "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _iter_float_env(key: str, default: float) -> float:
    raw = get_env(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def _get_verifier() -> _EvidenceContradictionVerifier | None:
    if not _is_enabled():
        return None
    model_name = get_env("NLI_CONTRADICTION_MODEL", DEFAULT_NLI_MODEL) or DEFAULT_NLI_MODEL
    try:
        return _EvidenceContradictionVerifier(
            model_name=model_name,
            min_entailment_score=_iter_float_env("NLI_MIN_ENTAILMENT_SCORE", 0.5),
            min_contradiction_score=_iter_float_env("NLI_MIN_CONTRADICTION_SCORE", 0.5),
        )
    except Exception as exc:
        print(f"[NLI] contradiction verifier unavailable: {exc}", flush=True)
        return None
