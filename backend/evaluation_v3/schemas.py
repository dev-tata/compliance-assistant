from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ComplianceLabel = Literal["satisfied", "partial", "not_satisfied"]
StageKey = Literal["stage_1", "stage_2", "stage_3", "final"]
EvidenceStatus = Literal["supported", "partial", "missing", "conflicting"]
RequirementType = Literal[
    "residual_risk_acceptability",
    "benefit_risk_rationale",
    "single_field",
    "relationship",
    "list_or_table",
    "per_function",
    "control_measure",
    "conditional",
    "generic",
]
ContradictionType = Literal[
    "none",
    "missing_evidence",
    "direct_conflict",
    "wrong_entity",
    "temporal_conflict",
    "reference_clarification",
    "reference_conflict",
]

EVALUATION_V3_FINDING_FIELDS: Final[tuple[str, ...]] = (
    "weight",
    "required_evidence_count",
    "evidence_status",
    "contradiction_type",
    "evidence_score",
)

EVALUATION_V3_ANALYSIS_METRICS: Final[tuple[str, ...]] = (
    "satisfied_count",
    "partial_count",
    "not_satisfied_count",
    "supported_count",
    "missing_count",
    "conflicting_count",
    "avg_grounded_evidence_count",
    "avg_evidence_coverage_ratio",
)


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


class DeliverableNode(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    deliverable_id: str
    source_document: str = ""
    section_label: str = ""
    heading_title: str = ""
    requirement_text: str
    weight: float = Field(default=1.0, gt=0.0)
    required_evidence_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def normalize_fields(self) -> "DeliverableNode":
        self.deliverable_id = _normalize_text(self.deliverable_id)
        self.source_document = _normalize_text(self.source_document)
        self.section_label = _normalize_text(self.section_label)
        self.heading_title = _normalize_text(self.heading_title)
        self.requirement_text = _normalize_text(self.requirement_text)
        return self


class EvidenceNode(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    evidence_id: str = ""
    source_document: str = ""
    section_id: str = ""
    subsection_id: str = ""
    section_label: str = ""
    heading_title: str = ""
    text: str
    reranker_score: float | None = None
    raw_retrieval_score: float | None = None
    retrieval_score: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def normalize_fields(self) -> "EvidenceNode":
        self.evidence_id = _normalize_text(self.evidence_id)
        self.source_document = _normalize_text(self.source_document)
        self.section_id = _normalize_text(self.section_id)
        self.subsection_id = _normalize_text(self.subsection_id) or self.section_id
        self.section_label = _normalize_text(self.section_label)
        self.heading_title = _normalize_text(self.heading_title)
        self.text = _normalize_text(self.text)
        return self


class ReferenceNode(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    reference_id: str = ""
    source_document: str = ""
    section_id: str = ""
    subsection_id: str = ""
    section_label: str = ""
    heading_title: str = ""
    text: str
    reranker_score: float | None = None
    raw_retrieval_score: float | None = None
    retrieval_score: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def normalize_fields(self) -> "ReferenceNode":
        self.reference_id = _normalize_text(self.reference_id)
        self.source_document = _normalize_text(self.source_document)
        self.section_id = _normalize_text(self.section_id)
        self.subsection_id = _normalize_text(self.subsection_id) or self.section_id
        self.section_label = _normalize_text(self.section_label)
        self.heading_title = _normalize_text(self.heading_title)
        self.text = _normalize_text(self.text)
        return self


class StageJudgment(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    stage_key: StageKey
    label: ComplianceLabel | None = None
    rationale: str = ""
    conflict_flag: bool = False
    supporting_record_evidence_ids: list[str] = Field(default_factory=list)
    supporting_record_evidence_items: list[dict[str, str]] = Field(default_factory=list)
    supporting_reference_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_fields(self) -> "StageJudgment":
        self.rationale = _normalize_text(self.rationale)
        self.supporting_record_evidence_ids = _dedupe_preserve_order(
            [_normalize_text(item) for item in self.supporting_record_evidence_ids if _normalize_text(item)]
        )
        normalized_items: list[dict[str, str]] = []
        for item in self.supporting_record_evidence_items:
            if not isinstance(item, dict):
                continue
            normalized = {
                "evidence_id": _normalize_text(item.get("evidence_id")),
                "section_id": _normalize_text(item.get("section_id")),
                "subsection_id": _normalize_text(item.get("subsection_id")),
                "section_label": _normalize_text(item.get("section_label")),
                "heading_title": _normalize_text(item.get("heading_title")),
                "source_document": _normalize_text(item.get("source_document")),
                "text": _normalize_text(item.get("text")),
            }
            if normalized["text"]:
                normalized_items.append(normalized)
        self.supporting_record_evidence_items = normalized_items
        self.supporting_reference_ids = _dedupe_preserve_order(
            [_normalize_text(item) for item in self.supporting_reference_ids if _normalize_text(item)]
        )
        return self


class MiniKGLinks(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    deliverable_id: str
    stage_judgment_keys: list[StageKey] = Field(default_factory=list)
    record_evidence_ids: list[str] = Field(default_factory=list)
    reference_ids: list[str] = Field(default_factory=list)
    stage_1_record_evidence_ids: list[str] = Field(default_factory=list)
    stage_2_record_evidence_ids: list[str] = Field(default_factory=list)
    stage_3_record_evidence_ids: list[str] = Field(default_factory=list)
    stage_3_reference_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_fields(self) -> "MiniKGLinks":
        self.deliverable_id = _normalize_text(self.deliverable_id)
        self.stage_judgment_keys = _dedupe_preserve_order(
            [_normalize_text(item) for item in self.stage_judgment_keys if _normalize_text(item)]
        )
        self.record_evidence_ids = _dedupe_preserve_order(
            [_normalize_text(item) for item in self.record_evidence_ids if _normalize_text(item)]
        )
        self.reference_ids = _dedupe_preserve_order(
            [_normalize_text(item) for item in self.reference_ids if _normalize_text(item)]
        )
        self.stage_1_record_evidence_ids = _dedupe_preserve_order(
            [_normalize_text(item) for item in self.stage_1_record_evidence_ids if _normalize_text(item)]
        )
        self.stage_2_record_evidence_ids = _dedupe_preserve_order(
            [_normalize_text(item) for item in self.stage_2_record_evidence_ids if _normalize_text(item)]
        )
        self.stage_3_record_evidence_ids = _dedupe_preserve_order(
            [_normalize_text(item) for item in self.stage_3_record_evidence_ids if _normalize_text(item)]
        )
        self.stage_3_reference_ids = _dedupe_preserve_order(
            [_normalize_text(item) for item in self.stage_3_reference_ids if _normalize_text(item)]
        )
        return self


class VerifierResult(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    verifier_name: str = ""
    passed: bool = False
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: str = ""
    flagged_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_fields(self) -> "VerifierResult":
        self.verifier_name = _normalize_text(self.verifier_name)
        self.notes = _normalize_text(self.notes)
        self.flagged_fields = _dedupe_preserve_order(
            [_normalize_text(item) for item in self.flagged_fields if _normalize_text(item)]
        )
        return self


class EvaluationV3Metrics(BaseModel):
    satisfied_count: int = Field(default=0, ge=0)
    partial_count: int = Field(default=0, ge=0)
    not_satisfied_count: int = Field(default=0, ge=0)
    supported_count: int = Field(default=0, ge=0)
    missing_count: int = Field(default=0, ge=0)
    conflicting_count: int = Field(default=0, ge=0)
    avg_grounded_evidence_count: float = Field(default=0.0, ge=0.0)
    avg_evidence_coverage_ratio: float = Field(default=0.0, ge=0.0)


class EvaluationV3ResultRow(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    deliverable_id: str
    final_label: ComplianceLabel | None = None
    stage_1_label: ComplianceLabel | None = None
    stage_2_label: ComplianceLabel | None = None
    stage_3_label: ComplianceLabel | None = None
    stage_1_evidence_status: EvidenceStatus = "missing"
    stage_2_evidence_status: EvidenceStatus = "missing"
    stage_3_evidence_status: EvidenceStatus = "missing"
    stage_1_grounded_evidence_count: int = Field(default=0, ge=0)
    stage_2_grounded_evidence_count: int = Field(default=0, ge=0)
    stage_3_grounded_evidence_count: int = Field(default=0, ge=0)
    stage_1_evidence_coverage_ratio: float = Field(default=0.0, ge=0.0)
    stage_2_evidence_coverage_ratio: float = Field(default=0.0, ge=0.0)
    stage_3_evidence_coverage_ratio: float = Field(default=0.0, ge=0.0)
    evidence_status: EvidenceStatus
    grounded_evidence_count: int = Field(default=0, ge=0)
    grounded_chunk_count: int = Field(default=0, ge=0)
    required_evidence_count: int = Field(default=1, ge=0)
    evidence_coverage_ratio: float = Field(default=0.0, ge=0.0)
    has_conflict: bool = False
    contradiction_type: ContradictionType = "none"

    @model_validator(mode="after")
    def normalize_fields(self) -> "EvaluationV3ResultRow":
        self.deliverable_id = _normalize_text(self.deliverable_id)
        return self


class EvaluationV3Result(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    case_id: str
    created_at: str
    source_compliance_saved_at: str
    compliance_provider: str = ""
    compliance_model: str = ""
    method: str = ""
    metrics: dict[str, int | float] = Field(default_factory=dict)
    units: list[EvaluationV3ResultRow] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_fields(self) -> "EvaluationV3Result":
        self.case_id = _normalize_text(self.case_id)
        self.created_at = _normalize_text(self.created_at)
        self.source_compliance_saved_at = _normalize_text(self.source_compliance_saved_at)
        self.compliance_provider = _normalize_text(self.compliance_provider)
        self.compliance_model = _normalize_text(self.compliance_model)
        self.method = _normalize_text(self.method)
        return self


class EvaluationUnit(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    deliverable: DeliverableNode
    weight: float = Field(default=1.0, gt=0.0)
    requirement_type: RequirementType = "generic"
    base_required_evidence_count: int = Field(default=1, ge=1)
    weight_modifier: int = Field(default=0, ge=0)
    required_evidence_count_reason: str = ""
    required_evidence_count: int = Field(default=1, ge=0)
    evidence_status: EvidenceStatus = "missing"
    contradiction_type: ContradictionType = "none"
    evidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    record_evidence_chunks: list[EvidenceNode] = Field(default_factory=list)
    reference_evidence_chunks: list[ReferenceNode] = Field(default_factory=list)
    stage_1_answer: StageJudgment = Field(default_factory=lambda: StageJudgment(stage_key="stage_1"))
    stage_2_answer: StageJudgment = Field(default_factory=lambda: StageJudgment(stage_key="stage_2"))
    stage_3_answer: StageJudgment = Field(default_factory=lambda: StageJudgment(stage_key="stage_3"))
    final_label: ComplianceLabel | None = None
    final_rationale: str = ""
    mini_kg_links: MiniKGLinks | None = None
    verifier_result: VerifierResult | None = None
    metrics: EvaluationV3Metrics | None = None

    @model_validator(mode="after")
    def validate_cross_references(self) -> "EvaluationUnit":
        self.required_evidence_count_reason = _normalize_text(self.required_evidence_count_reason)
        self.final_rationale = _normalize_text(self.final_rationale)
        if self.weight <= 0:
            raise ValueError("weight must be greater than 0")
        if self.required_evidence_count == 0 and self.evidence_status == "supported":
            raise ValueError("evidence_status cannot be 'supported' when required_evidence_count is 0")
        if self.weight != self.deliverable.weight:
            self.weight = self.deliverable.weight

        record_ids = {
            item.evidence_id
            for item in self.record_evidence_chunks
            if item.evidence_id
        }
        reference_ids = {
            item.reference_id
            for item in self.reference_evidence_chunks
            if item.reference_id
        }

        if self.mini_kg_links is not None and self.mini_kg_links.deliverable_id != self.deliverable.deliverable_id:
            raise ValueError("mini_kg_links.deliverable_id must match deliverable.deliverable_id")

        self._validate_stage_links(self.stage_1_answer, record_ids, reference_ids)
        self._validate_stage_links(self.stage_2_answer, record_ids, reference_ids)
        self._validate_stage_links(self.stage_3_answer, record_ids, reference_ids)

        if self.mini_kg_links is not None:
            invalid_stage_keys = [
                item
                for item in self.mini_kg_links.stage_judgment_keys
                if item not in {"stage_1", "stage_2", "stage_3", "final"}
            ]
            if invalid_stage_keys:
                raise ValueError(f"mini_kg_links.stage_judgment_keys contains invalid keys: {invalid_stage_keys}")
            self._validate_link_ids(
                link_ids=self.mini_kg_links.record_evidence_ids,
                valid_ids=record_ids,
                field_name="mini_kg_links.record_evidence_ids",
            )
            self._validate_link_ids(
                link_ids=self.mini_kg_links.reference_ids,
                valid_ids=reference_ids,
                field_name="mini_kg_links.reference_ids",
            )
            self._validate_link_ids(
                link_ids=self.mini_kg_links.stage_1_record_evidence_ids,
                valid_ids=record_ids,
                field_name="mini_kg_links.stage_1_record_evidence_ids",
            )
            self._validate_link_ids(
                link_ids=self.mini_kg_links.stage_2_record_evidence_ids,
                valid_ids=record_ids,
                field_name="mini_kg_links.stage_2_record_evidence_ids",
            )
            self._validate_link_ids(
                link_ids=self.mini_kg_links.stage_3_record_evidence_ids,
                valid_ids=record_ids,
                field_name="mini_kg_links.stage_3_record_evidence_ids",
            )
            self._validate_link_ids(
                link_ids=self.mini_kg_links.stage_3_reference_ids,
                valid_ids=reference_ids,
                field_name="mini_kg_links.stage_3_reference_ids",
            )

        return self

    @staticmethod
    def _validate_stage_links(
        stage: StageJudgment,
        record_ids: set[str],
        reference_ids: set[str],
    ) -> None:
        EvaluationUnit._validate_link_ids(
            link_ids=stage.supporting_record_evidence_ids,
            valid_ids=record_ids,
            field_name=f"{stage.stage_key}.supporting_record_evidence_ids",
        )
        EvaluationUnit._validate_link_ids(
            link_ids=stage.supporting_reference_ids,
            valid_ids=reference_ids,
            field_name=f"{stage.stage_key}.supporting_reference_ids",
        )

    @staticmethod
    def _validate_link_ids(
        *,
        link_ids: list[str],
        valid_ids: set[str],
        field_name: str,
    ) -> None:
        missing = [item for item in link_ids if item not in valid_ids]
        if missing:
            raise ValueError(f"{field_name} contains unknown ids: {missing}")
