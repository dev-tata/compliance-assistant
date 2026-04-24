from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


ComplianceStatus = Literal["satisfied", "partial", "not_satisfied"]
ComplianceMethod = Literal["non_rag", "single_source_rag", "multi_source_rag"]
ComplianceRequirementSource = Literal["auto", "procedure_sections", "deliverables"]

class ComplianceRequest(BaseModel):
    provider: str
    model: str
    method: ComplianceMethod = "non_rag"
    instructions: str | None = None
    requirement_source: ComplianceRequirementSource = "deliverables"
    deliverable_file_name: str | None = None
    selected_deliverables_by_document: dict[str, list[str]] = Field(default_factory=dict)
    additional_document_filenames: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_requirement_source(self) -> "ComplianceRequest":
        if self.method in {"non_rag", "single_source_rag", "multi_source_rag"}:
            self.requirement_source = "deliverables"
        return self

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_method_names(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data["method"] = _normalize_compliance_method(data.get("method"))
        return data


class ComplianceFinding(BaseModel):
    requirement: str
    status: ComplianceStatus
    evidence: list[str] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    evidence_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    weight: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="before")
    @classmethod
    def normalize_status(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data["status"] = _normalize_compliance_status(data.get("status"))
        data["evidence"] = _normalize_string_list(data.get("evidence"))
        data["source_documents"] = _normalize_string_list(data.get("source_documents"))
        return data

class ComplianceLinkedRow(BaseModel):
    requirement: str = ""
    requirement_ref: str = ""
    status: ComplianceStatus
    gap: str = ""
    recommendation: str = ""
    record_recall_at_k: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def normalize_status(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data["status"] = _normalize_compliance_status(data.get("status"))
        return data


class ComplianceAnalysis(BaseModel):
    overall_assessment: str
    completion_percent: int = Field(default=0, ge=0, le=100)
    gaps: list[str] = Field(default_factory=list)
    linked_rows: list[ComplianceLinkedRow] = Field(default_factory=list)
    findings: list[ComplianceFinding] = Field(default_factory=list)
    procedure_to_record: list[ComplianceFinding] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def sync_directional_views(self) -> "ComplianceAnalysis":
        self.gaps = _normalize_string_list(self.gaps)
        self.recommended_actions = _normalize_string_list(self.recommended_actions)
        if self.procedure_to_record and not self.findings:
            self.findings = self.procedure_to_record
        elif self.findings and not self.procedure_to_record:
            self.procedure_to_record = self.findings
        if self.linked_rows:
            normalized_gaps = [item for item in self.gaps if item and item.strip()]
            normalized_recommendations = [
                item for item in self.recommended_actions if item and item.strip()
            ]

            if normalized_gaps:
                self.gaps = normalized_gaps
            if normalized_recommendations:
                self.recommended_actions = normalized_recommendations

            if not self.gaps:
                self.gaps = [row.gap for row in self.linked_rows if row.gap]
            if not self.recommended_actions:
                self.recommended_actions = [
                    row.recommendation for row in self.linked_rows if row.recommendation
                ]
        return self


class ComplianceScores(BaseModel):
    m2_ordinal_score: float
    m3_evidence_weighted_score: float
    m5_grounding_score: float


class SectionMatch(BaseModel):
    procedure_document: str
    procedure_section_label: str | None = None
    procedure_heading_title: str | None = None
    record_document: str | None = None
    record_section_label: str | None = None
    record_heading_title: str | None = None
    match_percent: float = Field(ge=0.0, le=100.0)
    match_basis: str


class RetrievalMetrics(BaseModel):
    record_recall_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    record_k: int | None = Field(default=None, ge=1)
    evaluated_requirements: int = Field(default=0, ge=0)
    hit_requirements: int = Field(default=0, ge=0)


class ComplianceResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    case_id: str
    compliance_provider: str = Field(
        validation_alias=AliasChoices("compliance_provider", "provider"),
        serialization_alias="compliance_provider",
    )
    compliance_model: str = Field(
        validation_alias=AliasChoices("compliance_model", "model"),
        serialization_alias="compliance_model",
    )
    extraction_provider: str | None = None
    extraction_model: str | None = None
    method: ComplianceMethod = "non_rag"
    reference_stored_filenames: list[str] = Field(default_factory=list)
    created_at: str
    saved_at: str
    analysis: ComplianceAnalysis
    scores: ComplianceScores
    section_matches: list[SectionMatch] = Field(default_factory=list)
    retrieval_metrics: RetrievalMetrics | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_method_names(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data["method"] = _normalize_compliance_method(data.get("method"))
        return data


def _normalize_status_value(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_compliance_method(value: object) -> str:
    normalized = _normalize_status_value(value)
    alias_map = {
        "simple_rag": "single_source_rag",
        "nested_rag": "multi_source_rag",
    }
    return alias_map.get(normalized, normalized)


def _normalize_compliance_status(value: object) -> str:
    normalized = _normalize_status_value(value)
    alias_map = {
        "matched": "satisfied",
        "unmatched": "not_satisfied",
        "unsatisfied": "not_satisfied",
        "not_matched": "not_satisfied",
        "no_match": "not_satisfied",
    }
    return alias_map.get(normalized, normalized)


def _normalize_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, list):
        normalized_items: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                normalized_items.append(text)
        return normalized_items
    text = str(value).strip()
    return [text] if text else []
