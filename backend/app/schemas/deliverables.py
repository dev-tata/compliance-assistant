from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

RequirementType = Literal[
    "document_output",
    "recorded_information",
    "approval_or_signoff",
    "update_or_notification",
    "archival_or_storage",
    "change_control",
    "validation_activity",
]


class DeliverableExtractionRequest(BaseModel):
    provider: str
    model: str = "gpt-5.4-nano"
    instructions: str | None = None


class DeliverableUpdateRequest(BaseModel):
    deliverables: list["DeliverableItem"] = Field(default_factory=list)


class DeliverableItem(BaseModel):
    section_label: str
    heading_title: str
    requirement_text: str
    requirement_type: RequirementType
    mandatory: bool = True
    source_quote: str
    source_document: str
    required_by_procedure: bool = True
    weight: float = Field(default=1.0, gt=0.0)
    validated_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def populate_required_fields_from_legacy_payload(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        if "requirement_text" not in data or not data.get("requirement_text"):
            data["requirement_text"] = data.get("description") or data.get("name") or ""
        if "source_quote" not in data or not data.get("source_quote"):
            data["source_quote"] = data.get("requirement_text") or data.get("description") or data.get("name") or ""
        if "requirement_type" not in data or not data.get("requirement_type"):
            data["requirement_type"] = "recorded_information"
        if "section_label" not in data or not data.get("section_label"):
            data["section_label"] = "unknown"
        if "heading_title" not in data or not data.get("heading_title"):
            data["heading_title"] = "Unknown section"
        if "validated_confidence" not in data or data.get("validated_confidence") is None:
            data["validated_confidence"] = data.get("confidence", 0.0)
        if "weight" not in data or not data.get("weight"):
            data["weight"] = 1.0
        return data

    @model_validator(mode="after")
    def populate_derived_fields(self) -> "DeliverableItem":
        normalized_requirement = " ".join(self.requirement_text.split()).strip()
        normalized_quote = " ".join(self.source_quote.split()).strip()
        self.requirement_text = normalized_requirement
        self.source_quote = (
            ""
            if normalized_quote == normalized_requirement
            else normalized_quote
        )
        return self


class DeliverableExtractionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    case_id: str | None = None
    document_stored_filename: str | None = None
    source_filename: str | None = None
    extraction_provider: str = Field(
        validation_alias=AliasChoices("extraction_provider", "provider"),
        serialization_alias="extraction_provider",
    )
    extraction_model: str = Field(
        validation_alias=AliasChoices("extraction_model", "model"),
        serialization_alias="extraction_model",
    )
    prompt_version: str = "deliverable_extraction_v2"
    parser_version: str = "unknown"
    created_at: str
    saved_at: str
    deliverables: list[DeliverableItem] = Field(default_factory=list)


class DeliverableExtractionSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    case_id: str
    file_name: str
    created_at: str
    saved_at: str
    extraction_provider: str = Field(
        validation_alias=AliasChoices("extraction_provider", "provider"),
        serialization_alias="extraction_provider",
    )
    extraction_model: str = Field(
        validation_alias=AliasChoices("extraction_model", "model"),
        serialization_alias="extraction_model",
    )
    prompt_version: str = "deliverable_extraction_v2"
    parser_version: str = "unknown"
    deliverable_count: int = 0
