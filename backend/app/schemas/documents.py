from __future__ import annotations

from enum import Enum

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class DocumentType(str, Enum):
    procedure = "procedure"
    record = "record"
    template = "template"
    registry = "registry"
    risk_assessment = "risk_assessment"
    requirement_specification = "requirement_specification"
    validation_plan = "validation_plan"
    validation_report = "validation_report"
    test_plan = "test_plan"
    test_execution = "test_execution"
    change_request = "change_request"
    reference = "reference"


class DocumentLanguage(str, Enum):
    en = "en"
    sv = "sv"
    mixed = "mixed"


class DocumentRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source_filename: str = Field(
        validation_alias=AliasChoices("source_filename", "filename"),
        serialization_alias="source_filename",
    )
    created_at: str | None = None
    stored_filename: str
    stored_at: str
    document_type: DocumentType | None = None
    language: DocumentLanguage | None = None
    group_id: str | None = None
    parsed_json_at: str | None = None
    content_hash: str | None = None
