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


RECORD_DOCUMENT_TYPES = frozenset(
    {
        DocumentType.record,
        DocumentType.template,
        DocumentType.registry,
        DocumentType.risk_assessment,
        DocumentType.requirement_specification,
        DocumentType.validation_plan,
        DocumentType.validation_report,
        DocumentType.test_plan,
        DocumentType.test_execution,
        DocumentType.change_request,
    }
)


def is_record_document_type(value: DocumentType | str | None) -> bool:
    if value is None:
        return False
    if isinstance(value, DocumentType):
        return value in RECORD_DOCUMENT_TYPES
    try:
        return DocumentType(value) in RECORD_DOCUMENT_TYPES
    except ValueError:
        return False


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
    frozen: bool = False


class DocumentFreezeUpdateRequest(BaseModel):
    frozen: bool
