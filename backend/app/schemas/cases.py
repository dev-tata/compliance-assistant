from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.documents import DocumentRecord
from app.schemas.parsing import ParsedDocument


class CaseCreate(BaseModel):
    title: str
    procedure_stored_filenames: list[str] = Field(default_factory=list)
    record_stored_filenames: list[str] = Field(default_factory=list)
    reference_stored_filenames: list[str] = Field(default_factory=list)
    notes: str | None = None


class CaseRecordDocumentsUpdate(BaseModel):
    record_stored_filenames: list[str] = Field(default_factory=list)


class CaseRecord(BaseModel):
    case_id: str
    created_at: str | None = None
    title: str
    procedure_stored_filenames: list[str]
    record_stored_filenames: list[str]
    reference_stored_filenames: list[str] = Field(default_factory=list)
    notes: str | None = None


class CaseDocuments(BaseModel):
    case_id: str
    title: str
    procedure_documents: list[DocumentRecord]
    record_documents: list[DocumentRecord]
    reference_documents: list[DocumentRecord]


class ParsedCase(BaseModel):
    case_id: str
    title: str
    procedure_documents: list[ParsedDocument]
    record_documents: list[ParsedDocument]
    reference_documents: list[ParsedDocument]


class ComplianceSummary(BaseModel):
    case_id: str
    file_name: str
    created_at: str
    saved_at: str
    provider: str
    model: str
    method: str = "non_rag"
    overall_assessment: str
    completion_percent: int = Field(default=0, ge=0, le=100)
    satisfied_count: int = Field(default=0, ge=0)
    partial_count: int = Field(default=0, ge=0)
    not_satisfied_count: int = Field(default=0, ge=0)
    m3_evidence_weighted_score: float = Field(default=0.0, ge=0.0)
    m5_grounding_score: float = Field(default=0.0, ge=0.0)
    reference_stored_filenames: list[str] = Field(default_factory=list)
