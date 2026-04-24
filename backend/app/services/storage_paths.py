from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STORAGE_DIR = Path("storage")

DOCUMENTS_DIR = STORAGE_DIR / "documents"
UPLOAD_DIR = DOCUMENTS_DIR / "uploads"
PARSED_DIR = DOCUMENTS_DIR / "parsed"
DOCUMENT_REGISTRY_PATH = DOCUMENTS_DIR / "registry.json"

EXTRACTION_DIR = STORAGE_DIR / "extraction"
PROCEDURE_EXTRACTION_DIR = EXTRACTION_DIR

RETRIEVAL_DIR = STORAGE_DIR / "retrieval"
REFERENCE_INDEXES_DIR = RETRIEVAL_DIR / "references"
RECORD_INDEXES_DIR = RETRIEVAL_DIR / "records"

CASES_DIR = STORAGE_DIR / "cases"
CASE_REGISTRY_PATH = CASES_DIR / "registry.json"

for path in (
    STORAGE_DIR,
    DOCUMENTS_DIR,
    UPLOAD_DIR,
    PARSED_DIR,
    EXTRACTION_DIR,
    RETRIEVAL_DIR,
    REFERENCE_INDEXES_DIR,
    RECORD_INDEXES_DIR,
    CASES_DIR,
):
    path.mkdir(parents=True, exist_ok=True)


def get_procedure_extraction_dir(content_hash: str) -> Path:
    directory = PROCEDURE_EXTRACTION_DIR / content_hash
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_procedure_extraction_latest_path(content_hash: str) -> Path:
    return get_procedure_extraction_dir(content_hash) / "latest.json"


def get_procedure_extraction_history_dir(content_hash: str) -> Path:
    directory = get_procedure_extraction_dir(content_hash) / "history"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_procedure_document_extraction_dir(content_hash: str, stored_filename: str) -> Path:
    directory = PROCEDURE_EXTRACTION_DIR / stored_filename
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_procedure_document_extraction_latest_path(content_hash: str, stored_filename: str) -> Path:
    return get_procedure_document_extraction_dir(content_hash, stored_filename) / "latest.json"


def get_procedure_document_extraction_history_dir(content_hash: str, stored_filename: str) -> Path:
    directory = get_procedure_document_extraction_dir(content_hash, stored_filename) / "history"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_case_dir(case_id: str) -> Path:
    directory = CASES_DIR / case_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_case_manifest_path(case_id: str) -> Path:
    return get_case_dir(case_id) / "case.json"


def get_case_compliance_dir(case_id: str) -> Path:
    directory = get_case_dir(case_id) / "compliance"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_case_manifest(case_record: dict[str, Any]) -> None:
    case_id = case_record.get("case_id")
    if not case_id:
        return
    get_case_manifest_path(case_id).write_text(
        json.dumps(case_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
