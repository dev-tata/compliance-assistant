from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

from typing import Any

from app.schemas.documents import DocumentRecord, DocumentType
from app.schemas.deliverables import DeliverableExtractionResponse, DeliverableItem
from app.schemas.parsing import ParsedDocument
from app.services.parsing.parser_service import parse_document

STORAGE_DIR = Path("storage")
CACHE_DIR = STORAGE_DIR / "cache"
UPLOAD_DIR = STORAGE_DIR / "uploads"
PARSED_DIR = STORAGE_DIR / "parsed"
COMPLIANCE_DIR = STORAGE_DIR / "compliance"
DELIVERABLES_DIR = STORAGE_DIR / "deliverables"
INDEXES_DIR = CACHE_DIR / "indexes"
REGISTRY_PATH = STORAGE_DIR / "document_registry.json"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PARSED_DIR.mkdir(parents=True, exist_ok=True)
COMPLIANCE_DIR.mkdir(parents=True, exist_ok=True)
DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)
INDEXES_DIR.mkdir(parents=True, exist_ok=True)


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_document_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []

    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
            registry = json.load(file)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Document registry corrupted")

    normalized_registry: list[dict] = []
    changed = False
    for item in registry:
        document_type = item.get("document_type")
        procedure_group_id = item.get("group_id")

        if document_type == DocumentType.procedure.value and not procedure_group_id:
            procedure_group_id = extract_procedure_group_id(
                item.get("source_filename") or item.get("filename") or ""
            )
        elif not document_type and not procedure_group_id:
            procedure_group_id = extract_procedure_group_id(
                item.get("source_filename") or item.get("filename") or ""
            )
            if procedure_group_id:
                item.setdefault("document_type", DocumentType.procedure.value)

        if item.get("document_type") == DocumentType.procedure.value and procedure_group_id:
            item.setdefault("group_id", procedure_group_id)

        if "created_at" not in item:
            item["created_at"] = current_timestamp()
            changed = True
        if "parsed_json_at" not in item:
            item["parsed_json_at"] = None
            changed = True
        if "language" not in item:
            item["language"] = None
            changed = True
        if not item.get("content_hash"):
            stored_at = item.get("stored_at")
            if stored_at and Path(stored_at).exists():
                item["content_hash"] = compute_file_hash(Path(stored_at))
                changed = True
            else:
                item["content_hash"] = None
        normalized_registry.append(DocumentRecord(**item).model_dump(by_alias=True))
    if changed:
        save_document_registry(normalized_registry)
    return normalized_registry


def save_document_registry(registry: list[dict]) -> None:
    with open(REGISTRY_PATH, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2, ensure_ascii=False)


def find_document_or_404(registry: list[dict], stored_filename: str) -> tuple[int, dict]:
    for index, item in enumerate(registry):
        if item["stored_filename"] == stored_filename:
            return index, item
    raise HTTPException(status_code=404, detail="Document not found")


def get_or_parse_document(
    registry: list[dict],
    index: int,
    document: DocumentRecord,
) -> ParsedDocument:
    cached_path = get_cached_parse_path(document)
    if cached_path.exists():
        if registry[index].get("parsed_json_at") != cached_path.as_posix():
            registry[index]["parsed_json_at"] = cached_path.as_posix()
            save_document_registry(registry)
        parsed_document = ParsedDocument.model_validate_json(cached_path.read_text(encoding="utf-8"))
        return parsed_document

    parsed_document = parse_document(document)
    cached_path.write_text(
        parsed_document.model_dump_json(
            indent=2,
            exclude_none=True,
        ),
        encoding="utf-8",
    )
    registry[index]["parsed_json_at"] = cached_path.as_posix()
    save_document_registry(registry)
    return parsed_document


def resolve_case_record_filenames(
    registry: list[dict],
    procedure_stored_filenames: list[str],
    record_stored_filenames: list[str] | None = None,
) -> list[str]:
    explicit_records = [item for item in (record_stored_filenames or []) if item]
    if explicit_records:
        return explicit_records

    procedure_group_ids: set[str] = set()
    for stored_filename in procedure_stored_filenames:
        _, item = find_document_or_404(registry, stored_filename)
        document = DocumentRecord(**item)
        if document.group_id:
            procedure_group_ids.add(document.group_id)

    if not procedure_group_ids:
        return []

    resolved_records: list[str] = []
    for item in registry:
        document = DocumentRecord(**item)
        if document.group_id not in procedure_group_ids:
            continue
        if document.document_type in {DocumentType.procedure, DocumentType.reference}:
            continue
        resolved_records.append(document.stored_filename)

    return resolved_records


def remove_document_files(document: DocumentRecord, registry: list[dict] | None = None) -> None:
    stored_path = Path(document.stored_at)
    has_other_references = False
    has_other_parsed_references = False
    if registry is not None:
        has_other_references = any(
            item.get("stored_filename") != document.stored_filename
            and item.get("stored_at") == document.stored_at
            for item in registry
        )
        has_other_parsed_references = any(
            item.get("stored_filename") != document.stored_filename
            and (
                item.get("content_hash") == document.content_hash
                or item.get("parsed_json_at") == get_cached_parse_path(document).as_posix()
            )
            for item in registry
        )

    if stored_path.exists() and not has_other_references:
        stored_path.unlink()

    cached_path = get_cached_parse_path(document)
    if cached_path.exists() and not has_other_parsed_references:
        cached_path.unlink()

    for deliverable_path in _get_legacy_document_deliverable_paths(document):
        if deliverable_path.exists():
            deliverable_path.unlink()

    if not has_other_parsed_references:
        for deliverable_path in _get_shared_document_deliverable_paths(document):
            if deliverable_path.exists():
                deliverable_path.unlink()


def compute_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_document_by_content_hash(
    registry: list[dict],
    content_hash: str,
) -> dict | None:
    for item in registry:
        if item.get("content_hash") == content_hash:
            return item
    return None


def validate_extension(filename: str) -> None:
    allowed_suffixes = {".pdf", ".docx", ".xlsx"}
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix or 'missing extension'}",
        )


def extract_procedure_group_id(filename: str) -> str | None:
    match = re.match(r"^(?P<group_id>\d{4})-v\.\d+(?:\.\d+)*\b", filename, re.IGNORECASE)
    if not match:
        return None
    return match.group("group_id")


def get_cached_parse_path(document: DocumentRecord) -> Path:
    if document.content_hash:
        prefix = f"{document.group_id}_" if document.group_id else ""
        source_stem = Path(document.source_filename).name
        return PARSED_DIR / f"{prefix}{document.content_hash}_{source_stem}.json"

    if document.parsed_json_at:
        return Path(document.parsed_json_at)

    base_name = (
        f"{document.group_id}_{document.stored_filename}"
        if document.group_id
        else document.stored_filename
    )
    return PARSED_DIR / f"{base_name}.json"


def get_document_extraction_payload(stored_filename: str) -> dict[str, Any]:
    registry = load_document_registry()
    index, item = find_document_or_404(registry, stored_filename)
    document = DocumentRecord(**item)

    if document.document_type not in {DocumentType.procedure, DocumentType.reference}:
        raise HTTPException(
            status_code=400,
            detail="Deliverable extraction is supported only for procedure and reference documents.",
        )

    try:
        get_or_parse_document(registry, index, document)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document = DocumentRecord(**registry[index])
    parsed_payload = _load_parsed_json_file(document)
    return {
        "case_id": None,
        "title": document.source_filename,
        "notes": None,
        "procedures": [
            {
                "document_type": document.document_type.value if document.document_type else None,
                "source_filename": document.source_filename,
                "stored_filename": document.stored_filename,
                "group_id": document.group_id,
                "language": document.language.value if document.language else None,
                "content_hash": document.content_hash,
                "parsed_json": parsed_payload,
            }
        ],
        "records": [],
    }


def get_latest_document_deliverable_result(stored_filename: str) -> DeliverableExtractionResponse:
    from app.services.deliverable_methods.extraction_method_common import (
        compute_deliverable_confidence,
    )

    registry = load_document_registry()
    _, item = find_document_or_404(registry, stored_filename)
    document = DocumentRecord(**item)

    if document.document_type not in {DocumentType.procedure, DocumentType.reference}:
        raise HTTPException(
            status_code=400,
            detail="Deliverable extraction results are supported only for procedure and reference documents.",
        )

    matching_paths = _get_document_deliverable_paths(document, stored_filename=stored_filename)
    if not matching_paths:
        raise HTTPException(status_code=404, detail="No saved deliverable extraction found for this document")

    path = matching_paths[0]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Deliverable extraction file corrupted") from exc

    payload.setdefault("created_at", current_timestamp())
    payload.setdefault("saved_at", path.as_posix())
    payload.setdefault("method", "non_rag")
    payload.setdefault("document_stored_filename", stored_filename)
    payload.setdefault("source_filename", document.source_filename)

    response = DeliverableExtractionResponse(**payload)
    normalized_deliverables = [
        item.model_copy(
            update={
                "confidence": compute_deliverable_confidence(item),
            }
        )
        for item in response.deliverables
    ]
    if any(
        abs(item.confidence - normalized.confidence) > 1e-9
        for item, normalized in zip(response.deliverables, normalized_deliverables)
    ):
        response = response.model_copy(update={"deliverables": normalized_deliverables})
        path.write_text(
            response.model_dump_json(
                indent=2,
                exclude_none=True,
            ),
            encoding="utf-8",
        )
    return response


def update_latest_document_deliverable_result(
    stored_filename: str,
    deliverables: list[DeliverableItem],
) -> DeliverableExtractionResponse:
    from app.services.deliverable_methods.extraction_method_common import (
        compute_deliverable_confidence,
    )

    registry = load_document_registry()
    _, item = find_document_or_404(registry, stored_filename)
    document = DocumentRecord(**item)

    if document.document_type not in {DocumentType.procedure, DocumentType.reference}:
        raise HTTPException(
            status_code=400,
            detail="Deliverable extraction results are supported only for procedure and reference documents.",
        )

    matching_paths = _get_document_deliverable_paths(document, stored_filename=stored_filename)
    if not matching_paths:
        raise HTTPException(status_code=404, detail="No saved deliverable extraction found for this document")

    path = matching_paths[0]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Deliverable extraction file corrupted") from exc

    normalized_deliverables = [
        item.model_copy(
            update={
                "confidence": compute_deliverable_confidence(item),
            }
        )
        for item in deliverables
    ]

    response = DeliverableExtractionResponse(
        **{
            **payload,
            "deliverables": [item.model_dump() for item in normalized_deliverables],
            "saved_at": path.as_posix(),
            "document_stored_filename": stored_filename,
            "source_filename": document.source_filename,
        }
    )
    path.write_text(
        response.model_dump_json(
            indent=2,
            exclude_none=True,
        ),
        encoding="utf-8",
    )
    return response


def _get_document_deliverable_paths(
    document: DocumentRecord,
    *,
    stored_filename: str | None = None,
) -> list[Path]:
    paths: dict[str, Path] = {}
    for path in _get_document_deliverable_paths_by_patterns(
        _build_document_deliverable_patterns(document, include_legacy=True)
    ):
        paths[path.name] = path
    sorted_paths = sorted(paths.values(), key=_deliverable_sort_key, reverse=True)
    if not stored_filename:
        return sorted_paths

    exact_paths = [
        path
        for path in sorted_paths
        if _deliverable_belongs_to_stored_filename(path, stored_filename)
    ]
    if exact_paths:
        return exact_paths
    return sorted_paths


def _get_document_deliverable_paths_by_patterns(patterns: list[str]) -> list[Path]:
    matching_paths: list[Path] = []
    for pattern in patterns:
        matching_paths.extend(DELIVERABLES_DIR.glob(pattern))
    return matching_paths


def _build_document_deliverable_patterns(document: DocumentRecord) -> list[str]:
    return _build_document_deliverable_patterns(document, include_legacy=True)


def _build_document_deliverable_patterns(
    document: DocumentRecord,
    *,
    include_legacy: bool,
) -> list[str]:
    patterns: list[str] = []
    if document.content_hash:
        patterns.append(f"document_{document.content_hash}_*_deliverables_*.json")
    if include_legacy:
        patterns.append(f"document_{document.stored_filename}_deliverables_*.json")
    return patterns


def _get_shared_document_deliverable_paths(document: DocumentRecord) -> list[Path]:
    return sorted(
        _get_document_deliverable_paths_by_patterns(
            _build_document_deliverable_patterns(document, include_legacy=False)
        ),
        key=_deliverable_sort_key,
        reverse=True,
    )


def _get_legacy_document_deliverable_paths(document: DocumentRecord) -> list[Path]:
    return sorted(
        DELIVERABLES_DIR.glob(f"document_{document.stored_filename}_deliverables_*.json"),
        key=_deliverable_sort_key,
        reverse=True,
    )


def _deliverable_sort_key(path: Path) -> tuple[str, float]:
    timestamp_match = re.search(r"_deliverables_(\d{8}T\d{6}Z)_", path.name)
    if timestamp_match:
        return (timestamp_match.group(1), path.stat().st_mtime)
    return ("", path.stat().st_mtime)


def _deliverable_belongs_to_stored_filename(path: Path, stored_filename: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("document_stored_filename") == stored_filename


def _load_parsed_json_file(document: DocumentRecord) -> dict[str, Any]:
    if not document.parsed_json_at:
        raise HTTPException(
            status_code=500,
            detail=f"Parsed JSON path missing for document {document.stored_filename}",
        )

    parsed_path = Path(document.parsed_json_at)
    if not parsed_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Parsed JSON file not found for document {document.stored_filename}",
        )

    try:
        with open(parsed_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Parsed JSON file corrupted for document {document.stored_filename}",
        ) from exc
