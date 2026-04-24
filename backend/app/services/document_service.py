from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from typing import Any

from app.schemas.documents import DocumentRecord, DocumentType, is_record_document_type
from app.schemas.deliverables import DeliverableExtractionResponse, DeliverableItem
from app.schemas.parsing import ParsedDocument
from app.services.parsing.parser_service import parse_document
from app.services.retrieval.record_index_service import remove_record_index
from app.services.retrieval.reference_index_service import remove_reference_index
from app.services.storage_paths import (
    DOCUMENT_REGISTRY_PATH,
    DOCUMENTS_DIR,
    EXTRACTION_DIR,
    PARSED_DIR,
    PROCEDURE_EXTRACTION_DIR,
    RETRIEVAL_DIR,
    STORAGE_DIR,
    UPLOAD_DIR,
    get_procedure_document_extraction_history_dir,
    get_procedure_document_extraction_latest_path,
    get_procedure_extraction_history_dir,
    get_procedure_extraction_latest_path,
)

REGISTRY_PATH = DOCUMENT_REGISTRY_PATH


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_document_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []

    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8-sig") as file:
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
        if "frozen" not in item:
            item["frozen"] = False
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


def ensure_procedure_not_frozen(document: DocumentRecord, *, action: str) -> None:
    if document.document_type == DocumentType.procedure and document.frozen:
        raise HTTPException(
            status_code=409,
            detail=f'Procedure "{document.source_filename}" is frozen and cannot be used for {action}.',
        )


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
        if is_record_document_type(document.document_type):
            resolved_records.append(document.stored_filename)

    return resolved_records


def remove_document_files(document: DocumentRecord, registry: list[dict] | None = None) -> None:
    stored_path = Path(document.stored_at)
    has_other_references = False
    has_other_parsed_references = False
    has_other_reference_entries = False
    has_other_record_entries = False
    has_other_procedure_entries = False
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
        has_other_reference_entries = any(
            item.get("stored_filename") != document.stored_filename
            and item.get("content_hash") == document.content_hash
            and item.get("document_type") == DocumentType.reference.value
            for item in registry
        )
        has_other_record_entries = any(
            item.get("stored_filename") != document.stored_filename
            and item.get("content_hash") == document.content_hash
            and _is_record_like_document_type(item.get("document_type"))
            for item in registry
        )
        has_other_procedure_entries = any(
            item.get("stored_filename") != document.stored_filename
            and item.get("content_hash") == document.content_hash
            and item.get("document_type") == DocumentType.procedure.value
            for item in registry
        )

    if stored_path.exists() and not has_other_references:
        stored_path.unlink()

    cached_path = get_cached_parse_path(document)
    if cached_path.exists() and not has_other_parsed_references:
        cached_path.unlink()

    if not has_other_reference_entries:
        remove_reference_index(
            {
                "stored_filename": document.stored_filename,
                "content_hash": document.content_hash,
            }
        )
    if not has_other_record_entries:
        remove_record_index(
            {
                "stored_filename": document.stored_filename,
                "content_hash": document.content_hash,
            }
        )

    procedure_document_dir = _get_document_extraction_dir(document)
    if procedure_document_dir.exists():
        for path in sorted(procedure_document_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        procedure_document_dir.rmdir()

    if not has_other_procedure_entries:
        extraction_dir = _get_procedure_extraction_dir_for_content(document)
        if extraction_dir.exists():
            for path in sorted(extraction_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            extraction_dir.rmdir()


def _is_record_like_document_type(value: str | None) -> bool:
    return is_record_document_type(value)


def _get_procedure_extraction_dir_for_content(document: DocumentRecord) -> Path:
    content_hash = document.content_hash or ""
    if not content_hash:
        raise HTTPException(status_code=500, detail="Procedure document is missing content_hash")
    return PROCEDURE_EXTRACTION_DIR / content_hash


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
        return PARSED_DIR / f"{document.content_hash}.json"

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

    if document.document_type != DocumentType.procedure:
        raise HTTPException(
            status_code=400,
            detail="Deliverable extraction is supported only for procedure documents.",
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

    if document.document_type != DocumentType.procedure:
        raise HTTPException(
            status_code=400,
            detail="Deliverable extraction results are supported only for procedure documents.",
        )

    path = _resolve_document_extraction_latest_path(document)
    if not path.exists():
        raise HTTPException(status_code=404, detail="No saved deliverable extraction found for this document")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Deliverable extraction file corrupted") from exc

    payload.setdefault("created_at", current_timestamp())
    payload.setdefault("saved_at", path.as_posix())
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

    if document.document_type != DocumentType.procedure:
        raise HTTPException(
            status_code=400,
            detail="Deliverable extraction results are supported only for procedure documents.",
        )

    source_path = _resolve_document_extraction_latest_path(document)
    path = _get_document_extraction_latest_save_path(document)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="No saved deliverable extraction found for this document")
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
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
    _write_document_extraction_history_snapshot(document, response)
    return response


def list_document_deliverable_results(stored_filename: str) -> list[DeliverableExtractionResponse]:
    from app.services.deliverable_methods.extraction_method_common import (
        compute_deliverable_confidence,
    )

    registry = load_document_registry()
    _, item = find_document_or_404(registry, stored_filename)
    document = DocumentRecord(**item)

    if document.document_type != DocumentType.procedure:
        raise HTTPException(
            status_code=400,
            detail="Deliverable extraction results are supported only for procedure documents.",
        )

    history_dir = _get_document_extraction_history_dir(document)
    history_paths = sorted(history_dir.glob("*.json"), reverse=True)

    if not history_paths and document.content_hash:
        history_paths = sorted(get_procedure_extraction_history_dir(document.content_hash).glob("*.json"), reverse=True)
    if not history_paths and document.content_hash:
        legacy_document_history_dir = (
            PROCEDURE_EXTRACTION_DIR / document.content_hash / "documents" / document.stored_filename / "history"
        )
        if legacy_document_history_dir.exists():
            history_paths = sorted(legacy_document_history_dir.glob("*.json"), reverse=True)

    responses: list[DeliverableExtractionResponse] = []
    for path in history_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail="Deliverable extraction file corrupted") from exc

        if payload.get("document_stored_filename") not in (None, stored_filename):
            continue

        payload.setdefault("created_at", current_timestamp())
        payload.setdefault("saved_at", path.as_posix())
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
        responses.append(response)

    return responses


def _get_document_extraction_dir(document: DocumentRecord) -> Path:
    content_hash = document.content_hash or ""
    return get_procedure_document_extraction_history_dir(content_hash, document.stored_filename).parent


def _get_document_extraction_latest_save_path(document: DocumentRecord) -> Path:
    content_hash = document.content_hash or ""
    return get_procedure_document_extraction_latest_path(content_hash, document.stored_filename)


def _resolve_document_extraction_latest_path(document: DocumentRecord) -> Path:
    latest_path = _get_document_extraction_latest_save_path(document)
    if latest_path.exists():
        return latest_path

    content_hash = document.content_hash or ""
    if content_hash:
        legacy_document_dir = PROCEDURE_EXTRACTION_DIR / content_hash / "documents" / document.stored_filename
        legacy_document_latest_path = legacy_document_dir / "latest.json"
        if legacy_document_latest_path.exists():
            return legacy_document_latest_path

        legacy_history_paths = sorted(get_procedure_extraction_history_dir(content_hash).glob("*.json"), reverse=True)
        for path in legacy_history_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if payload.get("document_stored_filename") == document.stored_filename:
                return path

        legacy_latest_path = get_procedure_extraction_latest_path(content_hash)
        if legacy_latest_path.exists():
            return legacy_latest_path

    return latest_path


def _get_document_extraction_history_dir(document: DocumentRecord) -> Path:
    content_hash = document.content_hash or ""
    return get_procedure_document_extraction_history_dir(content_hash, document.stored_filename)


def _write_document_extraction_history_snapshot(
    document: DocumentRecord,
    response: DeliverableExtractionResponse,
) -> None:
    content_hash = document.content_hash or ""
    if not content_hash:
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history_path = _get_document_extraction_history_dir(document) / (
        f"{timestamp}_{response.extraction_provider}_{response.extraction_model}_{uuid4().hex}.json"
    )
    history_path.write_text(
        response.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )


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
