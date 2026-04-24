from __future__ import annotations

import mimetypes
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.schemas.documents import (
    DocumentFreezeUpdateRequest,
    DocumentLanguage,
    DocumentRecord,
    DocumentType,
    is_record_document_type,
)
from app.schemas.deliverables import (
    DeliverableExtractionRequest,
    DeliverableExtractionResponse,
    DeliverableUpdateRequest,
)
from app.schemas.parsing import ParsedDocument
from app.services.document_service import (
    UPLOAD_DIR,
    compute_file_hash,
    current_timestamp,
    ensure_procedure_not_frozen,
    extract_procedure_group_id,
    find_document_by_content_hash,
    find_document_or_404,
    get_document_extraction_payload,
    get_latest_document_deliverable_result,
    get_or_parse_document,
    list_document_deliverable_results,
    load_document_registry,
    remove_document_files,
    save_document_registry,
    update_latest_document_deliverable_result,
    validate_extension,
)
from app.services.deliverable_extraction_service import run_document_deliverable_extraction
from app.services.llm.errors import LLMConfigurationError, LLMGenerationError
from app.services.retrieval.record_index_service import ensure_record_index
from app.services.retrieval.reference_index_service import ensure_reference_index

router = APIRouter()


@router.post("/documents/upload", response_model=DocumentRecord)
async def upload_document(
    file: UploadFile = File(...),
    document_type: DocumentType = Form(...),
    language: DocumentLanguage = Form(...),
    group_id: str | None = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    validate_extension(file.filename)

    stored_filename = f"{uuid4()}_{file.filename}"
    file_path = UPLOAD_DIR / stored_filename

    with open(file_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)

    content_hash = compute_file_hash(file_path)

    registry = load_document_registry()
    existing_item = find_document_by_content_hash(registry, content_hash)
    canonical_file_path = file_path
    if existing_item:
        if file_path.exists():
            file_path.unlink()
        canonical_file_path = UPLOAD_DIR / Path(existing_item["stored_at"]).name

    procedure_group_id = group_id
    if document_type == DocumentType.procedure and not procedure_group_id:
        procedure_group_id = extract_procedure_group_id(file.filename)

    record = {
        "source_filename": file.filename,
        "created_at": current_timestamp(),
        "stored_filename": stored_filename,
        "stored_at": canonical_file_path.as_posix(),
        "document_type": document_type,
        "language": language,
        "group_id": procedure_group_id,
        "parsed_json_at": None,
        "content_hash": content_hash,
        "frozen": False,
    }

    registry.append(record)
    save_document_registry(registry)

    try:
        document = DocumentRecord(**record)
        parsed_document = get_or_parse_document(registry, len(registry) - 1, document)
        if document.document_type == DocumentType.reference:
            ensure_reference_index(
                {
                    "source_filename": document.source_filename,
                    "stored_filename": document.stored_filename,
                    "content_hash": document.content_hash,
                    "document_type": document.document_type.value,
                    "parsed_json": parsed_document.model_dump(mode="json"),
                }
            )
        elif is_record_document_type(document.document_type):
            ensure_record_index(
                {
                    "source_filename": document.source_filename,
                    "stored_filename": document.stored_filename,
                    "content_hash": document.content_hash,
                    "document_type": document.document_type.value,
                    "parsed_json": parsed_document.model_dump(mode="json"),
                }
            )
        updated_record = registry[-1]
        return DocumentRecord(**updated_record)
    except (ValueError, RuntimeError) as exc:
        if file_path.exists():
            file_path.unlink()
        registry.pop()
        save_document_registry(registry)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/documents", response_model=list[DocumentRecord])
def list_documents():
    return [DocumentRecord(**item) for item in load_document_registry()]


@router.delete("/documents/{stored_filename}", response_model=DocumentRecord)
def delete_document(stored_filename: str):
    registry = load_document_registry()

    for index, item in enumerate(registry):
        if item["stored_filename"] != stored_filename:
            continue

        document = DocumentRecord(**item)
        ensure_procedure_not_frozen(document, action="deletion")
        removed_item = registry.pop(index)
        document = DocumentRecord(**removed_item)
        remove_document_files(document, registry)
        save_document_registry(registry)
        return document

    raise HTTPException(status_code=404, detail="Document not found")


@router.patch("/documents/{stored_filename}/freeze", response_model=DocumentRecord)
def update_document_freeze(stored_filename: str, request: DocumentFreezeUpdateRequest):
    registry = load_document_registry()
    index, item = find_document_or_404(registry, stored_filename)
    document = DocumentRecord(**item)
    if document.document_type != DocumentType.procedure:
        raise HTTPException(status_code=400, detail="Freeze is supported only for procedure documents.")

    registry[index]["frozen"] = request.frozen
    save_document_registry(registry)
    return DocumentRecord(**registry[index])


@router.get("/documents/parse/{stored_filename}", response_model=ParsedDocument)
def parse_document_by_stored_filename(stored_filename: str):
    registry = load_document_registry()
    index, item = find_document_or_404(registry, stored_filename)

    try:
        document = DocumentRecord(**item)
        return get_or_parse_document(registry, index, document)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/documents/file/{stored_filename}")
def get_document_file(stored_filename: str):
    registry = load_document_registry()
    _, item = find_document_or_404(registry, stored_filename)
    document = DocumentRecord(**item)
    media_type, _ = mimetypes.guess_type(document.source_filename)

    return FileResponse(
        path=document.stored_at,
        filename=document.source_filename,
        media_type=media_type or "application/octet-stream",
        content_disposition_type="inline",
    )


@router.post("/documents/{stored_filename}/deliverables/extract", response_model=DeliverableExtractionResponse)
def extract_document_deliverables(stored_filename: str, request: DeliverableExtractionRequest):
    registry = load_document_registry()
    _, item = find_document_or_404(registry, stored_filename)
    ensure_procedure_not_frozen(DocumentRecord(**item), action="requirement updates")
    document_payload = get_document_extraction_payload(stored_filename)

    try:
        return run_document_deliverable_extraction(
            stored_filename=stored_filename,
            document_payload=document_payload,
            request=request,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/documents/{stored_filename}/deliverables/latest", response_model=DeliverableExtractionResponse)
def get_latest_document_deliverables(stored_filename: str):
    return get_latest_document_deliverable_result(stored_filename)


@router.get("/documents/{stored_filename}/deliverables/history", response_model=list[DeliverableExtractionResponse])
def get_document_deliverables_history(stored_filename: str):
    return list_document_deliverable_results(stored_filename)


@router.put("/documents/{stored_filename}/deliverables/latest", response_model=DeliverableExtractionResponse)
def update_latest_document_deliverables(stored_filename: str, request: DeliverableUpdateRequest):
    registry = load_document_registry()
    _, item = find_document_or_404(registry, stored_filename)
    ensure_procedure_not_frozen(DocumentRecord(**item), action="requirement updates")
    return update_latest_document_deliverable_result(
        stored_filename,
        request.deliverables,
    )
