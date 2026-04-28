from __future__ import annotations

import shutil
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.schemas.cases import (
    CaseCreate,
    CaseDocuments,
    CaseRecord,
    CaseRecordDocumentsUpdate,
    ComplianceSummary,
    ParsedCase,
)
from app.schemas.compliance import ComplianceRequest, ComplianceResponse
from app.schemas.documents import DocumentRecord, is_record_document_type
from app.services.case_service import (
    _load_parsed_json_file,
    delete_case_compliance_result,
    get_case_deliverables_payload,
    get_case_compliance_result,
    get_case_compliance_payload,
    get_case_documents,
    get_case_or_404,
    get_parsed_case_by_id,
    get_case_procedure_deliverables_payload,
    list_all_compliances,
    list_case_compliances,
    load_case_registry,
    save_case_registry,
)
from app.services.compliance_service import (
    run_case_compliance_analysis,
)
from app.services.document_service import (
    current_timestamp,
    find_document_or_404,
    get_or_parse_document,
    load_document_registry,
    resolve_case_record_filenames,
)
from app.services.retrieval.record_index_service import prepare_record_indexes
from app.services.retrieval.reference_index_service import prepare_reference_indexes
from app.services.llm.errors import (
    LLMConfigurationError,
    LLMGenerationError,
    LLMQuotaExceededError,
)
from app.services.storage_paths import get_case_dir

router = APIRouter(prefix="/cases", tags=["cases"])


@router.post("", response_model=CaseRecord)
def create_case(case: CaseCreate):
    if not case.procedure_stored_filenames:
        raise HTTPException(status_code=400, detail="At least one procedure document is required")

    documents = load_document_registry()
    resolved_record_filenames = resolve_case_record_filenames(
        documents,
        case.procedure_stored_filenames,
        case.record_stored_filenames,
    )

    for stored_filename in case.procedure_stored_filenames:
        find_document_or_404(documents, stored_filename)
    for stored_filename in resolved_record_filenames:
        find_document_or_404(documents, stored_filename)
    for stored_filename in case.reference_stored_filenames:
        find_document_or_404(documents, stored_filename)

    if not resolved_record_filenames:
        raise HTTPException(
            status_code=400,
            detail="At least one record document with a matching group ID is required",
        )

    case_record = {
        "case_id": str(uuid4()),
        "created_at": current_timestamp(),
        "title": case.title,
        "procedure_stored_filenames": case.procedure_stored_filenames,
        "record_stored_filenames": resolved_record_filenames,
        "reference_stored_filenames": case.reference_stored_filenames,
        "notes": case.notes,
    }

    registry = load_case_registry()
    registry.append(case_record)
    save_case_registry(registry)
    return CaseRecord(**case_record)


@router.get("", response_model=list[CaseRecord])
def list_cases():
    return [CaseRecord(**item) for item in load_case_registry()]


@router.get("/compliances", response_model=list[ComplianceSummary])
def get_all_compliances():
    return list_all_compliances()


@router.get("/{case_id}", response_model=CaseRecord)
def get_case(case_id: str):
    return CaseRecord(**get_case_or_404(case_id))


@router.patch("/{case_id}/records", response_model=CaseRecord)
def update_case_records(case_id: str, update: CaseRecordDocumentsUpdate):
    registry = load_case_registry()
    documents = load_document_registry()

    for index, item in enumerate(registry):
        if item["case_id"] != case_id:
            continue

        next_record_filenames: list[str] = []
        seen: set[str] = set()
        for stored_filename in update.record_stored_filenames:
            if not stored_filename or stored_filename in seen:
                continue
            _, document_item = find_document_or_404(documents, stored_filename)
            document = DocumentRecord(**document_item)
            if not is_record_document_type(document.document_type):
                raise HTTPException(
                    status_code=400,
                    detail=f'Document "{document.source_filename}" is not a record document.',
                )
            next_record_filenames.append(stored_filename)
            seen.add(stored_filename)

        if not next_record_filenames:
            raise HTTPException(
                status_code=400,
                detail="At least one record document is required",
            )

        registry[index] = {
            **item,
            "record_stored_filenames": next_record_filenames,
        }
        save_case_registry(registry)
        return CaseRecord(**registry[index])

    raise HTTPException(status_code=404, detail="Case not found")


@router.delete("/{case_id}", response_model=CaseRecord)
def delete_case(case_id: str):
    registry = load_case_registry()

    for index, item in enumerate(registry):
        if item["case_id"] != case_id:
            continue

        removed_case = CaseRecord(**item)
        registry.pop(index)
        save_case_registry(registry)
        _remove_case_compliance_files(case_id)
        return removed_case

    raise HTTPException(status_code=404, detail="Case not found")


@router.get("/{case_id}/documents", response_model=CaseDocuments)
def get_case_document_set(case_id: str):
    return get_case_documents(case_id)


@router.get("/{case_id}/parsed", response_model=ParsedCase)
def get_parsed_case(case_id: str):
    return get_parsed_case_by_id(case_id)


@router.get("/{case_id}/compliances", response_model=list[ComplianceSummary])
def get_case_compliances(case_id: str):
    return list_case_compliances(case_id)


@router.get("/{case_id}/compliances/{file_name}", response_model=ComplianceResponse)
def get_case_compliance_result_by_file(case_id: str, file_name: str):
    return get_case_compliance_result(case_id, file_name)


@router.delete("/{case_id}/compliances/{file_name}", response_model=ComplianceSummary)
def delete_case_compliance_result_by_file(case_id: str, file_name: str):
    return delete_case_compliance_result(case_id, file_name)


@router.post("/{case_id}/compliance", response_model=ComplianceResponse)
def run_case_compliance(case_id: str, request: ComplianceRequest):
    case_payload = get_case_compliance_payload(case_id)
    if request.method == "two_stage_rag" and request.additional_document_filenames:
        _append_additional_documents(
            case_payload=case_payload,
            additional_document_filenames=request.additional_document_filenames,
        )
    if not case_payload["procedures"]:
        raise HTTPException(
            status_code=400,
            detail="Case must contain at least one procedure document for compliance.",
        )
    if not case_payload["records"]:
        raise HTTPException(
            status_code=400,
            detail="Case must contain at least one record document for compliance.",
        )

    try:
        deliverables_payload = (
            get_case_deliverables_payload(case_id, request.deliverable_file_name)
            if request.deliverable_file_name
            else get_case_procedure_deliverables_payload(case_id)
        )
        if deliverables_payload is None:
            raise HTTPException(
                status_code=400,
                detail="Compliance requires extracted deliverables for all procedure documents in the case.",
            )

        deliverables = list(deliverables_payload.get("deliverables", []))
        selected_by_document = request.selected_deliverables_by_document
        if selected_by_document:
            filtered_deliverables = []
            for item in deliverables:
                source_document = item.get("source_document")
                selected_names = selected_by_document.get(source_document or "", [])
                if not selected_names or item.get("requirement_text") in selected_names:
                    filtered_deliverables.append(item)
            deliverables = filtered_deliverables
        if not deliverables:
            raise HTTPException(
                status_code=400,
                detail="Compliance requires at least one extracted deliverable.",
            )

        case_payload["deliverables"] = deliverables
        if deliverables_payload.get("extraction_provider"):
            case_payload["extraction_provider"] = deliverables_payload.get("extraction_provider")
        if deliverables_payload.get("extraction_model"):
            case_payload["extraction_model"] = deliverables_payload.get("extraction_model")

        _validate_compliance_inputs(case_payload=case_payload, request=request)

        return run_case_compliance_analysis(
            case_id=case_id,
            case_payload=case_payload,
            request=request,
        )
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMQuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except LLMGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _remove_case_compliance_files(case_id: str) -> None:
    case_dir = get_case_dir(case_id)
    if case_dir.exists():
        shutil.rmtree(case_dir)


def _append_additional_documents(
    *,
    case_payload: dict,
    additional_document_filenames: list[str],
) -> None:
    documents = load_document_registry()
    existing_by_stored_filename = {
        item.get("stored_filename")
        for group in ("procedures", "records", "references")
        for item in case_payload.get(group, [])
    }

    for stored_filename in additional_document_filenames:
        if stored_filename in existing_by_stored_filename:
            continue

        index, item = find_document_or_404(documents, stored_filename)
        document = DocumentRecord(**item)
        try:
            get_or_parse_document(documents, index, document)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        payload = _build_compliance_payload_for_document(documents[index])
        if payload["document_type"] == "reference":
            case_payload.setdefault("references", []).append(payload)
        else:
            raise HTTPException(
                status_code=400,
                detail="Only reference documents can be added to nested RAG runs.",
            )
        existing_by_stored_filename.add(stored_filename)


def _build_compliance_payload_for_document(item: dict) -> dict:
    document = DocumentRecord(**item)
    return {
        "document_type": document.document_type.value if document.document_type else None,
        "source_filename": document.source_filename,
        "stored_filename": document.stored_filename,
        "group_id": document.group_id,
        "language": document.language.value if document.language else None,
        "content_hash": document.content_hash,
        "parsed_json": _load_parsed_json_file(document),
    }


def _validate_compliance_inputs(*, case_payload: dict, request: ComplianceRequest) -> None:
    deliverables = case_payload.get("deliverables", [])
    records = case_payload.get("records", [])
    references = case_payload.get("references", [])

    if not deliverables:
        raise HTTPException(status_code=400, detail="Compliance requires extracted deliverables.")
    if not records:
        raise HTTPException(status_code=400, detail="Compliance requires record documents.")

    if not prepare_record_indexes(records):
        raise HTTPException(
            status_code=400,
            detail=f"{request.method} requires retrievable record sections and record indexes.",
        )

    if not references:
        raise HTTPException(
            status_code=400,
            detail="two_stage_rag requires reference documents with retrievable sections.",
        )
    if not prepare_reference_indexes(references):
        raise HTTPException(
            status_code=400,
            detail="two_stage_rag requires retrievable reference sections and reference indexes.",
        )
