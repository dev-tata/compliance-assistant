from __future__ import annotations

import shutil
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.schemas.cases import CaseCreate, CaseDocuments, CaseRecord, ComplianceSummary, ParsedCase
from app.schemas.compliance import ComplianceRequest, ComplianceResponse
from app.services.case_service import (
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
    COMPLIANCE_DIR,
    DELIVERABLES_DIR,
    INDEXES_DIR,
    current_timestamp,
    find_document_or_404,
    load_document_registry,
    resolve_case_record_filenames,
)
from app.services.llm.errors import LLMConfigurationError, LLMGenerationError

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
        deliverables_payload = None
        if request.requirement_source == "deliverables":
            deliverables_payload = (
                get_case_deliverables_payload(case_id, request.deliverable_file_name)
                if request.deliverable_file_name
                else get_case_procedure_deliverables_payload(case_id)
            )
        elif request.requirement_source == "auto":
            deliverables_payload = get_case_procedure_deliverables_payload(case_id)

        if deliverables_payload is not None:
            deliverables = deliverables_payload.get("deliverables", [])
            selected_by_document = request.selected_deliverables_by_document
            if selected_by_document:
                filtered_deliverables = []
                for item in deliverables:
                    source_document = item.get("source_document")
                    selected_names = selected_by_document.get(source_document or "", [])
                    if not selected_names or item.get("requirement_text") in selected_names:
                        filtered_deliverables.append(item)
                deliverables = filtered_deliverables
            case_payload["deliverables"] = deliverables
            if deliverables_payload.get("extraction_provider"):
                case_payload["extraction_provider"] = deliverables_payload.get("extraction_provider")
            if deliverables_payload.get("extraction_model"):
                case_payload["extraction_model"] = deliverables_payload.get("extraction_model")

        return run_case_compliance_analysis(
            case_id=case_id,
            case_payload=case_payload,
            request=request,
        )
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _remove_case_compliance_files(case_id: str) -> None:
    for path in COMPLIANCE_DIR.glob(f"case_{case_id}_compliance_*.json"):
        if path.exists():
            path.unlink()
    for path in DELIVERABLES_DIR.glob(f"case_{case_id}_deliverables_*.json"):
        if path.exists():
            path.unlink()
    case_index_dir = INDEXES_DIR / f"case_{case_id}"
    if case_index_dir.exists():
        shutil.rmtree(case_index_dir)
