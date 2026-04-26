from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.schemas.cases import CaseDocuments, CaseRecord, ComplianceSummary, ParsedCase
from app.schemas.compliance import ComplianceFinding, ComplianceResponse
from app.schemas.deliverables import (
    DeliverableExtractionRequest,
    DeliverableExtractionResponse,
    DeliverableExtractionSummary,
)
from app.schemas.documents import DocumentRecord
from app.schemas.parsing import ParsedDocument
from app.services.deliverable_extraction_service import run_document_deliverable_extraction
from app.services.document_service import (
    current_timestamp,
    find_document_or_404,
    get_document_extraction_payload,
    get_latest_document_deliverable_result,
    get_or_parse_document,
    load_document_registry,
)
from app.services.compliance_methods.compliance_method_common import (
    compute_completion_percent,
    compute_overall_assessment_from_findings,
)
from app.services.storage_paths import CASES_DIR, CASE_REGISTRY_PATH, get_case_compliance_dir, write_case_manifest

CASES_DIR.mkdir(parents=True, exist_ok=True)


def load_case_registry() -> list[dict]:
    if not CASE_REGISTRY_PATH.exists():
        return []

    try:
        with open(CASE_REGISTRY_PATH, "r", encoding="utf-8-sig") as file:
            registry = json.load(file)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Case registry corrupted")

    normalized_registry: list[dict] = []
    changed = False
    for item in registry:
        procedure_stored_filenames = item.get("procedure_stored_filenames")
        record_stored_filenames = item.get("record_stored_filenames")

        # Support older registry entries written before the case schema rename.
        if procedure_stored_filenames is None:
            procedure_stored_filenames = item.get("source_stored_filenames", [])
        if record_stored_filenames is None:
            record_stored_filenames = []
        reference_stored_filenames = item.get("reference_stored_filenames", [])

        if "created_at" not in item:
            item["created_at"] = current_timestamp()
            changed = True
        normalized_registry.append(
            CaseRecord(
                case_id=item.get("case_id"),
                created_at=item.get("created_at"),
                title=item.get("title"),
                procedure_stored_filenames=procedure_stored_filenames,
                record_stored_filenames=record_stored_filenames,
                reference_stored_filenames=reference_stored_filenames,
                notes=item.get("notes"),
            ).model_dump()
        )

    if changed:
        save_case_registry(normalized_registry)
    return normalized_registry


def save_case_registry(registry: list[dict]) -> None:
    with open(CASE_REGISTRY_PATH, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2, ensure_ascii=False)
    for item in registry:
        write_case_manifest(item)


def get_case_or_404(case_id: str) -> dict:
    registry = load_case_registry()
    for item in registry:
        if item["case_id"] == case_id:
            return item
    raise HTTPException(status_code=404, detail="Case not found")


def get_case_documents(case_id: str) -> CaseDocuments:
    case = CaseRecord(**get_case_or_404(case_id))
    documents = load_document_registry()

    return CaseDocuments(
        case_id=case.case_id,
        title=case.title,
        procedure_documents=[
            DocumentRecord(**find_document_or_404(documents, stored_filename)[1])
            for stored_filename in case.procedure_stored_filenames
        ],
        record_documents=[
            DocumentRecord(**find_document_or_404(documents, stored_filename)[1])
            for stored_filename in case.record_stored_filenames
        ],
        reference_documents=[
            DocumentRecord(**find_document_or_404(documents, stored_filename)[1])
            for stored_filename in case.reference_stored_filenames
        ],
    )


def get_parsed_case_by_id(case_id: str) -> ParsedCase:
    case = CaseRecord(**get_case_or_404(case_id))
    documents = load_document_registry()

    return ParsedCase(
        case_id=case.case_id,
        title=case.title,
        procedure_documents=_parse_documents(documents, case.procedure_stored_filenames),
        record_documents=_parse_documents(documents, case.record_stored_filenames),
        reference_documents=_parse_documents(documents, case.reference_stored_filenames),
    )


def get_case_compliance_payload(case_id: str) -> dict[str, Any]:
    case = CaseRecord(**get_case_or_404(case_id))
    documents = load_document_registry()

    procedure_documents: list[dict[str, Any]] = []
    record_documents: list[dict[str, Any]] = []
    payload_cache: dict[str, dict[str, Any]] = {}

    for stored_filename in {
        *case.procedure_stored_filenames,
        *case.record_stored_filenames,
        *case.reference_stored_filenames,
    }:
        index, item = find_document_or_404(documents, stored_filename)
        document = DocumentRecord(**item)

        try:
            get_or_parse_document(documents, index, document)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        document = DocumentRecord(**documents[index])
        parsed_payload = _load_parsed_json_file(document)
        payload_cache[stored_filename] = {
            "document_type": document.document_type.value if document.document_type else None,
            "source_filename": document.source_filename,
            "stored_filename": document.stored_filename,
            "group_id": document.group_id,
            "language": document.language.value if document.language else None,
            "content_hash": document.content_hash,
            "parsed_json": parsed_payload,
        }

    for stored_filename in case.procedure_stored_filenames:
        procedure_documents.append(payload_cache[stored_filename])

    for stored_filename in case.record_stored_filenames:
        record_documents.append(payload_cache[stored_filename])

    return {
        "case_id": case.case_id,
        "title": case.title,
        "notes": case.notes,
        "procedures": procedure_documents,
        "records": record_documents,
        "references": [
            payload_cache[stored_filename]
            for stored_filename in case.reference_stored_filenames
        ],
    }


def list_case_deliverables(case_id: str) -> list[DeliverableExtractionSummary]:
    get_case_or_404(case_id)
    return _load_deliverable_summaries(
        directory=get_case_compliance_dir(case_id),
        pattern=f"case_{case_id}_deliverables_*.json",
        fallback_case_id=case_id,
    )


def get_case_deliverable_result(case_id: str, file_name: str) -> DeliverableExtractionResponse:
    get_case_or_404(case_id)

    path = get_case_compliance_dir(case_id) / file_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Deliverable extraction result not found")
    if not path.name.startswith(f"case_{case_id}_deliverables_"):
        raise HTTPException(status_code=404, detail="Deliverable extraction result not found for this case")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Deliverable extraction file corrupted") from exc

    payload.setdefault("created_at", _timestamp_from_path(path))
    payload.setdefault("saved_at", path.as_posix())
    return DeliverableExtractionResponse(**payload)


def delete_case_deliverable_result(case_id: str, file_name: str) -> DeliverableExtractionSummary:
    get_case_or_404(case_id)

    path = get_case_compliance_dir(case_id) / file_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Deliverable extraction result not found")
    if not path.name.startswith(f"case_{case_id}_deliverables_"):
        raise HTTPException(status_code=404, detail="Deliverable extraction result not found for this case")

    summary = _load_deliverable_summary(path, fallback_case_id=case_id)
    if summary is None:
        raise HTTPException(status_code=500, detail="Deliverable extraction file corrupted")

    path.unlink()
    return summary


def get_case_deliverables_payload(case_id: str, file_name: str | None = None) -> dict[str, Any] | None:
    summaries = list_case_deliverables(case_id)
    if file_name:
        result = get_case_deliverable_result(case_id, file_name)
        return result.model_dump()
    if not summaries:
        return None
    latest = summaries[0]
    result = get_case_deliverable_result(case_id, latest.file_name)
    return result.model_dump()


def get_case_procedure_deliverables_payload(case_id: str) -> dict[str, Any] | None:
    case = CaseRecord(**get_case_or_404(case_id))
    if not case.procedure_stored_filenames:
        return None

    combined_deliverables: list[dict[str, Any]] = []
    extraction_provider: str | None = None
    extraction_model: str | None = None

    for stored_filename in case.procedure_stored_filenames:
        try:
            result = get_latest_document_deliverable_result(stored_filename)
        except HTTPException as exc:
            if exc.status_code == 404:
                return None
            raise
        if extraction_provider is None:
            extraction_provider = result.extraction_provider
        if extraction_model is None:
            extraction_model = result.extraction_model
        combined_deliverables.extend(
            item.model_dump()
            for item in result.deliverables
        )

    return {
        "case_id": case.case_id,
        "extraction_provider": extraction_provider or "unknown",
        "extraction_model": extraction_model or "unknown",
        "deliverables": combined_deliverables,
    }


def ensure_case_procedure_deliverables_payload(
    case_id: str,
    *,
    provider: str,
    model: str,
) -> dict[str, Any] | None:
    case = CaseRecord(**get_case_or_404(case_id))
    if not case.procedure_stored_filenames:
        return None

    combined_deliverables: list[dict[str, Any]] = []
    request = DeliverableExtractionRequest(
        provider=provider,
        model=model,
    )

    for stored_filename in case.procedure_stored_filenames:
        document_payload = get_document_extraction_payload(stored_filename)
        result = run_document_deliverable_extraction(
            stored_filename=stored_filename,
            document_payload=document_payload,
            request=request,
        )
        combined_deliverables.extend(
            item.model_dump()
            for item in result.deliverables
        )

    return {
        "case_id": case.case_id,
        "provider": provider,
        "model": model,
        "deliverables": combined_deliverables,
    }


def list_case_compliances(case_id: str) -> list[ComplianceSummary]:
    get_case_or_404(case_id)
    return _load_compliance_summaries(
        paths=sorted(get_case_compliance_dir(case_id).glob(f"case_{case_id}_compliance_*.json"), reverse=True),
        fallback_case_id=case_id,
    )


def list_all_compliances() -> list[ComplianceSummary]:
    return _load_compliance_summaries(
        paths=sorted(CASES_DIR.glob("*/compliance/case_*_compliance_*.json"), reverse=True),
        fallback_case_id="",
    )


def get_case_compliance_result(case_id: str, file_name: str) -> ComplianceResponse:
    get_case_or_404(case_id)

    path = get_case_compliance_dir(case_id) / file_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Compliance result not found")
    if not path.name.startswith(f"case_{case_id}_compliance_"):
        raise HTTPException(status_code=404, detail="Compliance result not found for this case")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Compliance result file corrupted") from exc

    payload.setdefault("created_at", _timestamp_from_path(path))
    payload.setdefault("saved_at", path.as_posix())
    payload["analysis"] = _normalize_compliance_analysis_payload(payload.get("analysis", {}))

    return ComplianceResponse(**payload)


def delete_case_compliance_result(case_id: str, file_name: str) -> ComplianceSummary:
    get_case_or_404(case_id)

    path = get_case_compliance_dir(case_id) / file_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Compliance result not found")
    if not path.name.startswith(f"case_{case_id}_compliance_"):
        raise HTTPException(status_code=404, detail="Compliance result not found for this case")

    summary = _load_compliance_summary(path, fallback_case_id=case_id)
    if summary is None:
        raise HTTPException(status_code=500, detail="Compliance result file corrupted")

    path.unlink()
    return summary


def _load_deliverable_summaries(*, directory: Path, pattern: str, fallback_case_id: str) -> list[DeliverableExtractionSummary]:
    summaries: list[DeliverableExtractionSummary] = []
    for path in sorted(directory.glob(pattern), reverse=True):
        summary = _load_deliverable_summary(path, fallback_case_id=fallback_case_id)
        if summary is not None:
            summaries.append(summary)
    return summaries


def _load_deliverable_summary(
    path: Path,
    *,
    fallback_case_id: str,
) -> DeliverableExtractionSummary | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    return DeliverableExtractionSummary(
        case_id=payload.get("case_id", fallback_case_id),
        file_name=path.name,
        created_at=payload.get("created_at", _timestamp_from_path(path)),
        saved_at=payload.get("saved_at", path.as_posix()),
        extraction_provider=payload.get("extraction_provider", payload.get("provider", "unknown")),
        extraction_model=payload.get("extraction_model", payload.get("model", "unknown")),
        deliverable_count=len(payload.get("deliverables", [])),
    )


def _load_compliance_summaries(*, paths: list[Path], fallback_case_id: str) -> list[ComplianceSummary]:
    summaries: list[ComplianceSummary] = []
    for path in paths:
        summary = _load_compliance_summary(path, fallback_case_id=fallback_case_id)
        if summary is not None:
            summaries.append(summary)
    return summaries


def _load_compliance_summary(path: Path, *, fallback_case_id: str) -> ComplianceSummary | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    analysis_payload = _normalize_compliance_analysis_payload(payload.get("analysis", {}))
    status_counts = _compute_compliance_status_counts(analysis_payload)
    scores_payload = payload.get("scores", {}) if isinstance(payload.get("scores", {}), dict) else {}

    return ComplianceSummary(
        case_id=payload.get("case_id", fallback_case_id),
        file_name=path.name,
        created_at=payload.get("created_at", _timestamp_from_path(path)),
        saved_at=payload.get("saved_at", path.as_posix()),
        provider=payload.get("compliance_provider", payload.get("provider", "unknown")),
        model=payload.get("compliance_model", payload.get("model", "unknown")),
        method=payload.get("method", "non_rag"),
        overall_assessment=analysis_payload.get("overall_assessment", "unknown"),
        completion_percent=analysis_payload.get("completion_percent", 0),
        satisfied_count=status_counts["satisfied"],
        partial_count=status_counts["partial"],
        not_satisfied_count=status_counts["not_satisfied"],
        m3_evidence_weighted_score=scores_payload.get("m3_evidence_weighted_score", 0.0),
        m5_grounding_score=scores_payload.get("m5_grounding_score", 0.0),
        reference_stored_filenames=payload.get("reference_stored_filenames", []),
    )


def _timestamp_from_path(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _normalize_compliance_analysis_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"overall_assessment": "Completed_0_20", "completion_percent": 0}

    normalized = dict(payload)
    findings_payload = normalized.get("procedure_to_record") or normalized.get("findings") or []
    findings = [
        ComplianceFinding(**item)
        for item in findings_payload
        if isinstance(item, dict)
    ]
    normalized["completion_percent"] = compute_completion_percent(findings)
    normalized["overall_assessment"] = compute_overall_assessment_from_findings(findings)
    return normalized


def _compute_compliance_status_counts(payload: dict[str, Any]) -> dict[str, int]:
    findings_payload = payload.get("procedure_to_record") or payload.get("findings") or []
    findings = [
        ComplianceFinding(**item)
        for item in findings_payload
        if isinstance(item, dict)
    ]
    counts = {"satisfied": 0, "partial": 0, "not_satisfied": 0}
    for finding in findings:
        counts[finding.status] += 1
    return counts


def _parse_documents(
    registry: list[dict],
    stored_filenames: list[str],
) -> list[ParsedDocument]:
    parsed_documents: list[ParsedDocument] = []
    for stored_filename in stored_filenames:
        index, item = find_document_or_404(registry, stored_filename)
        document = DocumentRecord(**item)
        try:
            parsed_documents.append(get_or_parse_document(registry, index, document))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return parsed_documents


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
