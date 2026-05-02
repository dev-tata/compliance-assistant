from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import HTTPException

from app.schemas.compliance import ComplianceRequest
from app.schemas.deliverables import DeliverableExtractionResponse
from app.schemas.documents import DocumentLanguage, DocumentRecord, DocumentType, is_record_document_type
from app.services.compliance_service import run_case_compliance_analysis
from app.services.document_service import (
    UPLOAD_DIR,
    compute_file_hash,
    current_timestamp,
    extract_procedure_group_id,
    find_document_by_content_hash,
    find_document_or_404,
    get_latest_document_deliverable_result,
    get_or_parse_document,
    load_document_registry,
    remove_document_files,
    save_document_registry,
    validate_extension,
)
from app.services.evaluation_v3_service import EVALUATION_V3_RUNTIME_DIR
from app.services.retrieval.record_index_service import ensure_record_index
from app.services.retrieval.reference_index_service import get_reference_index_dir, load_reference_chunks
from app.services.storage_paths import get_case_dir

OUTPUT_ROOT_DEFAULT = BACKEND_DIR / "evaluation_runs"
CACHE_CATALOG_DEFAULT = BACKEND_DIR / "evaluation_v3" / "cache" / "synthetic_record_registry.csv"


@dataclass
class RecordInput:
    path: Path
    record_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the existing compliance pipeline sequentially for multiple record files."
    )
    parser.add_argument("--record-path", type=Path, action="append", default=[])
    parser.add_argument("--records-catalog", type=Path, default=CACHE_CATALOG_DEFAULT)
    parser.add_argument("--procedure-path", type=Path, required=True)
    parser.add_argument("--reference-path", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--extraction-provider", default=None)
    parser.add_argument("--extraction-model", default=None)
    parser.add_argument("--procedure-language", choices=("en", "sv", "mixed"), default="en")
    parser.add_argument("--reference-language", choices=("en", "sv", "mixed"), default="en")
    parser.add_argument("--record-language", choices=("en", "sv", "mixed"), default="en")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    return parser.parse_args()


def _slugify(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-") or "unknown"


def _build_run_id(run_id: str | None) -> str:
    if run_id and run_id.strip():
        return _slugify(run_id)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}__batch-compliance"


def _normalize_record_id(path: Path, seen: dict[str, int]) -> str:
    base = _slugify(path.stem)
    count = seen.get(base, 0) + 1
    seen[base] = count
    if count == 1:
        return base
    return f"{base}-{count}"


def _normalize_record_id_from_text(value: str, seen: dict[str, int]) -> str:
    base = _slugify(value)
    count = seen.get(base, 0) + 1
    seen[base] = count
    if count == 1:
        return base
    return f"{base}-{count}"


def _batch_record_id(index: int) -> str:
    return f"record_{index:03d}"


def _build_neutral_filename(*, record_id: str, original_path: Path) -> str:
    suffix = original_path.suffix.lower() or ".docx"
    return f"{record_id}{suffix}"


def _coerce_language(value: str) -> DocumentLanguage:
    return DocumentLanguage(value)


def _ensure_evaluation_v3_enabled() -> None:
    os.environ["ENABLE_EVALUATION_V3"] = "1"


def _register_or_reuse_document(
    path: Path,
    *,
    document_type: DocumentType,
    language: DocumentLanguage,
    source_filename_override: str | None = None,
    allow_registry_reuse: bool = True,
) -> tuple[DocumentRecord, bool]:
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    validate_extension(path.name)

    registry = load_document_registry()
    content_hash = compute_file_hash(path)
    existing_item = find_document_by_content_hash(registry, content_hash)
    if allow_registry_reuse and existing_item and existing_item.get("document_type") == document_type.value:
        index, item = find_document_or_404(registry, str(existing_item.get("stored_filename")))
        document = DocumentRecord(**item)
        parsed_document = get_or_parse_document(registry, index, document)
        _ensure_indexes_for_document(document=document, parsed_json=parsed_document.model_dump(mode="json"))
        return DocumentRecord(**registry[index]), False

    source_filename = source_filename_override or path.name
    stored_filename = f"{uuid4()}_{source_filename}"
    copied_path = UPLOAD_DIR / stored_filename
    shutil.copy2(path, copied_path)

    canonical_path = copied_path
    if existing_item:
        canonical_path = Path(str(existing_item["stored_at"]))
        if copied_path.exists():
            copied_path.unlink()

    group_id = None
    if document_type == DocumentType.procedure:
        group_id = extract_procedure_group_id(path.name)

    record = {
        "source_filename": source_filename,
        "created_at": current_timestamp(),
        "stored_filename": stored_filename,
        "stored_at": canonical_path.as_posix(),
        "document_type": document_type.value,
        "language": language.value,
        "group_id": group_id,
        "parsed_json_at": None,
        "content_hash": content_hash,
        "frozen": False,
    }
    registry.append(record)
    save_document_registry(registry)

    try:
        document = DocumentRecord(**record)
        parsed_document = get_or_parse_document(registry, len(registry) - 1, document)
        _ensure_indexes_for_document(document=document, parsed_json=parsed_document.model_dump(mode="json"))
        return DocumentRecord(**registry[-1]), True
    except Exception:
        registry = load_document_registry()
        registry = [item for item in registry if item.get("stored_filename") != stored_filename]
        save_document_registry(registry)
        if copied_path.exists():
            copied_path.unlink()
        raise


def _ensure_indexes_for_document(*, document: DocumentRecord, parsed_json: dict[str, Any]) -> None:
    if is_record_document_type(document.document_type):
        ensure_record_index(
            {
                "source_filename": document.source_filename,
                "stored_filename": document.stored_filename,
                "content_hash": document.content_hash,
                "document_type": document.document_type.value if document.document_type else None,
                "parsed_json": parsed_json,
            }
        )


def _require_existing_reference_index(document: DocumentRecord) -> None:
    payload = {
        "stored_filename": document.stored_filename,
        "content_hash": document.content_hash,
    }
    index_dir = get_reference_index_dir(payload)
    meta_path = index_dir / "meta.json"
    faiss_path = index_dir / "index.faiss"
    chunks = load_reference_chunks(payload)
    if index_dir.exists() and meta_path.exists() and faiss_path.exists() and chunks:
        return
    raise RuntimeError(
        "Missing cached reference index for the selected reference document. "
        "Build/select the reference index manually before running the batch."
    )


def _build_document_payload(registry: list[dict[str, Any]], stored_filename: str) -> dict[str, Any]:
    index, item = find_document_or_404(registry, stored_filename)
    document = DocumentRecord(**item)
    parsed = get_or_parse_document(registry, index, document)
    document = DocumentRecord(**registry[index])
    return {
        "document_type": document.document_type.value if document.document_type else None,
        "source_filename": document.source_filename,
        "stored_filename": document.stored_filename,
        "group_id": document.group_id,
        "language": document.language.value if document.language else None,
        "content_hash": document.content_hash,
        "parsed_json": parsed.model_dump(mode="json"),
    }


def _ensure_procedure_deliverables(
    *,
    procedure_stored_filename: str,
) -> DeliverableExtractionResponse:
    try:
        return get_latest_document_deliverable_result(procedure_stored_filename)
    except HTTPException as exc:
        if exc.status_code == 404:
            raise RuntimeError(
                "No saved deliverable extraction exists for the selected procedure. "
                "Create/select procedure deliverables manually before running the batch."
            ) from exc
        raise


def _build_case_payload(
    *,
    case_id: str,
    procedure_stored_filename: str,
    record_stored_filename: str,
    reference_stored_filename: str,
    deliverable_response: DeliverableExtractionResponse,
) -> dict[str, Any]:
    registry = load_document_registry()
    return {
        "case_id": case_id,
        "title": "batch-evaluation",
        "notes": "sequential batch compliance run",
        "procedures": [_build_document_payload(registry, procedure_stored_filename)],
        "records": [_build_document_payload(registry, record_stored_filename)],
        "references": [_build_document_payload(registry, reference_stored_filename)],
        "deliverables": [item.model_dump() for item in deliverable_response.deliverables],
        "extraction_provider": deliverable_response.extraction_provider,
        "extraction_model": deliverable_response.extraction_model,
    }


def _find_evaluation_v3_run_dir(*, compliance_saved_at: str) -> Path:
    if not EVALUATION_V3_RUNTIME_DIR.exists():
        raise FileNotFoundError("evaluation_v3 runtime directory does not exist.")

    matches: list[Path] = []
    for run_dir in EVALUATION_V3_RUNTIME_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        for candidate in run_dir.glob("*.json"):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if payload.get("source_compliance_saved_at") == compliance_saved_at:
                matches.append(run_dir)
                break

    if not matches:
        raise FileNotFoundError(
            f"No evaluation_v3 run matched compliance output: {compliance_saved_at}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def _copy_case_outputs(*, compliance_saved_at: str, run_dir: Path, target_dir: Path) -> tuple[Path, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    compliance_result_path = BACKEND_DIR / compliance_saved_at
    result_path = run_dir / "evaluation_v3_result.json"
    summary_path = run_dir / "evaluation_v3_summary.json"
    debug_json_path = next(run_dir.glob("*_debug.json"), None)
    debug_csv_path = next(run_dir.glob("*_debug.csv"), None)
    if not result_path.exists():
        raise FileNotFoundError(f"Missing evaluation_v3_result.json in {run_dir}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing evaluation_v3_summary.json in {run_dir}")
    if not compliance_result_path.exists():
        raise FileNotFoundError(f"Missing compliance result JSON: {compliance_result_path}")

    copied_compliance_path = target_dir / "compliance_result.json"
    copied_result_path = target_dir / "evaluation_v3_result.json"
    copied_summary_path = target_dir / "evaluation_v3_summary.json"
    shutil.copy2(compliance_result_path, copied_compliance_path)
    shutil.copy2(result_path, copied_result_path)
    shutil.copy2(summary_path, copied_summary_path)
    if debug_json_path and debug_json_path.exists():
        shutil.copy2(debug_json_path, target_dir / "evaluation_v3_debug.json")
    if debug_csv_path and debug_csv_path.exists():
        shutil.copy2(debug_csv_path, target_dir / "evaluation_v3_debug.csv")
    return copied_result_path, copied_summary_path


def _cleanup_registered_documents(stored_filenames: list[str]) -> None:
    if not stored_filenames:
        return
    registry = load_document_registry()
    for stored_filename in stored_filenames:
        for index, item in enumerate(list(registry)):
            if item.get("stored_filename") != stored_filename:
                continue
            removed = registry.pop(index)
            document = DocumentRecord(**removed)
            remove_document_files(document, registry)
            break
    save_document_registry(registry)


def _write_aggregate_summary(
    *,
    run_output_dir: Path,
    summary_paths: list[Path],
) -> Path:
    total_satisfied = 0
    total_partial = 0
    total_not_satisfied = 0
    coverage_values: list[float] = []
    grounded_values: list[float] = []

    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        total_satisfied += int(payload.get("satisfied") or 0)
        total_partial += int(payload.get("partial") or 0)
        total_not_satisfied += int(payload.get("not_satisfied") or 0)
        coverage_values.append(float(payload.get("avg_evidence_coverage") or 0.0))
        grounded_values.append(float(payload.get("avg_grounded_evidence") or 0.0))

    aggregate_payload = {
        "total_cases": len(summary_paths),
        "total_satisfied": total_satisfied,
        "total_partial": total_partial,
        "total_not_satisfied": total_not_satisfied,
        "avg_coverage": round(sum(coverage_values) / len(coverage_values), 4) if coverage_values else 0.0,
        "avg_grounded_evidence": (
            round(sum(grounded_values) / len(grounded_values), 4) if grounded_values else 0.0
        ),
    }
    aggregate_path = run_output_dir / "aggregate_summary.json"
    aggregate_path.write_text(
        json.dumps(aggregate_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return aggregate_path


def _write_error_payload(*, target_dir: Path, record_id: str, error: Exception) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    error_path = target_dir / "error.json"
    error_path.write_text(
        json.dumps(
            {
                "record_id": record_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return error_path


def _load_record_inputs_from_catalog(
    *,
    catalog_path: Path,
    limit: int | None,
) -> list[RecordInput]:
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Records catalog not found: {catalog_path}. Pass --record-path explicitly or provide a valid --records-catalog."
        )

    with catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    inputs: list[RecordInput] = []
    for index, row in enumerate(rows, start=1):
        stored_at = Path(str(row.get("stored_at") or "").strip())
        if not stored_at.exists():
            continue
        inputs.append(
            RecordInput(
                path=stored_at,
                record_id=_batch_record_id(len(inputs) + 1),
            )
        )
        if limit is not None and len(inputs) >= limit:
            break
    if not inputs:
        raise RuntimeError(f"No usable record paths found in catalog: {catalog_path}")
    return inputs


def _resolve_record_inputs(
    *,
    record_paths: list[Path],
    records_catalog: Path,
    limit: int | None,
) -> list[RecordInput]:
    if record_paths:
        selected_paths = record_paths[:limit] if limit is not None else record_paths
        return [
            RecordInput(
                path=record_path,
                record_id=_batch_record_id(index),
            )
            for index, record_path in enumerate(selected_paths, start=1)
        ]
    return _load_record_inputs_from_catalog(
        catalog_path=records_catalog,
        limit=limit,
    )


def main() -> None:
    args = parse_args()
    _ensure_evaluation_v3_enabled()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0.")

    run_id = _build_run_id(args.run_id)
    run_output_dir = args.output_root / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    record_inputs = _resolve_record_inputs(
        record_paths=args.record_path,
        records_catalog=args.records_catalog,
        limit=args.limit,
    )

    procedure_document, created_procedure = _register_or_reuse_document(
        args.procedure_path,
        document_type=DocumentType.procedure,
        language=_coerce_language(args.procedure_language),
    )
    reference_document, created_reference = _register_or_reuse_document(
        args.reference_path,
        document_type=DocumentType.reference,
        language=_coerce_language(args.reference_language),
    )
    _require_existing_reference_index(reference_document)
    deliverable_response = _ensure_procedure_deliverables(
        procedure_stored_filename=procedure_document.stored_filename,
    )

    created_shared_documents: list[str] = []
    if created_procedure:
        created_shared_documents.append(procedure_document.stored_filename)
    if created_reference:
        created_shared_documents.append(reference_document.stored_filename)

    copied_summary_paths: list[Path] = []

    try:
        for index, record_input in enumerate(record_inputs, start=1):
            target_dir = run_output_dir / record_input.record_id
            case_id = str(uuid4())
            created_record_documents: list[str] = []

            print(
                f"[{index}/{len(record_inputs)}] Running compliance for {record_input.record_id}",
                flush=True,
            )

            try:
                record_document, created_record = _register_or_reuse_document(
                    record_input.path,
                    document_type=DocumentType.risk_assessment,
                    language=_coerce_language(args.record_language),
                    source_filename_override=_build_neutral_filename(
                        record_id=record_input.record_id,
                        original_path=record_input.path,
                    ),
                    allow_registry_reuse=False,
                )
                if created_record:
                    created_record_documents.append(record_document.stored_filename)

                case_payload = _build_case_payload(
                    case_id=case_id,
                    procedure_stored_filename=procedure_document.stored_filename,
                    record_stored_filename=record_document.stored_filename,
                    reference_stored_filename=reference_document.stored_filename,
                    deliverable_response=deliverable_response,
                )
                print(
                    f"[{index}/{len(record_inputs)}] LLM start for {record_input.record_id}",
                    flush=True,
                )
                response = run_case_compliance_analysis(
                    case_id=case_id,
                    case_payload=case_payload,
                    request=ComplianceRequest(
                        provider=args.provider,
                        model=args.model,
                        method="two_stage_rag",
                    ),
                )
                print(
                    f"[{index}/{len(record_inputs)}] LLM success for {record_input.record_id}",
                    flush=True,
                )
                evaluation_run_dir = _find_evaluation_v3_run_dir(
                    compliance_saved_at=response.saved_at,
                )
                _, copied_summary_path = _copy_case_outputs(
                    compliance_saved_at=response.saved_at,
                    run_dir=evaluation_run_dir,
                    target_dir=target_dir,
                )
                copied_summary_paths.append(copied_summary_path)
                print(f"[{index}/{len(record_inputs)}] Saved outputs to {target_dir}", flush=True)
            except Exception as exc:
                error_path = _write_error_payload(
                    target_dir=target_dir,
                    record_id=record_input.record_id,
                    error=exc,
                )
                print(
                    f"[{index}/{len(record_inputs)}] Failed {record_input.record_id}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                print(f"[{index}/{len(record_inputs)}] Saved error details to {error_path}", flush=True)
            finally:
                case_dir = get_case_dir(case_id)
                if case_dir.exists():
                    shutil.rmtree(case_dir, ignore_errors=True)
                _cleanup_registered_documents(created_record_documents)

        aggregate_path = _write_aggregate_summary(
            run_output_dir=run_output_dir,
            summary_paths=copied_summary_paths,
        )
        print(f"Saved aggregate summary to {aggregate_path}", flush=True)
    finally:
        _cleanup_registered_documents(created_shared_documents)


if __name__ == "__main__":
    main()
