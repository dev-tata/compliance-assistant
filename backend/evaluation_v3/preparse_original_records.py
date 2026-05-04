from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

EVALUATION_V3_DIR = BACKEND_DIR / "evaluation_v3"
CACHE_ORIGINAL_DIR = EVALUATION_V3_DIR / "cache_original"
OUTPUT_DIR = EVALUATION_V3_DIR / "output"
PARSED_DIR = CACHE_ORIGINAL_DIR / "parsed"
RECORD_INDEXES_DIR = CACHE_ORIGINAL_DIR / "retrieval" / "records"
DATASET_ROOT_DEFAULT = BACKEND_DIR / "original_data"
CATALOG_PATH_DEFAULT = CACHE_ORIGINAL_DIR / "original_record_registry.csv"

PARSED_DIR.mkdir(parents=True, exist_ok=True)
RECORD_INDEXES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from app.schemas.documents import DocumentLanguage, DocumentRecord, DocumentType
from app.services.document_service import compute_file_hash, current_timestamp, validate_extension
from app.services.parsing.parser_service import parse_document
import app.services.retrieval.record_index_service as record_index_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-parse original risk assessment records and build retrieval indexes for evaluation_v3."
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--catalog-path", type=Path, default=CATALOG_PATH_DEFAULT)
    parser.add_argument("--build-indexes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-name", default="original_retrieval_prep")
    return parser.parse_args()


def _slugify(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-") or "unknown"


def _build_sanitized_record_filename(*, doc_id: str, original_filename: str) -> str:
    suffix = Path(original_filename).suffix.lower() or ".docx"
    normalized_doc_id = _slugify(doc_id).replace("-", "_").upper() or uuid4().hex[:8].upper()
    return f"{normalized_doc_id}{suffix}"


def _load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_catalog(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        doc_id = str(row.get("doc_id") or "").strip()
        if doc_id:
            indexed[doc_id] = row
    return indexed


def _append_catalog_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _append_jsonl_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_run_output_dir(base_dir: Path, *, run_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = base_dir / f"{timestamp}__{_slugify(run_name)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _register_persistent_record(path: Path, *, sanitized_source_filename: str) -> DocumentRecord:
    validate_extension(path.name)
    content_hash = compute_file_hash(path)
    stored_filename = f"{uuid4()}_{sanitized_source_filename}"
    record = {
        "source_filename": sanitized_source_filename,
        "created_at": current_timestamp(),
        "stored_filename": stored_filename,
        "stored_at": path.as_posix(),
        "document_type": DocumentType.risk_assessment.value,
        "language": DocumentLanguage.en.value,
        "group_id": None,
        "parsed_json_at": None,
        "content_hash": content_hash,
        "frozen": False,
    }
    return DocumentRecord(**record)


def _resolve_or_register_record(
    *,
    manifest_row: dict[str, str],
    dataset_root: Path,
    catalog_by_doc_id: dict[str, dict[str, str]],
) -> DocumentRecord:
    doc_id = str(manifest_row.get("doc_id") or "").strip()
    filename = str(manifest_row.get("filename") or "").strip()
    catalog_row = catalog_by_doc_id.get(doc_id)
    if catalog_row:
        parsed_json_at = str(catalog_row.get("parsed_json_at") or "").strip()
        if parsed_json_at and Path(parsed_json_at).exists():
            return DocumentRecord(
                source_filename=str(catalog_row.get("source_filename") or ""),
                created_at=catalog_row.get("created_at"),
                stored_filename=str(catalog_row.get("stored_filename") or ""),
                stored_at=str(catalog_row.get("stored_at") or ""),
                document_type=DocumentType(
                    str(catalog_row.get("document_type") or DocumentType.risk_assessment.value)
                ),
                language=DocumentLanguage(str(catalog_row.get("language") or DocumentLanguage.en.value)),
                group_id=None,
                parsed_json_at=parsed_json_at,
                content_hash=str(catalog_row.get("content_hash") or ""),
                frozen=False,
            )

    path = dataset_root / filename
    if not path.exists():
        raise FileNotFoundError(f"Original record not found: {path}")

    sanitized_source_filename = _build_sanitized_record_filename(
        doc_id=doc_id,
        original_filename=filename,
    )
    return _register_persistent_record(
        path,
        sanitized_source_filename=sanitized_source_filename,
    )


def _ensure_record_parsed_and_indexed(
    record_document: DocumentRecord,
    *,
    build_indexes: bool,
) -> tuple[DocumentRecord, bool]:
    parsed_path = PARSED_DIR / f"{record_document.content_hash}.json"
    if parsed_path.exists():
        parsed_json = json.loads(parsed_path.read_text(encoding="utf-8"))
    else:
        parsed_document = parse_document(record_document)
        parsed_json = parsed_document.model_dump(mode="json")
        parsed_path.write_text(
            json.dumps(parsed_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    updated_document = record_document.model_copy(update={"parsed_json_at": parsed_path.as_posix()})

    index_ready = False
    if build_indexes:
        record_index_service.RECORD_INDEXES_DIR = RECORD_INDEXES_DIR
        record_index_service.ensure_record_index(
            {
                "source_filename": updated_document.source_filename,
                "stored_filename": updated_document.stored_filename,
                "content_hash": updated_document.content_hash,
                "document_type": updated_document.document_type.value if updated_document.document_type else None,
                "parsed_json": parsed_json,
            }
        )
        index_ready = True
    return updated_document, index_ready


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or (args.dataset_root / "manifest.csv")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    output_dir = _build_run_output_dir(OUTPUT_DIR, run_name=args.run_name)
    failures_path = output_dir / "preparse_failures.jsonl"
    summary_path = output_dir / "summary.json"
    run_config_path = output_dir / "run_config.json"

    manifest_rows = _load_manifest_rows(manifest_path)
    catalog_by_doc_id = _load_catalog(args.catalog_path)

    prepared_count = 0
    skipped_count = 0
    failure_count = 0

    for row in manifest_rows:
        doc_id = str(row.get("doc_id") or "").strip()
        filename = str(row.get("filename") or "").strip()
        if doc_id in catalog_by_doc_id:
            skipped_count += 1
            continue

        print(f"Preparing {filename}", flush=True)
        try:
            document = _resolve_or_register_record(
                manifest_row=row,
                dataset_root=args.dataset_root,
                catalog_by_doc_id=catalog_by_doc_id,
            )
            document, index_ready = _ensure_record_parsed_and_indexed(
                document,
                build_indexes=args.build_indexes,
            )
            catalog_row = {
                "doc_id": doc_id,
                "original_filename": filename,
                "stored_filename": document.stored_filename,
                "source_filename": document.source_filename,
                "stored_at": document.stored_at,
                "created_at": document.created_at or "",
                "content_hash": document.content_hash or "",
                "parsed_json_at": document.parsed_json_at or "",
                "document_type": document.document_type.value if document.document_type else "",
                "language": document.language.value if document.language else "",
                "index_ready": "true" if index_ready else "false",
            }
            _append_catalog_row(args.catalog_path, catalog_row)
            catalog_by_doc_id[doc_id] = catalog_row
            prepared_count += 1
        except Exception as exc:
            failure_count += 1
            _append_jsonl_row(
                failures_path,
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            print(f"Failed {filename}: {type(exc).__name__}: {exc}", flush=True)
            break

    run_config_path.write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "manifest_path": str(manifest_path),
                "catalog_path": str(args.catalog_path),
                "build_indexes": args.build_indexes,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "prepared_count": prepared_count,
                "skipped_count": skipped_count,
                "failure_count": failure_count,
                "catalog_path": str(args.catalog_path),
                "parsed_dir": str(PARSED_DIR),
                "record_indexes_dir": str(RECORD_INDEXES_DIR),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved original record catalog to {args.catalog_path}", flush=True)
    print(f"Saved run output to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
