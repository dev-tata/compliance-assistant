from __future__ import annotations

import json
import logging
import shutil
import sys
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.schemas.compliance import ComplianceResponse, ComplianceStageResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.append(str(BACKEND_ROOT))

from evaluation_v3 import (  # noqa: E402
    build_edge_case_debug_rows,
    build_debug_report_rows,
    build_debug_report_summary,
    build_compact_summary,
    build_evaluation_v3_result,
    build_evaluation_v3_result_rows,
    build_evaluation_v3_summary,
    build_suspicious_debug_rows,
    build_evaluation_unit,
    calculate_aggregate_metrics,
    evaluation_v3_config,
)
from evaluation_v3.builder import CONFLICT_MARKERS  # noqa: E402

logger = logging.getLogger(__name__)

EVALUATION_V3_RUNTIME_DIR = BACKEND_ROOT / "evaluation_v3" / "runs"
EVALUATION_V3_MANUAL_COMPARISONS_DIR = BACKEND_ROOT / "evaluation_v3" / "manual_comparisons"
CONTRADICTION_PRIORITY = (
    "direct_conflict",
    "reference_conflict",
    "reference_clarification",
    "missing_evidence",
    "none",
)
MAX_MERGED_RATIONALE_LENGTH = 1200


def write_evaluation_v3_runtime_output(
    *,
    case_id: str,
    compliance_response: ComplianceResponse,
    deliverables: list[dict[str, Any]],
    retrieved_payload: list[dict[str, Any]],
) -> Path:
    _ensure_evaluation_v3_runtime_dir()
    _log_retrieval_score_debug(retrieved_payload)
    run_dir = build_evaluation_v3_runtime_dir(case_id=case_id)
    timestamp = _current_timestamp()
    units = build_evaluation_v3_units(
        deliverables=deliverables,
        compliance_response=compliance_response,
        retrieved_payload=retrieved_payload,
    )
    aggregate_metrics = calculate_aggregate_metrics(units)
    saved_path = build_evaluation_v3_runtime_path(run_dir=run_dir, case_id=case_id)
    result_export_path = build_evaluation_v3_result_export_path(run_dir=run_dir)
    summary_export_path = build_evaluation_v3_summary_export_path(run_dir=run_dir)
    comparison_export_path = build_evaluation_v3_comparison_export_path(run_dir=run_dir)
    result_rows = build_evaluation_v3_result_rows(units)
    result_rows_by_id = {
        row.deliverable_id: row
        for row in result_rows
    }
    trace_metadata = _build_evaluation_v3_trace_metadata(
        model_name=compliance_response.compliance_model,
        timestamp=timestamp,
    )
    payload = build_evaluation_v3_result(
        case_id=case_id,
        created_at=timestamp,
        source_compliance_saved_at=compliance_response.saved_at,
        compliance_provider=compliance_response.compliance_provider,
        compliance_model=compliance_response.compliance_model,
        method=compliance_response.method,
        units=units,
        aggregate_metrics=aggregate_metrics,
    ).model_dump(exclude_none=True)
    if "config" in trace_metadata:
        payload["config"] = trace_metadata["config"]
    if "diagnostic_config" in trace_metadata:
        payload["diagnostic_config"] = trace_metadata["diagnostic_config"]
    payload["model_name"] = trace_metadata["model_name"]
    payload["timestamp"] = trace_metadata["timestamp"]
    saved_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result_export_path.write_text(
        json.dumps(
            {
                **trace_metadata,
                "units": [
                    row.model_dump(exclude_none=True)
                    for row in result_rows
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary_export_path.write_text(
        json.dumps(
            {
                **trace_metadata,
                **build_evaluation_v3_summary(result_rows),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    comparison_export_path.write_text(
        json.dumps(
            {
                **trace_metadata,
                "units": [
                    {
                        "deliverable_id": unit.deliverable.deliverable_id,
                        "stage_3_label": unit.stage_3_answer.label,
                        "final_label": result_rows_by_id[unit.deliverable.deliverable_id].final_label,
                        "evidence_status": result_rows_by_id[unit.deliverable.deliverable_id].evidence_status,
                        "grounded_evidence_count": result_rows_by_id[
                            unit.deliverable.deliverable_id
                        ].grounded_evidence_count,
                        "has_conflict": result_rows_by_id[unit.deliverable.deliverable_id].has_conflict,
                        "contradiction_type": result_rows_by_id[
                            unit.deliverable.deliverable_id
                        ].contradiction_type,
                    }
                    for unit in sorted(units, key=lambda item: item.deliverable.deliverable_id)
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_debug_reports(
        case_id=case_id,
        source_saved_path=saved_path,
        compliance_model=compliance_response.compliance_model,
        units=units,
    )
    return saved_path


def build_evaluation_v3_units(
    *,
    deliverables: list[dict[str, Any]],
    compliance_response: ComplianceResponse,
    retrieved_payload: list[dict[str, Any]],
) -> list[Any]:
    aligned_inputs = _build_evaluation_v3_input_mapping(
        deliverables=deliverables,
        compliance_response=compliance_response,
        retrieved_payload=retrieved_payload,
    )
    units: list[Any] = []
    for requirement_key in sorted(aligned_inputs):
        aligned = aligned_inputs[requirement_key]
        deliverable = aligned["deliverable"]
        payload = aligned["retrieved_rows"]
        record_chunks = _with_record_ids(
            deliverable_id=requirement_key,
            chunks=payload.get("retrieved_record_sections", []),
        )
        for chunk in record_chunks:
            print(
                {
                    "stage": "evaluation_v3_service.record_chunks",
                    "deliverable_id": requirement_key,
                    "text": str(chunk.get("text") or "")[:80],
                    "retrieval_score": chunk.get("retrieval_score"),
                    "raw_retrieval_score": chunk.get("raw_retrieval_score"),
                }
            )
        reference_chunks = _with_reference_ids(
            deliverable_id=requirement_key,
            chunks=payload.get("retrieved_requirement_context", []),
        )
        stage_1_finding = aligned["findings"].get("stage_1_non_rag")
        stage_1_row = aligned["linked_rows"].get("stage_1_non_rag")
        stage_1_output = _build_stage_output(
            finding=stage_1_finding,
            linked_row=stage_1_row,
        )
        stage_2_finding = aligned["findings"].get("stage_2_record_retrieval")
        stage_2_row = aligned["linked_rows"].get("stage_2_record_retrieval")
        stage_2_output = _build_stage_output(
            finding=stage_2_finding,
            linked_row=stage_2_row,
        )
        stage_3_finding = aligned["findings"].get("stage_3_reference_retrieval")
        stage_3_row = aligned["linked_rows"].get("stage_3_reference_retrieval")
        stage_3_output = _build_stage_output(
            finding=stage_3_finding,
            linked_row=stage_3_row,
            supporting_reference_ids=[
                item.get("reference_id")
                for item in reference_chunks
                if item.get("reference_id")
            ],
        )
        unit = build_evaluation_unit(
            frozen_deliverable={
                "deliverable_id": requirement_key,
                "source_document": deliverable.get("source_document", ""),
                "section_label": deliverable.get("section_label", ""),
                "heading_title": deliverable.get("heading_title", ""),
                "requirement_text": deliverable.get("requirement_text", ""),
                "weight": deliverable.get("weight", 1.0),
                "required_evidence_count": deliverable.get("required_evidence_count"),
            },
            retrieved_record_evidence_chunks=record_chunks,
            retrieved_reference_evidence_chunks=reference_chunks,
            stage_1_output=stage_1_output,
            stage_2_output=stage_2_output,
            stage_3_output=stage_3_output,
        )
        units.append(unit)
    return units


def safely_write_evaluation_v3_runtime_output(
    *,
    case_id: str,
    compliance_response: ComplianceResponse,
    deliverables: list[dict[str, Any]],
    retrieved_payload: list[dict[str, Any]],
) -> Path | None:
    try:
        return write_evaluation_v3_runtime_output(
            case_id=case_id,
            compliance_response=compliance_response,
            deliverables=deliverables,
            retrieved_payload=retrieved_payload,
        )
    except Exception as exc:
        logger.exception("Failed to write evaluation_v3 sidecar output for case %s", case_id)
        _write_evaluation_v3_error_report(
            case_id=case_id,
            compliance_response=compliance_response,
            error=exc,
        )
        return None


def delete_evaluation_v3_runs_for_compliance(*, compliance_saved_at: str) -> int:
    if not EVALUATION_V3_RUNTIME_DIR.exists():
        return 0
    deleted = 0
    for run_dir in EVALUATION_V3_RUNTIME_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        if _run_dir_matches_compliance(run_dir=run_dir, compliance_saved_at=compliance_saved_at):
            shutil.rmtree(run_dir)
            deleted += 1
    return deleted


def build_evaluation_v3_runtime_path(*, run_dir: Path, case_id: str) -> Path:
    return run_dir / f"case_{case_id}_evaluation_v3.json"


def build_evaluation_v3_result_export_path(*, run_dir: Path) -> Path:
    return run_dir / "evaluation_v3_result.json"


def build_evaluation_v3_summary_export_path(*, run_dir: Path) -> Path:
    return run_dir / "evaluation_v3_summary.json"


def build_evaluation_v3_comparison_export_path(*, run_dir: Path) -> Path:
    return run_dir / "evaluation_comparison.json"


def build_evaluation_v3_runtime_dir(*, case_id: str) -> Path:
    _ensure_evaluation_v3_runtime_dir()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = EVALUATION_V3_RUNTIME_DIR / f"case_{case_id}_evaluation_v3_{timestamp}_{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _ensure_evaluation_v3_runtime_dir() -> None:
    EVALUATION_V3_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _build_evaluation_v3_trace_metadata(*, model_name: str, timestamp: str) -> dict[str, Any]:
    trace_metadata: dict[str, Any] = {
        "model_name": model_name,
        "timestamp": timestamp,
    }
    public_config = _build_evaluation_v3_public_config()
    diagnostic_config = _build_evaluation_v3_diagnostic_config()
    if public_config:
        trace_metadata["config"] = public_config
    if diagnostic_config:
        trace_metadata["diagnostic_config"] = diagnostic_config
    return trace_metadata


def _build_evaluation_v3_public_config() -> dict[str, Any]:
    return {
        "label_logic": "accepted_evidence_count",
        "max_required_evidence_count": 5,
        "coverage_used_for_label": False,
    }


def _build_evaluation_v3_diagnostic_config() -> dict[str, Any]:
    return {
        "grounding_threshold": evaluation_v3_config["GROUNDING_SCORE_THRESHOLD"],
        "grounding_top_n": evaluation_v3_config["GROUNDING_TOP_N"],
        "subsection_threshold": evaluation_v3_config["SUBSECTION_COVERAGE_THRESHOLD"],
    }


def _write_debug_reports(
    *,
    case_id: str,
    source_saved_path: Path,
    compliance_model: str,
    units: list[Any],
) -> None:
    debug_rows = build_debug_report_rows(units)
    for row in debug_rows:
        if (
            int(row.get("record_evidence_section_count") or 0) > 0
            and int(row.get("grounded_evidence_count") or 0) == 0
        ):
            logger.warning(
                "Grounding failure: evidence exists but none accepted "
                "[deliverable_id=%s]",
                row.get("deliverable_id") or "",
            )
        if (
            int(row.get("grounded_evidence_count") or 0) == 0
            and row.get("final_label") != "not_satisfied"
        ):
            logger.warning(
                "Final label inconsistency in evaluation_v3 "
                "[deliverable_id=%s grounded_evidence_count=0 final_label=%s]",
                row.get("deliverable_id") or "",
                row.get("final_label") or "",
            )
        if (
            row.get("contradiction_type") not in {"none", "missing_evidence"}
            and not bool(row.get("has_conflict"))
        ):
            logger.warning(
                "Conflict mapping inconsistency in evaluation_v3 "
                "[deliverable_id=%s contradiction_type=%s has_conflict=%s]",
                row.get("deliverable_id") or "",
                row.get("contradiction_type") or "",
                row.get("has_conflict"),
            )
    json_path = source_saved_path.with_name(source_saved_path.stem + "_debug.json")
    csv_path = source_saved_path.with_name(source_saved_path.stem + "_debug.csv")
    json_path.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "created_at": _current_timestamp(),
                "compliance_model": compliance_model,
                "evaluation_v3_config": evaluation_v3_config,
                "source_evaluation_v3_path": source_saved_path.as_posix(),
                "compact_summary": build_compact_summary(units),
                "summary": build_debug_report_summary(units),
                "edge_case_debug_table": build_edge_case_debug_rows(units),
                "suspicious_rows": build_suspicious_debug_rows(units),
                "rows": debug_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_debug_csv(csv_path=csv_path, rows=debug_rows)
    _write_manual_debug_export(case_id=case_id, debug_payload_path=json_path)


def _write_debug_csv(*, csv_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "deliverable_id",
        "requirement_type",
        "base_required_evidence_count",
        "weight",
        "weight_modifier",
        "required_evidence_count_reason",
        "final_label",
        "evidence_status",
        "required_evidence_count",
        "grounded_evidence_count",
        "evidence_coverage_ratio",
        "grounded_chunk_count",
        "grounded_subsection_count",
        "has_conflict",
        "subsection_count",
        "subsection_ids",
        "subsection_coverage_ratio",
        "subsection_threshold",
        "subsection_downgrade_applied",
        "contradiction_type",
        "evidence_score",
        "record_evidence_section_count",
        "reference_evidence_section_count",
        "stage_1_label",
        "stage_2_label",
        "stage_3_label",
        "rationale",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manual_debug_export(*, case_id: str, debug_payload_path: Path) -> None:
    EVALUATION_V3_MANUAL_COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_path = (
        EVALUATION_V3_MANUAL_COMPARISONS_DIR
        / f"case_{case_id}_evaluation_v3_debug_export_{timestamp}.json"
    )
    shutil.copy2(debug_payload_path, target_path)
    latest_path = EVALUATION_V3_MANUAL_COMPARISONS_DIR / "evaluation_v3_debug_export.json"
    shutil.copy2(debug_payload_path, latest_path)


def _build_stage_output(
    *,
    finding: Any | None,
    linked_row: Any | None,
    supporting_reference_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    # Evaluation V3 metrics (ground truth)
    # Uses grounded evidence and retrieval.
    # IMPORTANT: evaluation_v3 must not read or depend on legacy finding.status.
    if finding is None and linked_row is None:
        return None
    evidence_items = list(_payload_value(finding, "evidence_items", [])) if finding is not None else []
    rationale = str(_payload_value(linked_row, "rationale", "")) if linked_row is not None else ""
    reference_ids = supporting_reference_ids or []
    record_ids = _dedupe_preserve_order(
        [
            str(_payload_value(item, "evidence_id", "") or "").strip()
            for item in evidence_items
            if str(_payload_value(item, "evidence_id", "") or "").strip()
        ]
    )
    label = _derive_stage_label_from_grounding(
        rationale=rationale,
        evidence_items=evidence_items,
        supporting_reference_ids=reference_ids,
    )
    contradiction_type = _pick_highest_priority_contradiction(
        [
            _payload_value(finding, "contradiction_type", None),
            _payload_value(linked_row, "contradiction_type", None),
        ]
    )
    conflict_flag = any(
        _coerce_bool_flag(value)
        for value in (
            _payload_value(finding, "conflict_flag", None),
            _payload_value(linked_row, "conflict_flag", None),
        )
    )
    return {
        "label": label,
        "rationale": rationale,
        "evidence": list(_payload_value(finding, "evidence", [])) if finding is not None else [],
        "evidence_items": [
            {
                "evidence_id": _payload_value(item, "evidence_id", ""),
                "section_id": _payload_value(item, "section_id", ""),
                "subsection_id": _payload_value(item, "subsection_id", ""),
                "section_label": _payload_value(item, "section_label", ""),
                "heading_title": _payload_value(item, "heading_title", ""),
                "source_document": _payload_value(item, "source_document", ""),
                "source_stage": _payload_value(item, "source_stage", ""),
                "text": _payload_value(item, "text", ""),
            }
            for item in evidence_items
        ],
        "supporting_record_evidence_ids": record_ids,
        "conflict_flag": conflict_flag,
        "contradiction_type": contradiction_type,
        "supporting_reference_ids": reference_ids,
    }


def _build_evaluation_v3_input_mapping(
    *,
    deliverables: list[dict[str, Any]],
    compliance_response: ComplianceResponse,
    retrieved_payload: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    entries, alias_to_key = _build_deliverable_entries(deliverables)
    expected_keys = set(entries)

    retrieved_map = _map_retrieved_payload_to_keys(
        retrieved_payload=retrieved_payload,
        alias_to_key=alias_to_key,
    )
    _validate_source_keys(
        source_name="retrieved_payload",
        expected_keys=expected_keys,
        actual_keys=set(retrieved_map),
    )
    for requirement_key, payload in retrieved_map.items():
        entries[requirement_key]["retrieved_rows"] = payload

    for stage_result in compliance_response.stages:
        findings_map = _map_stage_findings_to_keys(
            findings=stage_result.analysis.procedure_to_record or stage_result.analysis.findings,
            alias_to_key=alias_to_key,
            stage_key=stage_result.stage_key,
        )
        linked_rows_map = _map_stage_linked_rows_to_keys(
            linked_rows=stage_result.analysis.linked_rows,
            alias_to_key=alias_to_key,
            stage_key=stage_result.stage_key,
        )
        _validate_source_keys(
            source_name=f"{stage_result.stage_key}.findings",
            expected_keys=expected_keys,
            actual_keys=set(findings_map),
        )
        _validate_source_keys(
            source_name=f"{stage_result.stage_key}.linked_rows",
            expected_keys=expected_keys,
            actual_keys=set(linked_rows_map),
        )
        for requirement_key in expected_keys:
            entries[requirement_key]["findings"][stage_result.stage_key] = findings_map.get(requirement_key)
            entries[requirement_key]["linked_rows"][stage_result.stage_key] = linked_rows_map.get(requirement_key)

    return entries


def _build_deliverable_entries(
    deliverables: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    entries: dict[str, dict[str, Any]] = {}
    alias_to_key: dict[str, str] = {}
    for index, deliverable in enumerate(deliverables):
        requirement_key = _resolve_deliverable_key(deliverable, index)
        if requirement_key in entries:
            raise ValueError(f"Duplicate evaluation_v3 deliverable key: {requirement_key}")
        aliases = _collect_deliverable_aliases(deliverable=deliverable, index=index, requirement_key=requirement_key)
        entries[requirement_key] = {
            "deliverable": deliverable,
            "retrieved_rows": {},
            "findings": {},
            "linked_rows": {},
        }
        for alias in aliases:
            existing = alias_to_key.get(alias)
            if existing and existing != requirement_key:
                raise ValueError(
                    f"Ambiguous evaluation_v3 alignment alias '{alias}' maps to both '{existing}' and '{requirement_key}'."
                )
            alias_to_key[alias] = requirement_key
    return entries, alias_to_key


def _map_retrieved_payload_to_keys(
    *,
    retrieved_payload: list[dict[str, Any]],
    alias_to_key: dict[str, str],
) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for payload in retrieved_payload:
        requirement_key = _resolve_required_key(
            aliases=_collect_payload_aliases(payload),
            alias_to_key=alias_to_key,
            source_name="retrieved_payload",
        )
        if requirement_key in mapped:
            raise ValueError(f"Duplicate evaluation_v3 retrieved payload key: {requirement_key}")
        mapped[requirement_key] = payload
    return mapped


def _map_stage_findings_to_keys(
    *,
    findings: list[Any],
    alias_to_key: dict[str, str],
    stage_key: str,
) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for finding in findings:
        requirement_key = _resolve_required_key(
            aliases=_collect_finding_aliases(finding),
            alias_to_key=alias_to_key,
            source_name=f"{stage_key}.findings",
        )
        if requirement_key in mapped:
            logger.warning("Duplicate findings merged for deliverable_id=%s", requirement_key)
            mapped[requirement_key] = _merge_duplicate_finding(
                existing=mapped[requirement_key],
                incoming=finding,
            )
            continue
        mapped[requirement_key] = _finding_to_payload(finding)
    return mapped


def _map_stage_linked_rows_to_keys(
    *,
    linked_rows: list[Any],
    alias_to_key: dict[str, str],
    stage_key: str,
) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for linked_row in linked_rows:
        requirement_key = _resolve_required_key(
            aliases=_collect_linked_row_aliases(linked_row),
            alias_to_key=alias_to_key,
            source_name=f"{stage_key}.linked_rows",
        )
        if requirement_key in mapped:
            logger.warning("Duplicate linked rows merged for deliverable_id=%s", requirement_key)
            mapped[requirement_key] = _merge_duplicate_linked_row(
                existing=mapped[requirement_key],
                incoming=linked_row,
            )
            continue
        mapped[requirement_key] = _linked_row_to_payload(linked_row)
    return mapped


def _payload_value(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _derive_stage_label_from_grounding(
    *,
    rationale: str,
    evidence_items: list[Any],
    supporting_reference_ids: list[str],
) -> str | None:
    normalized_rationale = " ".join(str(rationale or "").split()).strip().lower()
    if _has_direct_conflict_marker(normalized_rationale):
        return "not_satisfied"
    if any(_has_direct_conflict_marker(_payload_value(item, "text", "")) for item in evidence_items):
        return "not_satisfied"
    grounded_evidence_items = [
        item
        for item in evidence_items
        if _payload_value(item, "source_document", None)
    ]
    if grounded_evidence_items:
        return "satisfied"
    if supporting_reference_ids:
        return "partial"
    if normalized_rationale:
        return "partial"
    return None


def _pick_highest_priority_contradiction(values: list[object]) -> str:
    normalized_values = {
        str(value or "").strip().lower().replace(" ", "_")
        for value in values
        if str(value or "").strip()
    }
    for item in CONTRADICTION_PRIORITY:
        if item in normalized_values:
            return item
    return "none"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _coerce_bool_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value or "").strip().lower()
    return normalized in {"true", "1", "yes", "y"}


def _merge_unique_rationales(values: list[object]) -> str:
    normalized = [
        " ".join(str(value or "").split()).strip()
        for value in values
    ]
    unique = [value for value in _dedupe_preserve_order(normalized) if value]
    merged = " | ".join(unique)
    if len(merged) <= MAX_MERGED_RATIONALE_LENGTH:
        return merged
    return merged[: MAX_MERGED_RATIONALE_LENGTH - 3].rstrip() + "..."


def _merge_evidence_items(values: list[Any]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for value in values:
        normalized = {
            "evidence_id": " ".join(str(_payload_value(value, "evidence_id", "") or "").split()).strip(),
            "section_id": " ".join(str(_payload_value(value, "section_id", "") or "").split()).strip(),
            "subsection_id": " ".join(str(_payload_value(value, "subsection_id", "") or "").split()).strip(),
            "section_label": " ".join(str(_payload_value(value, "section_label", "") or "").split()).strip(),
            "heading_title": " ".join(str(_payload_value(value, "heading_title", "") or "").split()).strip(),
            "source_document": " ".join(str(_payload_value(value, "source_document", "") or "").split()).strip(),
            "text": " ".join(str(_payload_value(value, "text", "") or "").split()).strip(),
        }
        if not normalized["text"]:
            continue
        if normalized in merged:
            continue
        merged.append(normalized)
    return merged


def _finding_to_payload(finding: Any) -> dict[str, Any]:
    return {
        "requirement": str(_payload_value(finding, "requirement", "") or ""),
        "evidence": _dedupe_preserve_order(
            [
                " ".join(str(item or "").split()).strip()
                for item in list(_payload_value(finding, "evidence", []))
                if " ".join(str(item or "").split()).strip()
            ]
        ),
        "source_document": str(_payload_value(finding, "source_document", "") or ""),
        "evidence_items": _merge_evidence_items(list(_payload_value(finding, "evidence_items", []))),
        "conflict_flag": bool(_payload_value(finding, "conflict_flag", False)),
        "contradiction_type": _pick_highest_priority_contradiction(
            [_payload_value(finding, "contradiction_type", None)]
        ),
        "_duplicate_merged": bool(_payload_value(finding, "_duplicate_merged", False)),
    }


def _linked_row_to_payload(linked_row: Any) -> dict[str, Any]:
    return {
        "requirement": str(_payload_value(linked_row, "requirement", "") or ""),
        "requirement_ref": str(_payload_value(linked_row, "requirement_ref", "") or ""),
        "rationale": " ".join(str(_payload_value(linked_row, "rationale", "") or "").split()).strip(),
        "record_recall_at_k": _payload_value(linked_row, "record_recall_at_k", None),
        "conflict_flag": bool(_payload_value(linked_row, "conflict_flag", False)),
        "contradiction_type": _pick_highest_priority_contradiction(
            [_payload_value(linked_row, "contradiction_type", None)]
        ),
        "_duplicate_merged": bool(_payload_value(linked_row, "_duplicate_merged", False)),
    }


def _merge_duplicate_finding(*, existing: Any, incoming: Any) -> dict[str, Any]:
    left = _finding_to_payload(existing)
    right = _finding_to_payload(incoming)
    merged_evidence_items = _merge_evidence_items(
        [*left.get("evidence_items", []), *right.get("evidence_items", [])]
    )
    merged_evidence = _dedupe_preserve_order(
        [
            *list(left.get("evidence", [])),
            *list(right.get("evidence", [])),
        ]
    )
    return {
        "requirement": str(left.get("requirement") or right.get("requirement") or ""),
        "evidence": merged_evidence,
        "source_document": str(
            left.get("source_document")
            or right.get("source_document")
            or next(
                (
                    item.get("source_document", "")
                    for item in merged_evidence_items
                    if item.get("source_document")
                ),
                "",
            )
        ),
        "evidence_items": merged_evidence_items,
        "conflict_flag": bool(left.get("conflict_flag")) or bool(right.get("conflict_flag")),
        "contradiction_type": _pick_highest_priority_contradiction(
            [left.get("contradiction_type"), right.get("contradiction_type")]
        ),
        "_duplicate_merged": True,
    }


def _merge_duplicate_linked_row(*, existing: Any, incoming: Any) -> dict[str, Any]:
    left = _linked_row_to_payload(existing)
    right = _linked_row_to_payload(incoming)
    recall_candidates = [
        value for value in (left.get("record_recall_at_k"), right.get("record_recall_at_k"))
        if isinstance(value, (int, float))
    ]
    return {
        "requirement": str(left.get("requirement") or right.get("requirement") or ""),
        "requirement_ref": str(left.get("requirement_ref") or right.get("requirement_ref") or ""),
        "rationale": _merge_unique_rationales([left.get("rationale"), right.get("rationale")]),
        "record_recall_at_k": max(recall_candidates) if recall_candidates else None,
        "conflict_flag": bool(left.get("conflict_flag")) or bool(right.get("conflict_flag")),
        "contradiction_type": _pick_highest_priority_contradiction(
            [left.get("contradiction_type"), right.get("contradiction_type")]
        ),
        "_duplicate_merged": True,
    }


def _validate_source_keys(
    *,
    source_name: str,
    expected_keys: set[str],
    actual_keys: set[str],
) -> None:
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if not missing and not extra:
        return
    message = (
        f"evaluation_v3 alignment mismatch for {source_name}: "
        f"missing_keys={missing}, extra_keys={extra}"
    )
    logger.error(message)
    raise ValueError(message)


def _resolve_deliverable_key(deliverable: dict[str, Any], index: int) -> str:
    for key in ("deliverable_id", "requirement_id", "id"):
        value = _normalize_alignment_text(deliverable.get(key))
        if value:
            return value
    return _deliverable_id(index)


def _collect_deliverable_aliases(
    *,
    deliverable: dict[str, Any],
    index: int,
    requirement_key: str,
) -> list[str]:
    aliases = [
        requirement_key,
        _normalize_alignment_text(deliverable.get("deliverable_id")),
        _normalize_alignment_text(deliverable.get("requirement_id")),
        _normalize_alignment_text(deliverable.get("id")),
        _build_requirement_ref(index),
        _normalize_alignment_text(deliverable.get("requirement_text")),
    ]
    return [alias for alias in aliases if alias]


def _collect_payload_aliases(payload: dict[str, Any]) -> list[str]:
    deliverable = payload.get("deliverable") if isinstance(payload.get("deliverable"), dict) else {}
    aliases = [
        _normalize_alignment_text(payload.get("deliverable_id")),
        _normalize_alignment_text(payload.get("requirement_id")),
        _normalize_alignment_text(payload.get("requirement_ref")),
        _normalize_alignment_text(payload.get("requirement_text")),
        _normalize_alignment_text(deliverable.get("deliverable_id")),
        _normalize_alignment_text(deliverable.get("requirement_id")),
        _normalize_alignment_text(deliverable.get("requirement_text")),
    ]
    return [alias for alias in aliases if alias]


def _collect_finding_aliases(finding: Any) -> list[str]:
    aliases = [
        _normalize_alignment_text(getattr(finding, "requirement_id", "")),
        _normalize_alignment_text(getattr(finding, "deliverable_id", "")),
        _normalize_alignment_text(getattr(finding, "requirement", "")),
    ]
    return [alias for alias in aliases if alias]


def _collect_linked_row_aliases(linked_row: Any) -> list[str]:
    aliases = [
        _normalize_alignment_text(getattr(linked_row, "requirement_ref", "")),
        _normalize_alignment_text(getattr(linked_row, "requirement_id", "")),
        _normalize_alignment_text(getattr(linked_row, "deliverable_id", "")),
        _normalize_alignment_text(getattr(linked_row, "requirement", "")),
    ]
    return [alias for alias in aliases if alias]


def _resolve_required_key(
    *,
    aliases: list[str],
    alias_to_key: dict[str, str],
    source_name: str,
) -> str:
    for alias in aliases:
        requirement_key = alias_to_key.get(alias)
        if requirement_key:
            return requirement_key
    raise ValueError(f"Unable to align evaluation_v3 {source_name}; aliases={aliases}")


def _build_requirement_ref(index: int) -> str:
    return f"REQ-{index + 1}"


def _normalize_alignment_text(value: object) -> str:
    normalized = " ".join(str(value or "").split()).strip().lower()
    if not normalized:
        return ""
    normalized = normalized.rstrip(".,:;")
    normalized = normalized.replace(" for the risk reduction", " for risk reduction")
    return normalized


def _with_record_ids(*, deliverable_id: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        enriched.append(
            {
                **chunk,
                "evidence_id": chunk.get("evidence_id") or f"{deliverable_id}:record:{index}",
            }
        )
    return enriched


def _with_reference_ids(*, deliverable_id: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        enriched.append(
            {
                **chunk,
                "reference_id": chunk.get("reference_id") or f"{deliverable_id}:reference:{index}",
            }
        )
    return enriched


def _deliverable_id(index: int) -> str:
    return f"DELIV-{index + 1:03d}"


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _has_direct_conflict_marker(value: object) -> bool:
    normalized = " ".join(str(value or "").split()).strip().lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in CONFLICT_MARKERS)


def _log_retrieval_score_debug(retrieved_payload: list[dict[str, Any]]) -> None:
    logged = 0
    for payload in retrieved_payload:
        for key in ("retrieved_record_sections", "retrieved_requirement_context"):
            for chunk in payload.get(key, []):
                raw_score = chunk.get("raw_retrieval_score", chunk.get("reranker_score"))
                normalized_score = chunk.get("retrieval_score")
                if not isinstance(raw_score, (int, float)) or not isinstance(normalized_score, (int, float)):
                    continue
                logger.warning(
                    "[retrieval_score_debug] raw=%.2f normalized=%.3f",
                    float(raw_score),
                    float(normalized_score),
                )
                logged += 1
                if logged >= 20:
                    return


def _write_evaluation_v3_error_report(
    *,
    case_id: str,
    compliance_response: ComplianceResponse,
    error: Exception,
) -> None:
    run_dir = build_evaluation_v3_runtime_dir(case_id=case_id)
    path = run_dir / f"case_{case_id}_evaluation_v3_error.json"
    payload = {
        "case_id": case_id,
        "created_at": _current_timestamp(),
        "source_compliance_saved_at": compliance_response.saved_at,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_dir_matches_compliance(*, run_dir: Path, compliance_saved_at: str) -> bool:
    candidate_files = [
        path for path in run_dir.iterdir()
        if path.is_file() and path.suffix == ".json"
    ]
    for path in candidate_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("source_compliance_saved_at") == compliance_saved_at:
            return True
    return False
