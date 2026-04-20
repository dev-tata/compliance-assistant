from __future__ import annotations

import json
from typing import Any

from app.schemas.compliance import ComplianceRequest
from app.services.compliance_methods.compliance_method_common import (
    build_shared_output_instructions,
    execute_compliance_method,
)


def run_non_rag_compliance(
    *,
    case_id: str,
    case_payload: dict[str, Any],
    request: ComplianceRequest,
):
    baseline_payload = _build_non_rag_case_context(case_payload)
    requirement_source = request.requirement_source
    prompt_parts = [
        *build_shared_output_instructions(),
        "Method: non_rag",
        "Work directly from the parsed document structure only.",
        "Do not use retrieval, chunk-ranking, vector-search, or external evidence selection.",
        (
            "Base the analysis on the extracted requirement items and the record sections provided below."
            " Use the original procedure sections below to verify and refine the extracted requirements."
            if requirement_source == "deliverables"
            else "Base the analysis only on the procedure sections and record sections provided below."
        ),
        f"Case context:\n{json.dumps(baseline_payload, ensure_ascii=False, indent=2)}",
    ]

    if request.instructions:
        prompt_parts.append(f"Additional user instructions:\n{request.instructions}")

    return execute_compliance_method(
        case_id=case_id,
        case_payload=case_payload,
        request=request,
        method="non_rag",
        prompt="\n\n".join(prompt_parts),
    )


def _build_non_rag_case_context(case_payload: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "case_id": case_payload.get("case_id"),
        "title": case_payload.get("title"),
        "notes": case_payload.get("notes"),
        "records": [_simplify_document(document) for document in case_payload.get("records", [])],
        "procedures": [
            _simplify_document(document) for document in case_payload.get("procedures", [])
        ],
    }
    deliverables = case_payload.get("deliverables", [])
    if deliverables:
        payload["deliverables"] = deliverables
    return payload


def _simplify_document(document: dict[str, Any]) -> dict[str, Any]:
    parsed_json = document.get("parsed_json") or {}
    return {
        "document_type": document.get("document_type"),
        "source_filename": document.get("source_filename"),
        "stored_filename": document.get("stored_filename"),
        "group_id": document.get("group_id"),
        "language": document.get("language"),
        "parsed_json": {
            "source_filename": parsed_json.get("source_filename"),
            "stored_filename": parsed_json.get("stored_filename"),
            "parser_used": parsed_json.get("parser_used"),
            "metadata": parsed_json.get("metadata"),
            "sections": parsed_json.get("sections", []),
        },
    }
