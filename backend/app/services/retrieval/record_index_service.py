from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.retrieval.faiss_retrieval import (
    FAISS_TOP_K,
    RERANK_TOP_K,
    build_document_section_chunks,
    build_faiss_index,
    fingerprint_chunks,
    load_cached_faiss_index,
    rerank_results,
    save_cached_faiss_index,
    search_index,
)
from app.services.storage_paths import RECORD_INDEXES_DIR

logger = logging.getLogger(__name__)
CURRENT_RECORD_INDEX_VERSION = "record_index_v3"


def ensure_record_index(document_payload: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    chunks = build_document_section_chunks([document_payload])
    if not chunks:
        raise RuntimeError("Record document does not contain any retrievable sections.")

    fingerprint = fingerprint_chunks(chunks)
    index_dir = get_record_index_dir(document_payload)
    rebuild_reason = _resolve_record_index_rebuild_reason(
        index_dir=index_dir,
        expected_chunk_count=len(chunks),
    )
    if rebuild_reason is None:
        cached_index = load_cached_faiss_index(
            index_dir=index_dir,
            expected_fingerprint=fingerprint,
        )
        if cached_index is not None:
            return cached_index, chunks
        rebuild_reason = "fingerprint_or_model_mismatch"

    logger.info(
        "Rebuilding record index for %s: %s",
        document_payload.get("source_filename") or document_payload.get("stored_filename") or "unknown-record",
        rebuild_reason,
    )

    index, _ = build_faiss_index(chunks)
    save_cached_faiss_index(
        index_dir=index_dir,
        index=index,
        chunks=chunks,
        fingerprint=fingerprint,
        index_version=CURRENT_RECORD_INDEX_VERSION,
        chunking_config=_build_record_chunking_config_snapshot(),
        build_metadata={
            "rebuilt_at": datetime.now(timezone.utc).isoformat(),
            "rebuild_reason": rebuild_reason,
        },
    )
    return index, chunks


def prepare_record_indexes(
    documents: list[dict[str, Any]],
) -> list[tuple[Any, list[dict[str, Any]]]]:
    prepared_indexes: list[tuple[Any, list[dict[str, Any]]]] = []
    for document_payload in documents:
        try:
            prepared_indexes.append(ensure_record_index(document_payload))
        except RuntimeError:
            continue
    return prepared_indexes


def search_prepared_record_indexes(
    *,
    prepared_indexes: list[tuple[Any, list[dict[str, Any]]]],
    query_text: str,
    top_k: int = FAISS_TOP_K,
    final_top_k: int = RERANK_TOP_K,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, chunks in prepared_indexes:
        results.extend(
            search_index(
                index=index,
                chunks=chunks,
                query_text=query_text,
                top_k=top_k,
            )
        )
    results.sort(
        key=lambda item: float(item.get("faiss_score") or 0.0),
        reverse=True,
    )
    reranked_results = rerank_results(
        query_text=query_text,
        candidates=results[: max(top_k, 1)],
        final_top_k=final_top_k,
    )
    print(
        {
            "stage": "record_index_service.search_prepared_record_indexes",
            "returned": "reranked_results",
            "count": len(reranked_results),
            "sample": [
                {
                    "faiss_score": item.get("faiss_score"),
                    "reranker_score": item.get("reranker_score"),
                    "raw_retrieval_score": item.get("raw_retrieval_score"),
                    "retrieval_score": item.get("retrieval_score"),
                }
                for item in reranked_results[:3]
            ],
        }
    )
    return reranked_results


def remove_record_index(document_payload: dict[str, Any]) -> None:
    index_dir = get_record_index_dir(document_payload)
    if not index_dir.exists():
        return

    for path in index_dir.iterdir():
        if path.is_file():
            path.unlink()
    index_dir.rmdir()


def get_record_index_dir(document_payload: dict[str, Any]) -> Path:
    index_key = document_payload.get("content_hash") or document_payload.get("stored_filename")
    if not index_key:
        raise RuntimeError("Record document is missing both content_hash and stored_filename.")
    return RECORD_INDEXES_DIR / str(index_key)


def _resolve_record_index_rebuild_reason(
    *,
    index_dir: Path,
    expected_chunk_count: int,
) -> str | None:
    meta_path = index_dir / "meta.json"
    chunks_path = index_dir / "chunks.json"
    index_path = index_dir / "index.faiss"
    if not meta_path.exists():
        return "missing_meta_json"
    if not chunks_path.exists():
        return "missing_chunks_json"
    if not index_path.exists():
        return "missing_faiss_index"

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "invalid_meta_json"

    chunk_count = metadata.get("chunk_count")
    if not isinstance(chunk_count, int) or chunk_count <= 0:
        return "invalid_chunk_count"
    if chunk_count != expected_chunk_count:
        return f"chunk_count_mismatch old={chunk_count} current={expected_chunk_count}"

    current_version = metadata.get("index_version")
    if current_version != CURRENT_RECORD_INDEX_VERSION:
        return (
            "stale index_version "
            f"old={current_version!r} current={CURRENT_RECORD_INDEX_VERSION!r}"
        )

    return None


def _build_record_chunking_config_snapshot() -> dict[str, Any]:
    return {
        "section_token_limit": getattr(build_document_section_chunks, "__globals__", {}).get("SECTION_TOKEN_LIMIT"),
        "subchunk_token_overlap": getattr(build_document_section_chunks, "__globals__", {}).get("SUBCHUNK_TOKEN_OVERLAP"),
        "min_retrieval_section_text_length": getattr(
            build_document_section_chunks,
            "__globals__",
        ).get("MIN_RETRIEVAL_SECTION_TEXT_LENGTH"),
        "min_final_subchunk_tokens": getattr(
            build_document_section_chunks,
            "__globals__",
        ).get("MIN_FINAL_SUBCHUNK_TOKENS"),
    }
