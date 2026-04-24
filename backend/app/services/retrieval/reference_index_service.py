from __future__ import annotations

import json
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
from app.services.storage_paths import REFERENCE_INDEXES_DIR

def ensure_reference_index(document_payload: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    chunks = build_document_section_chunks([document_payload])
    if not chunks:
        raise RuntimeError("Reference document does not contain any retrievable sections.")

    fingerprint = fingerprint_chunks(chunks)
    index_dir = get_reference_index_dir(document_payload)
    cached_index = load_cached_faiss_index(
        index_dir=index_dir,
        expected_fingerprint=fingerprint,
    )
    if cached_index is not None:
        return cached_index, chunks

    index, _ = build_faiss_index(chunks)
    save_cached_faiss_index(
        index_dir=index_dir,
        index=index,
        chunks=chunks,
        fingerprint=fingerprint,
    )
    return index, chunks


def search_reference_documents(
    *,
    documents: list[dict[str, Any]],
    query_text: str,
    top_k: int = FAISS_TOP_K,
    final_top_k: int = RERANK_TOP_K,
) -> list[dict[str, Any]]:
    return search_prepared_reference_indexes(
        prepared_indexes=prepare_reference_indexes(documents),
        query_text=query_text,
        top_k=top_k,
        final_top_k=final_top_k,
    )


def prepare_reference_indexes(
    documents: list[dict[str, Any]],
) -> list[tuple[Any, list[dict[str, Any]]]]:
    prepared_indexes: list[tuple[Any, list[dict[str, Any]]]] = []
    for document_payload in documents:
        try:
            prepared_indexes.append(ensure_reference_index(document_payload))
        except RuntimeError:
            continue
    return prepared_indexes


def search_prepared_reference_indexes(
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
    return rerank_results(
        query_text=query_text,
        candidates=results[: max(top_k, 1)],
        final_top_k=final_top_k,
    )


def remove_reference_index(document_payload: dict[str, Any]) -> None:
    index_dir = get_reference_index_dir(document_payload)
    if not index_dir.exists():
        return

    for path in index_dir.iterdir():
        if path.is_file():
            path.unlink()
    index_dir.rmdir()


def get_reference_index_dir(document_payload: dict[str, Any]) -> Path:
    index_key = document_payload.get("content_hash") or document_payload.get("stored_filename")
    if not index_key:
        raise RuntimeError("Reference document is missing both content_hash and stored_filename.")
    return REFERENCE_INDEXES_DIR / str(index_key)


def load_reference_chunks(document_payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    chunks_path = get_reference_index_dir(document_payload) / "chunks.json"
    if not chunks_path.exists():
        return None
    try:
        return json.loads(chunks_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
