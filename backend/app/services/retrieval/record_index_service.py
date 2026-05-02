from __future__ import annotations

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


def ensure_record_index(document_payload: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    chunks = build_document_section_chunks([document_payload])
    if not chunks:
        raise RuntimeError("Record document does not contain any retrievable sections.")

    fingerprint = fingerprint_chunks(chunks)
    index_dir = get_record_index_dir(document_payload)
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
