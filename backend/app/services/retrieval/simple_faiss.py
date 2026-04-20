from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


EMBED_DIM = 384
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def normalize_whitespace(value: str | None) -> str:
    return " ".join((value or "").split()).strip()


def summarize_text(value: str | None, *, max_chars: int = 1600) -> str:
    normalized = normalize_whitespace(value)
    return normalized[:max_chars]


def build_record_section_chunks(case_payload: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for document in case_payload.get("records", []):
        parsed_json = document.get("parsed_json") or {}
        source_filename = document.get("source_filename") or document.get("stored_filename") or "unknown"
        chunks.extend(
            _flatten_sections(
                parsed_json.get("sections", []),
                source_filename=source_filename,
            )
        )
    return [chunk for chunk in chunks if chunk.get("retrieval_text")]


def build_deliverable_chunks(deliverables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for index, item in enumerate(deliverables):
        requirement_text = normalize_whitespace(item.get("requirement_text"))
        if not requirement_text:
            continue
        heading_title = normalize_whitespace(item.get("heading_title"))
        section_label = normalize_whitespace(item.get("section_label"))
        source_document = normalize_whitespace(item.get("source_document"))
        source_quote = normalize_whitespace(item.get("source_quote"))
        retrieval_text = normalize_whitespace(
            " ".join(
                part
                for part in (
                    requirement_text,
                    heading_title,
                    section_label,
                    source_quote,
                    source_document,
                )
                if part
            )
        )
        chunks.append(
            {
                "chunk_id": f"deliverable:{index}",
                "requirement_text": requirement_text,
                "heading_title": heading_title,
                "section_label": section_label,
                "source_document": source_document,
                "source_quote": source_quote,
                "retrieval_text": retrieval_text,
            }
        )
    return chunks


def build_faiss_index(chunks: list[dict[str, Any]]) -> tuple[Any, np.ndarray]:
    if faiss is None:  # pragma: no cover
        raise RuntimeError("FAISS is not installed. Add `faiss-cpu` to the backend environment.")

    texts = [chunk.get("retrieval_text", "") for chunk in chunks]
    matrix = embed_texts(texts)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index, matrix


def load_cached_faiss_index(
    *,
    index_dir: Path,
    expected_fingerprint: str,
) -> Any | None:
    if faiss is None:  # pragma: no cover
        raise RuntimeError("FAISS is not installed. Add `faiss-cpu` to the backend environment.")

    meta_path = index_dir / "meta.json"
    index_path = index_dir / "index.faiss"
    if not meta_path.exists() or not index_path.exists():
        return None

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if metadata.get("fingerprint") != expected_fingerprint:
        return None

    try:
        return faiss.read_index(str(index_path))
    except Exception:
        return None


def save_cached_faiss_index(
    *,
    index_dir: Path,
    index: Any,
    chunks: list[dict[str, Any]],
    fingerprint: str,
) -> None:
    if faiss is None:  # pragma: no cover
        raise RuntimeError("FAISS is not installed. Add `faiss-cpu` to the backend environment.")

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (index_dir / "meta.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "chunk_count": len(chunks),
                "embedding_dim": EMBED_DIM,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def fingerprint_chunks(chunks: list[dict[str, Any]]) -> str:
    serialized = json.dumps(chunks, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def search_index(
    *,
    index: Any,
    chunks: list[dict[str, Any]],
    query_text: str,
    top_k: int,
) -> list[dict[str, Any]]:
    if not chunks:
        return []
    query_vector = embed_texts([query_text])
    limit = min(max(top_k, 1), len(chunks))
    scores, positions = index.search(query_vector, limit)
    results: list[dict[str, Any]] = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        chunk = chunks[int(position)]
        results.append(
            {
                **chunk,
                "retrieval_score": round(float(score), 4),
            }
        )
    return results


def embed_texts(texts: list[str]) -> np.ndarray:
    vectors = np.zeros((len(texts), EMBED_DIM), dtype="float32")
    for row_index, text in enumerate(texts):
        tokens = TOKEN_PATTERN.findall((text or "").lower())
        if not tokens:
            continue
        for token in tokens:
            column = _token_index(token)
            vectors[row_index, column] += 1.0
        norm = np.linalg.norm(vectors[row_index])
        if norm > 0:
            vectors[row_index] /= norm
    return vectors


def _token_index(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % EMBED_DIM


def _flatten_sections(
    sections: list[dict[str, Any]],
    *,
    source_filename: str,
    parent_heading: str | None = None,
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for index, section in enumerate(sections):
        heading_title = normalize_whitespace(section.get("heading_title"))
        combined_heading = " / ".join(
            part for part in (parent_heading, heading_title) if part
        ) or heading_title or "Untitled section"
        section_label = normalize_whitespace(section.get("section_label"))
        table_markdown = "\n\n".join(
            summarize_text(table.get("table_markdown"), max_chars=1000)
            for table in section.get("tables", [])[:3]
            if table.get("table_markdown")
        )
        text = summarize_text(section.get("text"), max_chars=1800)
        retrieval_text = normalize_whitespace(
            " ".join(
                part
                for part in (
                    combined_heading,
                    section_label,
                    text,
                    table_markdown,
                    source_filename,
                )
                if part
            )
        )
        flattened.append(
            {
                "chunk_id": f"{source_filename}:{section_label or index}",
                "source_document": source_filename,
                "section_label": section_label,
                "heading_title": combined_heading,
                "text": text,
                "table_markdown": table_markdown,
                "retrieval_text": retrieval_text,
            }
        )
        flattened.extend(
            _flatten_sections(
                section.get("subsections", []),
                source_filename=source_filename,
                parent_heading=combined_heading,
            )
        )
    return flattened
