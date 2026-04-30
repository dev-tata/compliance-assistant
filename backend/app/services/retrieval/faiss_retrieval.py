from __future__ import annotations

import hashlib
import json
import re
from threading import Lock
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError:  # pragma: no cover
    CrossEncoder = None
    SentenceTransformer = None

from app.services.retrieval.score_utils import normalize_retrieval_score


EMBED_MODEL_NAME = "BAAI/bge-base-en"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FAISS_TOP_K = 10
RERANK_TOP_K = 5
SECTION_TOKEN_LIMIT = 400
MIN_FINAL_SUBCHUNK_TOKENS = 200
SUBCHUNK_TOKEN_OVERLAP = 50
MIN_RETRIEVAL_SECTION_TEXT_LENGTH = 40
MIN_SUBSTANTIVE_PROSE_LENGTH = 240

_EMBEDDER: Any | None = None
_RERANKER: Any | None = None
_EMBEDDER_LOCK = Lock()
_RERANKER_LOCK = Lock()


def normalize_whitespace(value: str | None) -> str:
    return " ".join((value or "").split()).strip()


def clean_whitespace(value: str | None) -> str:
    return normalize_whitespace(value)


def build_record_section_chunks(case_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return build_document_section_chunks(case_payload.get("records", []))


def build_document_section_chunks(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for document in documents:
        parsed_json = document.get("parsed_json") or {}
        source_filename = document.get("source_filename") or document.get("stored_filename") or "unknown"
        document_type = normalize_whitespace(document.get("document_type")) or "unknown"
        chunks.extend(
            _flatten_sections(
                parsed_json.get("sections", []),
                source_filename=source_filename,
                document_type=document_type,
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
    if metadata.get("embedding_model") != EMBED_MODEL_NAME:
        return None
    if metadata.get("reranker_model") != RERANKER_MODEL_NAME:
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
                "embedding_dim": int(index.d),
                "embedding_model": EMBED_MODEL_NAME,
                "reranker_model": RERANKER_MODEL_NAME,
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
    top_k: int = FAISS_TOP_K,
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
                "faiss_score": round(float(score), 4),
            }
        )
    return results


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    embedder = get_embedder()
    vectors = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


def rerank_results(
    *,
    query_text: str,
    candidates: list[dict[str, Any]],
    final_top_k: int = RERANK_TOP_K,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    reranker = get_reranker()
    normalized_query = normalize_whitespace(query_text)
    pairs = [
        [normalized_query, _chunk_rerank_text(candidate)]
        for candidate in candidates
    ]
    scores = reranker.predict(pairs)
    reranked: list[dict[str, Any]] = []
    for candidate, score in zip(candidates, scores):
        raw_score = round(float(score), 4)
        reranked.append(
            {
                **candidate,
                "reranker_score": raw_score,
                "raw_retrieval_score": raw_score,
                "retrieval_score": round(normalize_retrieval_score(raw_score), 4),
            }
        )
    reranked.sort(
        key=lambda item: (
            float(item.get("reranker_score") or 0.0),
            float(item.get("faiss_score") or 0.0),
        ),
        reverse=True,
    )
    return reranked[: min(max(final_top_k, 1), len(reranked))]


def get_embedder() -> Any:
    global _EMBEDDER
    if SentenceTransformer is None:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is not installed. Add it to the backend environment."
        )
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:
                _EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBEDDER


def get_reranker() -> Any:
    global _RERANKER
    if CrossEncoder is None:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is not installed. Add it to the backend environment."
        )
    if _RERANKER is None:
        with _RERANKER_LOCK:
            if _RERANKER is None:
                _RERANKER = CrossEncoder(RERANKER_MODEL_NAME)
    return _RERANKER


def get_tokenizer() -> Any:
    tokenizer = getattr(get_embedder(), "tokenizer", None)
    if tokenizer is None:  # pragma: no cover
        raise RuntimeError("Embedding model tokenizer is unavailable.")
    return tokenizer


def _chunk_rerank_text(chunk: dict[str, Any]) -> str:
    return normalize_whitespace(
        " ".join(
            part
            for part in (
                chunk.get("heading_title"),
                chunk.get("section_label"),
                chunk.get("text"),
                chunk.get("table_markdown"),
                chunk.get("source_document"),
            )
            if part
        )
    )


def _flatten_sections(
    sections: list[dict[str, Any]],
    *,
    source_filename: str,
    document_type: str,
    parent_heading: str | None = None,
    root_section_id: str | None = None,
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    pending_heading_prefix: str | None = None
    for index, section in enumerate(sections):
        current_section_id = normalize_whitespace(section.get("section_id"))
        effective_section_id = root_section_id or current_section_id
        subsection_id = current_section_id or effective_section_id
        heading_title = normalize_whitespace(section.get("heading_title"))
        effective_heading = " / ".join(
            part for part in (pending_heading_prefix, heading_title) if part
        ) or heading_title
        combined_heading = " / ".join(
            part for part in (parent_heading, effective_heading) if part
        ) or effective_heading or "Untitled section"
        section_label = normalize_whitespace(section.get("section_label"))
        tables = section.get("tables", [])
        text = section.get("text")

        if _should_skip_retrieval_chunk(
            text=text,
            tables=tables,
        ):
            if index == 0 and not parent_heading and heading_title:
                pending_heading_prefix = heading_title
            flattened.extend(
                _flatten_sections(
                    section.get("subsections", []),
                    source_filename=source_filename,
                    document_type=document_type,
                    parent_heading=combined_heading,
                    root_section_id=effective_section_id,
                )
            )
            continue

        flattened.extend(
            _build_section_chunks(
                source_filename=source_filename,
                document_type=document_type,
                section_id=effective_section_id,
                subsection_id=subsection_id,
                section_label=section_label,
                heading_title=combined_heading,
                text=text,
                tables=tables,
                fallback_index=index,
            )
        )
        pending_heading_prefix = None
        flattened.extend(
            _flatten_sections(
                section.get("subsections", []),
                source_filename=source_filename,
                document_type=document_type,
                parent_heading=combined_heading,
                root_section_id=effective_section_id,
            )
        )
    return flattened


def _build_section_chunks(
    *,
    source_filename: str,
    document_type: str,
    section_id: str,
    subsection_id: str,
    section_label: str,
    heading_title: str,
    text: str | None,
    tables: list[dict[str, Any]],
    fallback_index: int,
) -> list[dict[str, Any]]:
    raw_table_markdown = _join_raw_table_markdown(tables)
    chunk_base_id = _build_chunk_base_id(
        source_filename=source_filename,
        section_id=section_id,
        section_label=section_label,
        fallback_index=fallback_index,
    )
    section_body = _build_clean_section_body(
        heading_title=heading_title,
        section_label=section_label,
        text=text,
        tables=tables,
    )
    if not section_body:
        retrieval_text = normalize_whitespace(
            " ".join(
                part
                for part in (heading_title, section_label, source_filename)
                if part
            )
        )
        return [
            {
                "chunk_id": chunk_base_id,
                "source_document": source_filename,
                "document_type": document_type,
                "section_id": section_id,
                "subsection_id": subsection_id or section_id,
                "section_label": section_label,
                "heading_title": heading_title,
                "text": "",
                "table_markdown": raw_table_markdown,
                "retrieval_text": retrieval_text,
            }
        ]

    body_chunks = _split_clean_section_body(section_body)
    chunks: list[dict[str, Any]] = []
    for group_index, chunk_text in enumerate(body_chunks):
        retrieval_header_parts = [
            f"[Document Type: {document_type}]",
            f"Document: {source_filename}",
        ]
        if section_label:
            retrieval_header_parts.append(f"Section: {section_label}")
        if heading_title:
            retrieval_header_parts.append(f"Title: {heading_title}")
        retrieval_text = clean_whitespace(
            "\n".join(retrieval_header_parts + ["", chunk_text])
        )
        chunks.append(
            {
                "chunk_id": f"{chunk_base_id}:{group_index}",
                "source_document": source_filename,
                "document_type": document_type,
                "section_id": section_id,
                "subsection_id": subsection_id or section_id,
                "section_label": section_label,
                "heading_title": heading_title,
                "text": chunk_text,
                "table_markdown": raw_table_markdown,
                "retrieval_text": retrieval_text,
            }
        )
    return chunks


def _build_chunk_base_id(
    *,
    source_filename: str,
    section_id: str,
    section_label: str,
    fallback_index: int,
) -> str:
    section_key = section_label or f"section-{fallback_index}"
    if section_id:
        return f"{source_filename}:{section_key}:{section_id}"
    return f"{source_filename}:{section_key}"


def _should_skip_retrieval_chunk(
    *,
    text: str | None,
    tables: list[dict[str, Any]],
) -> bool:
    if tables:
        return False
    normalized_text = normalize_whitespace(text)
    return len(normalized_text) < MIN_RETRIEVAL_SECTION_TEXT_LENGTH


def _build_clean_section_body(
    *,
    heading_title: str,
    section_label: str,
    text: str | None,
    tables: list[dict[str, Any]],
) -> str:
    parts: list[str] = []
    title_parts = [
        f"Title: {clean_whitespace(heading_title)}" if clean_whitespace(heading_title) else "",
        f"Section: {clean_whitespace(section_label)}" if clean_whitespace(section_label) else "",
    ]
    title_block = "\n".join(part for part in title_parts if part)
    if title_block:
        parts.append(title_block)

    normalized_text = _clean_text_block(text)
    if normalized_text:
        parts.append(normalized_text)
    include_inline_table_text = _should_inline_table_text(
        normalized_text=normalized_text,
        tables=tables,
    )
    if include_inline_table_text:
        for table in tables:
            table_text = _table_to_retrieval_text(table)
            if table_text:
                parts.append(table_text)
    return "\n\n".join(part for part in parts if part).strip()


def _split_clean_section_body(value: str) -> list[str]:
    normalized = value.strip()
    if not normalized:
        return []

    tokenizer = get_tokenizer()
    token_ids = _encode_tokens(tokenizer, normalized)
    if len(token_ids) <= SECTION_TOKEN_LIMIT:
        return [normalized]

    chunks: list[list[int]] = []
    step = max(1, SECTION_TOKEN_LIMIT - SUBCHUNK_TOKEN_OVERLAP)
    for start in range(0, len(token_ids), step):
        piece_tokens = token_ids[start:start + SECTION_TOKEN_LIMIT]
        if not piece_tokens:
            continue
        if len(piece_tokens) < MIN_FINAL_SUBCHUNK_TOKENS and chunks:
            chunks[-1].extend(piece_tokens)
            break
        chunks.append(piece_tokens)
        if start + SECTION_TOKEN_LIMIT >= len(token_ids):
            break

    decoded_chunks = [
        _normalize_tokenized_punctuation(
            tokenizer.decode(chunk, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        )
        for chunk in chunks
    ]
    return [chunk for chunk in decoded_chunks if chunk]


def _clean_text_block(value: str | None) -> str:
    if not value:
        return ""
    blocks = [_preserve_text_block_structure(block) for block in re.split(r"\n\s*\n", value)]
    blocks = [block for block in blocks if block]
    return "\n\n".join(blocks).strip()


def _preserve_text_block_structure(value: str | None) -> str:
    if not value:
        return ""
    cleaned_lines: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _table_to_retrieval_text(table: dict[str, Any]) -> str:
    rows = table.get("rows")
    cleaned_rows: list[str] = []
    if isinstance(rows, list):
        cleaned_rows.extend(_row_to_sentence(row) for row in rows if isinstance(row, dict))
    if not [row for row in cleaned_rows if row]:
        cleaned_rows.extend(_table_markdown_to_lines(table.get("table_markdown")))
    cleaned_rows = [row for row in cleaned_rows if row]
    if not cleaned_rows:
        return ""
    return "\n".join(cleaned_rows)


def _should_inline_table_text(
    *,
    normalized_text: str,
    tables: list[dict[str, Any]],
) -> bool:
    if not tables:
        return False
    if len(normalized_text) < MIN_SUBSTANTIVE_PROSE_LENGTH:
        return True
    return not any(_table_has_structured_content(table) for table in tables)


def _table_has_structured_content(table: dict[str, Any]) -> bool:
    rows = table.get("rows")
    if isinstance(rows, list) and rows:
        populated_rows = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            non_empty_values = [
                _clean_table_fragment(str(value))
                for value in row.values()
                if _clean_table_fragment(str(value))
            ]
            if len(non_empty_values) >= 2:
                populated_rows += 1
            if populated_rows >= 2:
                return True

    markdown = str(table.get("table_markdown") or "").strip()
    if not markdown:
        return False
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    data_lines = [
        line for line in lines
        if "|" in line and not all(char in {"|", "-", " ", ":"} for char in line)
    ]
    return len(data_lines) >= 3


def _row_to_sentence(row: dict[str, Any]) -> str:
    cleaned_items: list[tuple[str, str]] = []
    for key, value in row.items():
        key_text = _clean_table_fragment(str(key))
        value_text = _clean_table_fragment(str(value))
        if not key_text or not value_text:
            continue
        cleaned_items.append((key_text, value_text))

    if not cleaned_items:
        return ""

    if len(cleaned_items) == 2 and _looks_like_simple_pair(cleaned_items):
        return f"{cleaned_items[0][1]}: {cleaned_items[1][1]}"

    first_key, first_value = cleaned_items[0]
    remaining = cleaned_items[1:]
    if remaining and len(first_value.split()) <= 6:
        details = "; ".join(f"{key}: {value}" for key, value in remaining)
        return f"{first_value}. {details}".strip()

    return "; ".join(f"{key}: {value}" for key, value in cleaned_items)


def _clean_table_fragment(value: str | None) -> str:
    if not value:
        return ""
    cleaned = value.replace("<!-- rich cell -->", " ")
    cleaned = cleaned.replace("|", " ")
    cleaned = cleaned.replace("<br>", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.")
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if lowered in {"---", "n/a", "na"}:
        return ""
    return cleaned


def _normalize_tokenized_punctuation(value: str | None) -> str:
    text = value or ""
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    text = re.sub(r"(?<=\d)\s+\.\s+(?=\d)", ".", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _table_markdown_to_lines(value: str | None) -> list[str]:
    if not value:
        return []
    lines: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if set(line) <= {"|", "-", " ", ":"}:
            continue
        cols = [_clean_table_fragment(cell) for cell in line.strip("|").split("|")]
        cols = [cell for cell in cols if cell]
        if not cols:
            continue
        if len(cols) >= 2:
            lines.append(f"{cols[0]}: {cols[1]}")
        else:
            lines.append(cols[0])
    return lines


def _looks_like_simple_pair(items: list[tuple[str, str]]) -> bool:
    left_key, left_value = items[0]
    right_key, right_value = items[1]
    if any(not value for value in (left_value, right_value)):
        return False
    if any(_is_placeholder_label(label) for label in (left_key, right_key)):
        return False
    return len(left_value.split()) <= 6 and len(right_value.split()) <= 12


def _is_placeholder_label(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.startswith("column_") or lowered in {"label", "value", "item"}


def _join_raw_table_markdown(tables: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        str(table.get("table_markdown") or "").strip()
        for table in tables
        if str(table.get("table_markdown") or "").strip()
    ).strip()


def _encode_tokens(tokenizer: Any, value: str) -> list[int]:
    return _tokenize_ids(tokenizer, value)


def _tokenize_ids(tokenizer: Any, value: str) -> list[int]:
    encoded = tokenizer(
        value,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    return list(encoded.get("input_ids", []))
