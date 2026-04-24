# compliance assistant

Frontend: React + Vite  
Backend: FastAPI

## Run

Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## URLs

- Frontend: `http://localhost:5173`
- Backend: `http://127.0.0.1:8000`

## Workflow

1. Upload procedure, record, and optional reference documents.
2. Extract deliverables from procedure documents.
3. Create a case from selected files.
4. Run `non_rag`, `single_source_rag`, or `multi_source_rag`.
5. Review saved outputs in the UI.

## Extraction

Procedure extraction runs before compliance.

- Prompt: `deliverable_extraction_v3_single_call`
- Input: parsed sections and tables
- Output: normalized deliverables with section metadata, quotes, type, and confidence
- Storage: `backend/storage/extraction/procedures/...`

## Compliance

Compliance compares extracted deliverables against record documents.

- `non_rag`: uses full record documents as evidence
- `single_source_rag`: uses retrieved record sections as evidence
- `multi_source_rag`: uses retrieved record sections as evidence and reference context only for interpretation

All methods share the same prompt core, status rules, and JSON output format. Only evidence scope changes by method.

Status values:

- `satisfied`: all required elements are supported
- `partial`: some required elements are supported
- `not_satisfied`: support is missing, weak, ambiguous, or inadmissible

Overall labels are computed in backend from requirement scores, not taken from the LLM.

## Retrieval

RAG retrieval uses embeddings, FAISS, and reranking.

- Embeddings: `BAAI/bge-base-en`
- Index: `IndexFlatIP`
- Initial retrieval: top 10
- Final context: top 3 reranked chunks
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

`single_source_rag` retrieves record sections per requirement.  
`multi_source_rag` retrieves record sections first, then reference context.

## Chunking

Retrieval keeps sections as the main chunk unit.

- keeps section labels and titles
- merges table text into section chunks
- splits only large sections
- uses overlap for long sections
- avoids content truncation

## Parsing

Parsing uses Docling and saves structured JSON.

- preserves section hierarchy and page data
- keeps structured tables
- removes parser artifacts
- normalizes noisy table text

## Metrics

Compliance outputs include:

- `m2_ordinal_score`
- `m3_evidence_weighted_score`
- `m5_grounding_score`
- `completion_percent`
- `overall_assessment`

RAG results can also include retrieval metrics such as `record_recall_at_k`.

## UI

- upload documents and create cases
- freeze or unfreeze procedure documents
- block edits and deletion for frozen procedures
- review compliance history and results

## Storage

- parsed documents: `backend/storage/documents/parsed/...`
- retrieval data: `backend/storage/retrieval/...`
- extraction history: `backend/storage/extraction/procedures/...`
- case results: `backend/storage/cases/...`

After parser, extraction, or retrieval changes, rebuild saved data before comparing results.
