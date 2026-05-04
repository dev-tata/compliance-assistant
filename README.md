# Compliance Assistant

Frontend: React + Vite  
Backend: FastAPI + Pydantic v2

## Quick Start

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

URLs:
- Frontend: `http://localhost:5173`
- Backend: `http://127.0.0.1:8000`

## Core Workflow

1. **Upload** procedure, record, and optional reference documents
2. **Extract** deliverables from procedure documents
3. **Create** a case from selected files
4. **Run** compliance analysis (`non_rag`, `single_source_rag`, or `multi_source_rag`)
5. **Review** saved outputs in the UI

## Advanced: Batch Evaluation (Evaluation V3)

For systematic testing and benchmarking, use the evaluation_v3 system:

### Data Preparation

**Synthetic Records** (100 test cases):
```bash
cd backend
python evaluation_v3/preparse_synthetic_records.py
```

**Original Records** (your own risk assessments):
```bash
cd backend
python evaluation_v3/preparse_original_records.py
```

### Batch Compliance Analysis

Run evaluation on multiple records:
```bash
cd backend
python evaluation_v3/run_batch_compliance.py `
  --procedure-path "path/to/procedure.pdf" `
  --reference-path "path/to/reference.pdf" `
  --provider gemini `
  --model gemini-2.5-flash `
  --records-catalog evaluation_v3/cache/synthetic_record_registry.csv `
  --run-id my-test-run
```

**Key Features:**
- **3-Stage Pipeline**: Non-RAG baseline → Record retrieval → Reference retrieval
- **Evidence References**: One-to-one quote-element mappings with full traceability
- **Robust Parsing**: Enhanced JSON repair for LLM responses
- **Flexible Caching**: Separate indexes for synthetic vs original records

Results saved to `backend/evaluation_runs/<run-id>/` with detailed metrics and debug info.

## Extraction

Procedure extraction runs before compliance.

- **Prompt**: `deliverable_extraction_v3_single_call`
- **Input**: parsed sections and tables
- **Output**: normalized deliverables with section metadata, quotes, type, and confidence
- **Storage**: `backend/storage/extraction/procedures/...`

## Compliance Methods

Compliance compares extracted deliverables against record documents.

- **`non_rag`**: uses full record documents as evidence
- **`single_source_rag`**: uses retrieved record sections as evidence
- **`multi_source_rag`**: uses retrieved record sections + reference context for interpretation

**Legacy Compliance Status Values:**
- `satisfied`: all required elements supported
- `partial`: some required elements supported
- `not_satisfied`: support missing, weak, ambiguous, or inadmissible

**Evaluation V3 Status Values:**
- **Element Status**: `supported`, `partial`, `missing`, `contradicted`, `weak_match`
- **Evidence Status**: `supported`, `partial`, `missing`, `conflicting`
- **Evidence Audit Status**: `supported`, `partial`, `weak_match`, `missing`, `conflict`

Overall labels computed in backend from requirement scores, not LLM output.

## Retrieval System

RAG retrieval uses embeddings, FAISS, and reranking.

- **Embeddings**: `BAAI/bge-base-en`
- **Index**: `IndexFlatIP`
- **Initial retrieval**: top 10 chunks
- **Final context**: top 3 reranked chunks
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

`single_source_rag` retrieves record sections per requirement.  
`multi_source_rag` retrieves record sections first, then reference context.

## Document Processing

**Chunking**: Keeps sections as main units
- Preserves section labels and titles
- Merges table text into section chunks
- Splits only large sections with overlap
- Avoids content truncation

**Parsing**: Uses Docling for structured JSON
- Preserves section hierarchy and page data
- Keeps structured tables
- Removes parser artifacts
- Normalizes noisy table text

## Metrics & Outputs

**Legacy Compliance (deprecated):**
- `completion_percent`
- `weighted_completion_percent`
- `overall_coverage_percent`
- `weighted_coverage_percent`
- `average_evidence_strength`
- `weighted_average_evidence_strength`

**Evaluation V3 Metrics:**
- `satisfied_count`: Requirements fully satisfied
- `partial_count`: Requirements partially satisfied
- `not_satisfied_count`: Requirements not satisfied
- `supported_count`: Requirements with supported evidence status
- `missing_count`: Requirements with missing evidence status
- `requirements_with_conflict`: Requirements containing conflicts
- `total_conflict_findings`: Total number of conflict findings
- `requirements_by_conflict_type`: Conflicts grouped by type
- `conflict_findings_by_type`: Findings grouped by conflict type
- `avg_grounded_evidence_count`: Average grounded evidence per requirement
- `avg_evidence_coverage_ratio`: Average evidence coverage ratio

RAG results include retrieval metrics like `record_recall_at_k`.

## UI Features

- Upload documents and create cases
- Freeze/unfreeze procedure and reference documents
- Manually update procedure deliverables when unfrozen
- Block edits/deletion for frozen documents
- Review compliance history and results

## Storage Structure

- **Parsed documents**: `backend/storage/documents/parsed/...`
- **Retrieval data**: `backend/storage/retrieval/...`
- **Extraction history**: `backend/storage/extraction/procedures/...`
- **Case results**: `backend/storage/cases/...`
- **Evaluation runs**: `backend/evaluation_runs/...`
- **Evaluation cache**: `backend/evaluation_v3/cache/...`

**Note**: After parser, extraction, or retrieval changes, rebuild saved data before comparing results.
