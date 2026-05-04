# Evaluation V3

Status: `stable_v1`

## Evaluation Unit Template

Main fields:
- `deliverable_id`: unique requirement/deliverable ID
- `procedure_section_link`: link to procedure source section
- `requirement_text`: normalized requirement text
- `weight`: importance weight for aggregation
- `record_evidence_chunks`: record-side evidence chunks
- `reference_evidence_chunks`: reference-side evidence chunks
- `stage_1_answer`: stage 1 result (label + rationale)
- `stage_2_answer`: stage 2 result (label + rationale)
- `stage_3_answer`: stage 3 result (label + rationale)
- `final_label`: final compliance decision
- `final_rationale`: short final explanation

## Pipeline Stages

1. **Stage 1**: Non-RAG baseline compliance evaluation
2. **Stage 2**: Record retrieval + compliance evaluation
3. **Stage 3**: Reference retrieval + compliance evaluation
4. **Final**: Aggregated decision from all stages

## Data Preparation

### Synthetic Records Prep
Parse synthetic records and build retrieval indexes:

```powershell
cd backend
python evaluation_v3/preparse_synthetic_records.py
```

Locations:
- Catalog: `evaluation_v3/cache/synthetic_record_registry.csv`
- Parsed JSON: `evaluation_v3/cache/parsed/`
- Record indexes: `evaluation_v3/cache/retrieval/records/`
- Run outputs: `evaluation_v3/output/<timestamp>__synthetic-retrieval-prep/`

### Original Records Prep
Parse original risk assessment records and build indexes:

```powershell
cd backend
python evaluation_v3/preparse_original_records.py
```

Locations:
- Catalog: `evaluation_v3/cache_original/original_record_registry.csv`
- Parsed JSON: `evaluation_v3/cache_original/parsed/`
- Record indexes: `evaluation_v3/cache_original/retrieval/records/`
- Run outputs: `evaluation_v3/output/<timestamp>__original-retrieval-prep/`

## Batch Compliance Evaluation

Run compliance evaluation on multiple records:

```powershell
cd backend
python evaluation_v3/run_batch_compliance.py `
  --procedure-path "path/to/procedure.pdf" `
  --reference-path "path/to/reference.pdf" `
  --provider gemini `
  --model gemini-2.5-flash `
  --records-catalog evaluation_v3/cache/synthetic_record_registry.csv `
  --run-id my-run-name
```

For original records, use:
```powershell
--records-catalog evaluation_v3/cache_original/original_record_registry.csv
```

Optional filters:
- `--record-ids 021 064 091`: Run specific records only
- `--limit 10`: Limit to first N records

## Output Structure

Results saved to `backend/evaluation_runs/<run-id>/`:
- `record_001/`, `record_002/`, ...: Per-record outputs
  - `evaluation_v3_result.json`: Full evaluation results
  - `evaluation_v3_debug.json`: Debug info with conflict summaries
  - `evaluation_v3_summary.json`: Aggregated metrics
- `batch_summary.json`: Overall batch statistics

## Key Features

- **Evidence References**: One-to-one quote-element mappings with full traceability
- **Multi-stage Evaluation**: Progressive refinement through retrieval stages
- **Robust Parsing**: Enhanced JSON repair for LLM responses
- **Flexible Caching**: Separate caches for synthetic vs original records
