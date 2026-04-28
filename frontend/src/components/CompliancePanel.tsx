import type { FormEvent } from "react";

import { FrozenBadge } from "./FrozenBadge";
import type { CaseRecord, DocumentRecord, LLMProviderDescriptor } from "../types";
import { documentTypeLabels, languageLabels } from "../constants";
import { getModelOptions } from "../utils/llmModelOptions";
import { formatDateTime } from "../utils/formatDateTime";

type ExtractionInfo = {
  provider: string;
  model: string;
} | null;

function formatLanguage(value: DocumentRecord["language"]) {
  if (!value) return "unknown";
  return languageLabels[value] ?? value;
}

function formatDocumentType(value: DocumentRecord["document_type"]) {
  if (!value) return "Untyped";
  return documentTypeLabels[value] ?? value;
}

type CompliancePanelProps = {
  cases: CaseRecord[];
  documents: DocumentRecord[];
  extractionInfoByDocument: Record<string, ExtractionInfo>;
  providers: LLMProviderDescriptor[];
  selectedCaseId: string;
  provider: string;
  model: string;
  instructions: string;
  selectedAdditionalDocuments: string[];
  busy: string;
  onSubmit: (event: FormEvent) => void | Promise<void>;
  onSelectCase: (caseId: string) => void;
  onProviderChange: (value: string) => void;
  onModelChange: (value: string) => void;
  onInstructionsChange: (value: string) => void;
  onSelectedAdditionalDocumentsChange: (value: string[]) => void;
};

export function CompliancePanel({
  cases,
  documents,
  extractionInfoByDocument,
  providers,
  selectedCaseId,
  provider,
  model,
  instructions,
  selectedAdditionalDocuments,
  busy,
  onSubmit,
  onSelectCase,
  onProviderChange,
  onModelChange,
  onInstructionsChange,
  onSelectedAdditionalDocumentsChange,
}: CompliancePanelProps) {
  const selectedCase = cases.find((item) => item.case_id === selectedCaseId);
  const selectedProcedureIds = selectedCase?.procedure_stored_filenames ?? [];
  const selectedRecordIds = selectedCase?.record_stored_filenames ?? [];
  const selectedReferenceIds = selectedCase?.reference_stored_filenames ?? [];
  const caseDocumentIds = new Set([
    ...selectedProcedureIds,
    ...selectedRecordIds,
    ...selectedReferenceIds,
  ]);
  const sortedDocuments = documents
    .filter((doc) => !caseDocumentIds.has(doc.stored_filename))
    .filter((doc) => doc.document_type === "reference")
    .sort((left, right) => left.source_filename.localeCompare(right.source_filename));
  const missingProcedureExtraction = selectedProcedureIds.filter(
    (storedFilename) => !extractionInfoByDocument[storedFilename],
  );
  const hasReferencesForNested = selectedReferenceIds.length > 0 || selectedAdditionalDocuments.length > 0;
  const readinessMessage = missingProcedureExtraction.length > 0
    ? "Run deliverable extraction for all procedure documents before compliance."
    : !hasReferencesForNested
      ? "Two-Stage RAG needs at least one reference document in the case or selected below."
      : "";
  const isSubmitDisabled = !selectedCaseId || busy === "compliance" || Boolean(readinessMessage);
  const complianceModelOptions = getModelOptions(provider, model);

  return (
    <section className="panel panel-wide">
      <h2>Run compliance</h2>
      <form onSubmit={onSubmit} className="stack">
        <label className="field">
          <span>Case</span>
          <select value={selectedCaseId} onChange={(e) => onSelectCase(e.target.value)}>
            <option value="">Select case</option>
            {cases.map((item) => <option key={item.case_id} value={item.case_id}>{item.title}</option>)}
          </select>
        </label>
        <div className="row">
          <label className="field">
            <span>Compliance provider</span>
            <select value={provider} onChange={(e) => onProviderChange(e.target.value)}>
              {providers.map((item) => (
                <option key={item.key} value={item.key}>{item.label}</option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Compliance model</span>
            <select value={model} onChange={(e) => onModelChange(e.target.value)}>
              {complianceModelOptions.map((item) => (
                <option key={item} value={item}>{item}</option>
              ))}
            </select>
          </label>
        </div>
        <p className="empty-state">
          Uses extracted procedure deliverables, then a record-retrieval stage followed by a reference-informed upgrade stage.
        </p>
        <div>
          <h3>Reference files</h3>
          {sortedDocuments.map((doc) => (
            <div key={doc.stored_filename}>
              <label className="check">
                <input
                  type="checkbox"
                  checked={selectedAdditionalDocuments.includes(doc.stored_filename)}
                  onChange={(e) =>
                    onSelectedAdditionalDocumentsChange(
                      e.target.checked
                        ? [...selectedAdditionalDocuments, doc.stored_filename]
                        : selectedAdditionalDocuments.filter((item) => item !== doc.stored_filename),
                    )
                  }
                />
                <span className="document-option-copy">
                  <span className="document-option-heading">
                    <strong className="document-option-title">{doc.source_filename}</strong>
                    <FrozenBadge document={doc} />
                  </span>
                  <span className="document-option-meta">
                    {formatDocumentType(doc.document_type)} · {doc.group_id ?? "no-group"} · {formatLanguage(doc.language)}
                    {" · "}Created: {formatDateTime(doc.created_at)}
                    {extractionInfoByDocument[doc.stored_filename]
                      ? ` · Extraction: ${extractionInfoByDocument[doc.stored_filename]?.provider} · ${extractionInfoByDocument[doc.stored_filename]?.model}`
                      : ""}
                  </span>
                </span>
              </label>
            </div>
          ))}
          {sortedDocuments.length === 0 ? <p className="empty-state">No extra documents available.</p> : null}
        </div>
        {readinessMessage ? <p className="empty-state">{readinessMessage}</p> : null}
        <label className="field">
          <span>Instructions</span>
          <textarea className="textarea-row" value={instructions} onChange={(e) => onInstructionsChange(e.target.value)} rows={1} />
        </label>
        <button className="button" disabled={isSubmitDisabled}>
          {busy === "compliance" ? "Running..." : "Run compliance"}
        </button>
      </form>
    </section>
  );
}
