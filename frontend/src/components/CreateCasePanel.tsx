import type { FormEvent } from "react";

import { FrozenBadge } from "./FrozenBadge";
import type { DocumentRecord } from "../types";
import { documentTypeLabels, languageLabels } from "../constants";
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

type CreateCasePanelProps = {
  title: string;
  notes: string;
  procedureDocs: DocumentRecord[];
  recordDocs: DocumentRecord[];
  extractionInfoByDocument: Record<string, ExtractionInfo>;
  selectedProcedures: string[];
  selectedRecords: string[];
  busy: string;
  onSubmit: (event: FormEvent) => void | Promise<void>;
  onTitleChange: (value: string) => void;
  onNotesChange: (value: string) => void;
  onSelectedProceduresChange: (value: string[]) => void;
  onSelectedRecordsChange: (value: string[]) => void;
};

export function CreateCasePanel({
  title,
  notes,
  procedureDocs,
  recordDocs,
  extractionInfoByDocument,
  selectedProcedures,
  selectedRecords,
  busy,
  onSubmit,
  onTitleChange,
  onNotesChange,
  onSelectedProceduresChange,
  onSelectedRecordsChange,
}: CreateCasePanelProps) {
    function renderDocumentWithExtraction(
    doc: DocumentRecord,
    checked: boolean,
    onToggle: (checked: boolean) => void,
    showExtraction: boolean,
  ) {
    const extractionInfo = extractionInfoByDocument[doc.stored_filename];

    return (
      <div key={doc.stored_filename}>
        <label className="check">
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => onToggle(e.target.checked)}
          />
          <span className="document-option-copy">
            <span className="document-option-heading">
              <strong className="document-option-title">{doc.source_filename}</strong>
              <FrozenBadge document={doc} />
            </span>
            <span className="document-option-meta">
              {formatDocumentType(doc.document_type)} · {doc.group_id ?? "no-group"} · {formatLanguage(doc.language)}
              {" · "}Created: {formatDateTime(doc.created_at)}
              {showExtraction && extractionInfo ? ` · Extraction: ${extractionInfo.provider} · ${extractionInfo.model}` : ""}
              {doc.frozen ? " · Edit protection enabled" : ""}
            </span>
          </span>
        </label>
      </div>
    );
  }

  return (
    <section className="panel panel-wide">
      <h2>Create case</h2>
      <form onSubmit={onSubmit} className="stack">
        <label className="field">
          <span>Title</span>
          <input value={title} onChange={(e) => onTitleChange(e.target.value)} />
        </label>
        <label className="field">
          <span>Notes</span>
          <textarea className="textarea-row" value={notes} onChange={(e) => onNotesChange(e.target.value)} rows={1} />
        </label>
        <div className="split-select create-case-columns">
          <div>
            <h3>Procedure files</h3>
            {procedureDocs.map((doc) => (
              renderDocumentWithExtraction(
                doc,
                selectedProcedures.includes(doc.stored_filename),
                (checked) =>
                  onSelectedProceduresChange(
                    checked
                      ? [...selectedProcedures, doc.stored_filename]
                      : selectedProcedures.filter((item) => item !== doc.stored_filename),
                  ),
                true,
              )
            ))}
          </div>
          <div>
            <h3>Record files</h3>
            {recordDocs.map((doc) => (
              <div key={doc.stored_filename}>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={selectedRecords.includes(doc.stored_filename)}
                    onChange={(e) =>
                      onSelectedRecordsChange(
                        e.target.checked
                          ? [...selectedRecords, doc.stored_filename]
                          : selectedRecords.filter((item) => item !== doc.stored_filename),
                        )
                      }
                    />
                  <span className="document-option-copy">
                    <strong className="document-option-title">{doc.source_filename}</strong>
                    <span className="document-option-meta">
                      {formatDocumentType(doc.document_type)} · {doc.group_id ?? "no-group"} · {formatLanguage(doc.language)}
                      {" · "}Created: {formatDateTime(doc.created_at)}
                    </span>
                  </span>
                </label>
              </div>
            ))}
            {recordDocs.length === 0 ? <p className="empty-state">No record files.</p> : null}
          </div>
        </div>
        <button
          className="button"
          disabled={selectedProcedures.length === 0 || selectedRecords.length === 0 || busy === "case"}
        >
          {busy === "case" ? "Creating..." : "Create case"}
        </button>
      </form>
    </section>
  );
}
