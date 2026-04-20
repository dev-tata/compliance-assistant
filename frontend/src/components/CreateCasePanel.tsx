import type { FormEvent } from "react";

import type { DocumentRecord } from "../types";

type ExtractionInfo = {
  provider: string;
  model: string;
} | null;

type CreateCasePanelProps = {
  title: string;
  notes: string;
  procedureDocs: DocumentRecord[];
  recordDocs: DocumentRecord[];
  referenceDocs: DocumentRecord[];
  extractionInfoByDocument: Record<string, ExtractionInfo>;
  selectedProcedures: string[];
  selectedRecords: string[];
  selectedReferences: string[];
  busy: string;
  onSubmit: (event: FormEvent) => void | Promise<void>;
  onTitleChange: (value: string) => void;
  onNotesChange: (value: string) => void;
  onSelectedProceduresChange: (value: string[]) => void;
  onSelectedRecordsChange: (value: string[]) => void;
  onSelectedReferencesChange: (value: string[]) => void;
};

export function CreateCasePanel({
  title,
  notes,
  procedureDocs,
  recordDocs,
  referenceDocs,
  extractionInfoByDocument,
  selectedProcedures,
  selectedRecords,
  selectedReferences,
  busy,
  onSubmit,
  onTitleChange,
  onNotesChange,
  onSelectedProceduresChange,
  onSelectedRecordsChange,
  onSelectedReferencesChange,
}: CreateCasePanelProps) {
  function renderDocumentWithExtraction(doc: DocumentRecord, checked: boolean, onToggle: (checked: boolean) => void) {
    const extractionInfo = extractionInfoByDocument[doc.stored_filename];

    return (
      <div key={doc.stored_filename}>
        <label className="check">
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => onToggle(e.target.checked)}
          />
          <span>{doc.source_filename}</span>
        </label>
        {extractionInfo ? (
          <p className="empty-state">
            Extraction: {extractionInfo.provider} · {extractionInfo.model}
          </p>
        ) : null}
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
          <textarea value={notes} onChange={(e) => onNotesChange(e.target.value)} rows={4} />
        </label>
        <div className="split-select">
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
              )
            ))}
          </div>
          <div>
            <h3>Record files</h3>
            {recordDocs.map((doc) => (
              <label className="check" key={doc.stored_filename}>
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
                <span>{doc.source_filename}</span>
              </label>
            ))}
            {recordDocs.length === 0 ? <p className="empty-state">No record files.</p> : null}
          </div>
          <div>
            <h3>Reference files</h3>
            {referenceDocs.map((doc) => (
              renderDocumentWithExtraction(
                doc,
                selectedReferences.includes(doc.stored_filename),
                (checked) =>
                  onSelectedReferencesChange(
                    checked
                      ? [...selectedReferences, doc.stored_filename]
                      : selectedReferences.filter((item) => item !== doc.stored_filename),
                  ),
              )
            ))}
            {referenceDocs.length === 0 ? <p className="empty-state">No reference files.</p> : null}
          </div>
        </div>
        <button
          className="button"
          disabled={!title || selectedProcedures.length === 0 || selectedRecords.length === 0 || busy === "case"}
        >
          {busy === "case" ? "Creating..." : "Create case"}
        </button>
      </form>
    </section>
  );
}
