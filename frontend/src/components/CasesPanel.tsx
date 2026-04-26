import { useState } from "react";

import type { CaseDocuments, CaseRecord, DocumentRecord } from "../types";
import { formatDateTime } from "../utils/formatDateTime";

type ExtractionInfo = {
  provider: string;
  model: string;
} | null;

type CasesPanelProps = {
  cases: CaseRecord[];
  documents: DocumentRecord[];
  selectedCaseId: string;
  caseDocuments: CaseDocuments | null;
  extractionInfoByDocument: Record<string, ExtractionInfo>;
  busy: string;
  onShowDocuments: (caseId: string) => void;
  onUpdateCaseRecords: (caseId: string, recordStoredFilenames: string[]) => void;
  onDeleteCase: (caseId: string) => void;
};

export function CasesPanel({
  cases,
  documents,
  selectedCaseId,
  caseDocuments,
  extractionInfoByDocument,
  busy,
  onShowDocuments,
  onUpdateCaseRecords,
  onDeleteCase,
}: CasesPanelProps) {
  const [pendingRecordsByCase, setPendingRecordsByCase] = useState<Record<string, string[]>>({});

  return (
    <section className="panel">
      <h2>Cases</h2>
      <div className="list">
        {cases.map((item) => (
          <article className={`list-item case-item ${selectedCaseId === item.case_id ? "active" : ""}`} key={item.case_id}>
            <div className="case-row">
              <div className="case-select document-option-copy">
                <div className="document-option-heading">
                  <strong className="document-option-title">{item.title}</strong>
                  <span className="document-option-meta">
                    {item.procedure_stored_filenames.length} procedures · {item.record_stored_filenames.length} records · {item.reference_stored_filenames.length} references
                  </span>
                </div>
                <span className="document-option-meta">Created: {formatDateTime(item.created_at)}</span>
              </div>
              <div className="actions">
                <button
                  className="button button-small button-ghost"
                  onClick={() => onShowDocuments(item.case_id)}
                  disabled={busy === `case-docs:${item.case_id}`}
                >
                  {caseDocuments?.case_id === item.case_id ? "Close documents" : "Documents"}
                </button>
                <button className="button button-small button-danger" onClick={() => onDeleteCase(item.case_id)}>
                  Delete
                </button>
              </div>
            </div>
            {caseDocuments?.case_id === item.case_id ? (
              <div className="case-documents-inline">
                <div>
                  <h3>Procedure files</h3>
                  {caseDocuments.procedure_documents.length > 0 ? (
                    <ul className="inline-doc-list">
                      {caseDocuments.procedure_documents.map((doc) => (
                        <li key={doc.stored_filename}>
                          <span className="document-option-copy">
                            <strong className="document-option-title">{doc.source_filename}</strong>
                            {extractionInfoByDocument[doc.stored_filename] ? (
                              <span className="document-option-meta">
                                Extraction: {extractionInfoByDocument[doc.stored_filename]?.provider} · {extractionInfoByDocument[doc.stored_filename]?.model}
                              </span>
                            ) : null}
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="empty-state">No procedure files.</p>
                  )}
                </div>
                <div>
                  <h3>Record files</h3>
                  {caseDocuments.record_documents.length > 0 ? (
                    <ul className="inline-doc-list">
                      {caseDocuments.record_documents.map((doc) => (
                        <li key={doc.stored_filename}>
                          <span className="document-option-copy">
                            <strong className="document-option-title">{doc.source_filename}</strong>
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="empty-state">No record files.</p>
                  )}
                  {(() => {
                    const caseDocumentIds = new Set([
                      ...caseDocuments.procedure_documents.map((doc) => doc.stored_filename),
                      ...caseDocuments.record_documents.map((doc) => doc.stored_filename),
                      ...caseDocuments.reference_documents.map((doc) => doc.stored_filename),
                    ]);
                    const availableRecordDocs = documents.filter(
                      (doc) =>
                        !caseDocumentIds.has(doc.stored_filename)
                        && doc.document_type !== "procedure"
                        && doc.document_type !== "reference",
                    );
                    const pending = pendingRecordsByCase[item.case_id] ?? [];
                    if (availableRecordDocs.length === 0) {
                      return null;
                    }
                    return (
                      <div className="stack">
                        <h3>Add record files</h3>
                        {availableRecordDocs.map((doc) => (
                          <label className="check" key={doc.stored_filename}>
                            <input
                              type="checkbox"
                              checked={pending.includes(doc.stored_filename)}
                              onChange={(event) =>
                                setPendingRecordsByCase((current) => ({
                                  ...current,
                                  [item.case_id]: event.target.checked
                                    ? [...pending, doc.stored_filename]
                                    : pending.filter((entry) => entry !== doc.stored_filename),
                                }))
                              }
                            />
                            <span className="document-option-copy">
                              <strong className="document-option-title">{doc.source_filename}</strong>
                            </span>
                          </label>
                        ))}
                        <button
                          className="button button-small"
                          type="button"
                          disabled={pending.length === 0 || busy === `update-case-records:${item.case_id}`}
                          onClick={() =>
                            onUpdateCaseRecords(
                              item.case_id,
                              [
                                ...caseDocuments.record_documents.map((doc) => doc.stored_filename),
                                ...pending,
                              ],
                            )
                          }
                        >
                          {busy === `update-case-records:${item.case_id}` ? "Updating..." : "Add selected records"}
                        </button>
                      </div>
                    );
                  })()}
                </div>
                <div>
                  <h3>Reference files</h3>
                  {caseDocuments.reference_documents.length > 0 ? (
                    <ul className="inline-doc-list">
                      {caseDocuments.reference_documents.map((doc) => (
                        <li key={doc.stored_filename}>
                          <span className="document-option-copy">
                            <strong className="document-option-title">{doc.source_filename}</strong>
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="empty-state">No reference files.</p>
                  )}
                </div>
              </div>
            ) : null}
          </article>
        ))}
      </div>
    </section>
  );
}
