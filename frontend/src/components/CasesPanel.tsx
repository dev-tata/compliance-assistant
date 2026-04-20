import type { CaseDocuments, CaseRecord } from "../types";
import { formatDateTime } from "../utils/formatDateTime";

type ExtractionInfo = {
  provider: string;
  model: string;
} | null;

type CasesPanelProps = {
  cases: CaseRecord[];
  selectedCaseId: string;
  caseDocuments: CaseDocuments | null;
  extractionInfoByDocument: Record<string, ExtractionInfo>;
  busy: string;
  onShowDocuments: (caseId: string) => void;
  onDeleteCase: (caseId: string) => void;
};

export function CasesPanel({
  cases,
  selectedCaseId,
  caseDocuments,
  extractionInfoByDocument,
  busy,
  onShowDocuments,
  onDeleteCase,
}: CasesPanelProps) {
  return (
    <section className="panel">
      <h2>Cases</h2>
      <div className="list">
        {cases.map((item) => (
          <article className={`list-item case-item ${selectedCaseId === item.case_id ? "active" : ""}`} key={item.case_id}>
            <div className="case-row">
              <div className="case-select">
                <strong>{item.title}</strong>
                <span>
                  {item.procedure_stored_filenames.length} procedures · {item.record_stored_filenames.length} records · {item.reference_stored_filenames.length} references
                </span>
                <span>Created: {formatDateTime(item.created_at)}</span>
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
                          <span>{doc.source_filename}</span>
                          {extractionInfoByDocument[doc.stored_filename] ? (
                            <div className="empty-state">
                              Extraction: {extractionInfoByDocument[doc.stored_filename]?.provider} · {extractionInfoByDocument[doc.stored_filename]?.model}
                            </div>
                          ) : null}
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
                        <li key={doc.stored_filename}>{doc.source_filename}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="empty-state">No record files.</p>
                  )}
                </div>
                <div>
                  <h3>Reference files</h3>
                  {caseDocuments.reference_documents.length > 0 ? (
                    <ul className="inline-doc-list">
                      {caseDocuments.reference_documents.map((doc) => (
                        <li key={doc.stored_filename}>{doc.source_filename}</li>
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
