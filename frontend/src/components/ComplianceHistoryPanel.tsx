import { useState } from "react";

import type { CaseDocuments, CaseRecord, ComplianceSummary, DocumentRecord } from "../types";
import { formatDateTime } from "../utils/formatDateTime";
import { formatMethodLabel } from "../utils/formatMethodLabel";

function normalizeSearchValue(value: string) {
  return value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function matchesSearchFields(fields: string[], query: string) {
  const normalizedQuery = normalizeSearchValue(query);
  if (!normalizedQuery) {
    return true;
  }

  const normalizedFields = fields
    .map((field) => normalizeSearchValue(field))
    .filter(Boolean);

  return normalizedQuery
    .split(" ")
    .every((token) => normalizedFields.some((field) => field.includes(token)));
}

type ComplianceHistoryPanelProps = {
  cases: CaseRecord[];
  caseDocuments: CaseDocuments | null;
  complianceHistory: ComplianceSummary[];
  documents: DocumentRecord[];
  openDocumentsFile: string;
  selectedComplianceFile: string;
  busy: string;
  extractionInfoByDocument: Record<string, { provider: string; model: string } | null>;
  onSelectCompliance: (caseId: string, fileName: string) => void;
  onShowDocuments: (caseId: string, fileName: string) => void;
  onDeleteCompliance: (caseId: string, fileName: string) => void;
  onDeleteAllCompliances: () => void;
};

export function ComplianceHistoryPanel({
  cases,
  caseDocuments,
  complianceHistory,
  documents,
  openDocumentsFile,
  selectedComplianceFile,
  busy,
  extractionInfoByDocument,
  onSelectCompliance,
  onShowDocuments,
  onDeleteCompliance,
  onDeleteAllCompliances,
}: ComplianceHistoryPanelProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const caseTitleById = new Map(cases.map((item) => [item.case_id, item.title]));
  const sortedComplianceHistory = [...complianceHistory].sort((left, right) => {
    const leftTime = left.created_at ? new Date(left.created_at).getTime() : 0;
    const rightTime = right.created_at ? new Date(right.created_at).getTime() : 0;
    return rightTime - leftTime;
  });
  const visibleComplianceHistory = sortedComplianceHistory.filter((item) => {
    if (!searchQuery.trim()) {
      return true;
    }

    return matchesSearchFields([item.provider, item.model], searchQuery);
  });

  function renderDocumentList(documents: DocumentRecord[], emptyLabel: string, showExtraction: boolean) {
    if (documents.length === 0) {
      return <p className="empty-state">{emptyLabel}</p>;
    }

    return (
      <ul className="inline-doc-list">
        {documents.map((doc) => (
          <li key={doc.stored_filename}>
            <span className="document-option-copy">
              <strong className="document-option-title">{doc.source_filename}</strong>
              {showExtraction && extractionInfoByDocument[doc.stored_filename] ? (
                <span className="document-option-meta">
                  Extraction: {extractionInfoByDocument[doc.stored_filename]?.provider} · {extractionInfoByDocument[doc.stored_filename]?.model}
                </span>
              ) : null}
            </span>
          </li>
        ))}
      </ul>
    );
  }

  return (
    <section className="panel compliance-history-panel">
      <div className="panel-head compliance-history-head">
        <h2>Compliances</h2>
        <button
          className="button button-small button-danger compliance-history-delete-all"
          onClick={onDeleteAllCompliances}
          disabled={complianceHistory.length === 0 || busy === "delete-all-compliances"}
          type="button"
        >
          Delete all
        </button>
      </div>
      <div className="panel-search">
        <input
          type="search"
          value={searchQuery}
          onChange={(event) => setSearchQuery(event.target.value)}
          placeholder="Search"
        />
      </div>
      <div className="history-list">
        {visibleComplianceHistory.length > 0 ? (
          visibleComplianceHistory.map((item) => (
            <article className={`history-item ${selectedComplianceFile === item.file_name ? "active" : ""}`} key={item.file_name}>
              {(() => {
                const isDocumentsOpen = openDocumentsFile === item.file_name && caseDocuments?.case_id === item.case_id;
                const referenceDocuments = item.reference_stored_filenames
                  .map((storedFilename) => documents.find((doc) => doc.stored_filename === storedFilename))
                  .filter((doc): doc is DocumentRecord => Boolean(doc));

                return (
                  <>
                    <div className="history-main">
                      <div className="history-title-row">
                        <span className="history-case">{caseTitleById.get(item.case_id) ?? item.case_id}</span>
                        <span className="document-option-meta">Completed <strong>{item.completion_percent}%</strong></span>
                        <span className="history-status-group">
                          <span className="history-status-chip history-status-chip-satisfied">SATISFIED</span>
                          <strong className="history-status-count">{item.satisfied_count}</strong>
                        </span>
                        <span className="history-status-group">
                          <span className="history-status-chip history-status-chip-partial">PARTIAL</span>
                          <strong className="history-status-count">{item.partial_count}</strong>
                        </span>
                        <span className="history-status-group">
                          <span className="history-status-chip history-status-chip-not-satisfied">NOT_SATISFIED</span>
                          <strong className="history-status-count">{item.not_satisfied_count}</strong>
                        </span>
                      </div>
                      <div className="history-meta-row document-option-meta">
                        <span>
                          {item.provider} · <strong>{item.model}</strong> · Method: <strong>{formatMethodLabel(item.method)}</strong> · Created: {formatDateTime(item.created_at)} · <code>{item.file_name}</code>
                        </span>
                      </div>
                    </div>
                    <div className="actions history-actions">
                      <button
                        className="button button-small button-ghost"
                        onClick={() => onShowDocuments(item.case_id, item.file_name)}
                        disabled={busy === `case-docs:${item.case_id}`}
                      >
                        {isDocumentsOpen ? "Close documents" : "Documents"}
                      </button>
                      <button
                        className="button button-small"
                        onClick={() => onSelectCompliance(item.case_id, item.file_name)}
                        disabled={busy === `compliance-history:${item.file_name}`}
                      >
                        Results
                      </button>
                      <button
                        className="button button-small button-danger"
                        onClick={() => onDeleteCompliance(item.case_id, item.file_name)}
                        disabled={busy === `delete-compliance:${item.file_name}`}
                      >
                        Delete
                      </button>
                    </div>
                    {isDocumentsOpen ? (
                      <div className="case-documents-inline">
                        <div>
                          <h3>Procedure files</h3>
                          {renderDocumentList(caseDocuments.procedure_documents, "No procedure files.", true)}
                        </div>
                        <div>
                          <h3>Record files</h3>
                          {renderDocumentList(caseDocuments.record_documents, "No record files.", false)}
                        </div>
                        <div>
                          <h3>Reference files</h3>
                          {renderDocumentList(referenceDocuments, "No reference files.", false)}
                        </div>
                      </div>
                    ) : null}
                  </>
                );
              })()}
            </article>
          ))
        ) : (
          <p className="empty-state">
            {complianceHistory.length > 0 ? "No compliances match the current search." : "No saved compliances yet."}
          </p>
        )}
      </div>
    </section>
  );
}
