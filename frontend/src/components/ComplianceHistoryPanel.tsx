import { useState } from "react";

import type { CaseRecord, ComplianceSummary } from "../types";
import { formatDateTime } from "../utils/formatDateTime";

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
  complianceHistory: ComplianceSummary[];
  selectedComplianceFile: string;
  busy: string;
  onSelectCompliance: (caseId: string, fileName: string) => void;
  onDeleteCompliance: (caseId: string, fileName: string) => void;
};

export function ComplianceHistoryPanel({
  cases,
  complianceHistory,
  selectedComplianceFile,
  busy,
  onSelectCompliance,
  onDeleteCompliance,
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

  return (
    <section className="panel">
      <h2>Compliances</h2>
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
              <button
                className="history-main"
                onClick={() => onSelectCompliance(item.case_id, item.file_name)}
                disabled={busy === `compliance-history:${item.file_name}`}
              >
                <span className="history-case">{caseTitleById.get(item.case_id) ?? item.case_id}</span>
                <strong>{item.provider} · {item.model}</strong>
                <p>Method: {item.method}</p>
                <p>Created: {formatDateTime(item.created_at)}</p>
                <p>{item.overall_assessment}</p>
                <code>{item.file_name}</code>
              </button>
              <button
                className="button button-small button-danger"
                onClick={() => onDeleteCompliance(item.case_id, item.file_name)}
                disabled={busy === `delete-compliance:${item.file_name}`}
              >
                Delete
              </button>
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
