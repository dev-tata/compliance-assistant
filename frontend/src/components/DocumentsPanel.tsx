import { useState } from "react";

import type { DocumentRecord } from "../types";
import { documentTypeLabels, languageLabels } from "../constants";
import { formatDateTime } from "../utils/formatDateTime";

function formatLanguage(value: DocumentRecord["language"]) {
  if (!value) return "unknown";
  return languageLabels[value] ?? value;
}

function formatDocumentType(value: DocumentRecord["document_type"]) {
  if (!value) return "Untyped";
  return documentTypeLabels[value] ?? value;
}

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

type ExtractionInfo = {
  provider: string;
  model: string;
} | null;

type DocumentsPanelProps = {
  documents: DocumentRecord[];
  extractionInfoByDocument: Record<string, ExtractionInfo>;
  busy: string;
  onRefresh: () => void;
  onParseDocument: (storedFilename: string) => void;
  onViewOriginal: (storedFilename: string) => void;
  onCheckRequirements: (storedFilename: string) => void;
  onToggleDocumentFreeze: (storedFilename: string, frozen: boolean) => void;
  onDeleteDocument: (storedFilename: string) => void;
};

export function DocumentsPanel({
  documents,
  extractionInfoByDocument,
  busy,
  onRefresh,
  onParseDocument,
  onViewOriginal,
  onCheckRequirements,
  onToggleDocumentFreeze,
  onDeleteDocument,
}: DocumentsPanelProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const sortedDocuments = [...documents].sort((left, right) => {
    const leftTime = left.created_at ? new Date(left.created_at).getTime() : 0;
    const rightTime = right.created_at ? new Date(right.created_at).getTime() : 0;
    return rightTime - leftTime;
  });
  const visibleDocuments = sortedDocuments.filter((doc) => {
    if (!searchQuery.trim()) {
      return true;
    }

    const extractionInfo = extractionInfoByDocument[doc.stored_filename];
    return matchesSearchFields(
      [
        extractionInfo?.provider ?? "",
        extractionInfo?.model ?? "",
        formatDocumentType(doc.document_type),
        doc.group_id ?? "",
        formatLanguage(doc.language),
      ],
      searchQuery,
    );
  });

  return (
    <section className="panel panel-wide">
      <div className="panel-head">
        <div className="panel-head-inline">
          <h2>Documents</h2>
          <button className="button button-ghost button-small" onClick={onRefresh}>
            Refresh
          </button>
        </div>
      </div>
      <div className="panel-search">
        <input
          type="search"
          value={searchQuery}
          onChange={(event) => setSearchQuery(event.target.value)}
          placeholder="Search"
        />
      </div>
      <div className="list">
        {visibleDocuments.length > 0 ? visibleDocuments.map((doc) => (
          <article className="list-item" key={doc.stored_filename}>
            <div className="document-option-copy">
              <strong className="document-option-title">{doc.source_filename}</strong>
              <span className="document-option-meta">
                {formatDocumentType(doc.document_type)} · {doc.group_id ?? "no-group"} · {formatLanguage(doc.language)}
                {" · "}Created: {formatDateTime(doc.created_at)}
                {extractionInfoByDocument[doc.stored_filename]
                  ? ` · Extraction: ${extractionInfoByDocument[doc.stored_filename]?.provider} · ${extractionInfoByDocument[doc.stored_filename]?.model}`
                  : ""}
              </span>
            </div>
            <div className="actions">
              {doc.document_type === "procedure" || doc.document_type === "reference" ? (
                <>
                  <button
                    className={`button button-small ${doc.frozen ? "button-unfreeze" : "button-ghost"}`}
                    onClick={() => onToggleDocumentFreeze(doc.stored_filename, !doc.frozen)}
                    disabled={busy === `freeze-doc:${doc.stored_filename}`}
                  >
                    {busy === `freeze-doc:${doc.stored_filename}`
                      ? (doc.frozen ? "Unfreezing..." : "Freezing...")
                      : (doc.frozen ? "Unfreeze" : "Freeze")}
                  </button>
                  {doc.document_type === "procedure" ? (
                    <button className="button button-small button-ghost" onClick={() => onCheckRequirements(doc.stored_filename)}>
                      Check requirements
                    </button>
                  ) : null}
                </>
              ) : null}
              <button className="button button-small" onClick={() => onParseDocument(doc.stored_filename)}>
                Show parsed
              </button>
              <button className="button button-small button-ghost" onClick={() => onViewOriginal(doc.stored_filename)}>
                Show original
              </button>
              <button
                className="button button-small button-danger"
                onClick={() => onDeleteDocument(doc.stored_filename)}
                disabled={(doc.document_type === "procedure" || doc.document_type === "reference") && doc.frozen}
                title={
                  (doc.document_type === "procedure" || doc.document_type === "reference") && doc.frozen
                    ? `Unfreeze this ${doc.document_type} before deleting it.`
                    : undefined
                }
              >
                Delete
              </button>
            </div>
          </article>
        )) : (
          <p className="empty-state">
            {documents.length > 0 ? "No documents match the current search." : "No documents uploaded yet."}
          </p>
        )}
      </div>
    </section>
  );
}
