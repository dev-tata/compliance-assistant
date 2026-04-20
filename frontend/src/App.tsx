import { FormEvent, useEffect, useState } from "react";

import {
  createCase,
  deleteCaseComplianceResult,
  deleteCase,
  deleteDocument,
  extractDocumentDeliverables,
  getCaseDocuments,
  getDocumentFileUrl,
  getCaseComplianceResult,
  getLatestDocumentDeliverables,
  getParsedDocumentUrl,
  listLLMProviders,
  listAllCompliances,
  listCases,
  listDocuments,
  runCompliance,
  updateLatestDocumentDeliverables,
  uploadDocument,
} from "./api";
import { AppHeader } from "./components/AppHeader";
import { CasesPanel } from "./components/CasesPanel";
import { ComplianceHistoryPanel } from "./components/ComplianceHistoryPanel";
import { CompliancePanel } from "./components/CompliancePanel";
import { CreateCasePanel } from "./components/CreateCasePanel";
import { DocumentsPanel } from "./components/DocumentsPanel";
import { UploadPanel } from "./components/UploadPanel";
import type {
  CaseRecord,
  CaseDocuments,
  ComplianceMethod,
  ComplianceResponse,
  ComplianceSummary,
  DeliverableItem,
  DeliverableExtractionResponse,
  DocumentLanguage,
  DocumentRecord,
  DocumentType,
  LLMProviderDescriptor,
  SelectedDeliverablesByDocument,
} from "./types";
import { formatDateTime } from "./utils/formatDateTime";

const FALLBACK_LLM_PROVIDERS: LLMProviderDescriptor[] = [
  {
    key: "openai",
    label: "OpenAI",
    default_model: "gpt-5.4-nano",
    description: "Hosted OpenAI chat completion models.",
    endpoint_mode: "remote",
  },
  {
    key: "gemini",
    label: "Gemini",
    default_model: "gemini-3.1-flash-lite-preview",
    description: "Hosted Google Gemini models.",
    endpoint_mode: "remote",
  },
];

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderList(items: string[]): string {
  if (items.length === 0) {
    return "<li>None</li>";
  }

  return items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function renderLinkedRows(
  rows: ComplianceResponse["analysis"]["linked_rows"],
  procedureToRecord: ComplianceResponse["analysis"]["procedure_to_record"] | ComplianceResponse["analysis"]["findings"],
): string {
  if (rows.length === 0) {
    return `
      <tr>
        <td>None</td>
        <td>None</td>
        <td>None</td>
        <td>None</td>
        <td>None</td>
        <td>None</td>
      </tr>
    `;
  }

  return rows.map((row, index) => {
    const confidence = procedureToRecord[index]?.confidence;
    const confidenceLabel = typeof confidence === "number"
      ? `${(confidence * 100).toFixed(1)}%`
      : "&mdash;";
    return `
    <tr>
      <td>${index + 1}</td>
      <td>${escapeHtml(row.requirement) || "&mdash;"}</td>
      <td>${confidenceLabel}</td>
      <td>${escapeHtml(row.status) || "&mdash;"}</td>
      <td>${escapeHtml(row.gap) || "&mdash;"}</td>
      <td>${escapeHtml(row.recommendation) || "&mdash;"}</td>
    </tr>
  `;
  }).join("");
}

function computeDeliverableConfidence(item: DeliverableItem): number {
  let score = 0.28;
  const requirementText = item.requirement_text.trim().replace(/\s+/g, " ");
  const sourceQuote = item.source_quote.trim().replace(/\s+/g, " ");

  if (item.mandatory) score += 0.1;
  if (item.section_label.trim()) score += 0.07;
  if (item.heading_title.trim()) score += 0.07;
  if (item.source_document.trim()) score += 0.05;
  if (/\b(shall|must|required|needs to)\b/i.test(requirementText)) score += 0.12;
  if (sourceQuote) score += 0.1;

  const reqWords = new Set((requirementText.toLowerCase().match(/[a-z0-9]+/g) ?? []));
  const quoteWords = new Set((sourceQuote.toLowerCase().match(/[a-z0-9]+/g) ?? []));
  if (reqWords.size > 0 && quoteWords.size > 0) {
    let overlapCount = 0;
    for (const word of reqWords) {
      if (quoteWords.has(word)) overlapCount += 1;
    }
    const overlap = overlapCount / reqWords.size;
    score += Math.min(0.12, overlap * 0.12);
  }

  const wordCount = requirementText ? requirementText.split(/\s+/).length : 0;
  if (wordCount >= 6 && wordCount <= 30) score += 0.08;
  else if (wordCount > 0 && (wordCount < 4 || wordCount > 45)) score -= 0.08;

  return Math.max(0, Math.min(0.98, Number(score.toFixed(4))));
}

function openComplianceWindow(compliance: ComplianceResponse, caseTitle?: string): void {
  const popup = window.open("", "_blank", "width=1280,height=900");
  if (!popup) {
    throw new Error("Popup blocked");
  }

  const procedureToRecord = (compliance.analysis.procedure_to_record?.length
    ? compliance.analysis.procedure_to_record
    : compliance.analysis.findings);

  const requirementEvaluations = procedureToRecord.map((finding, index) => `
    <article class="finding requirement-evaluation">
      <header>
        <div>
          <div class="requirement-label">Requirement ${index + 1}</div>
          <strong>${escapeHtml(finding.requirement)}</strong>
        </div>
        <span class="status status-${escapeHtml(finding.status)}">${escapeHtml(finding.status)}</span>
      </header>
      <div class="finding-metrics">
        <span><strong>Sources:</strong> ${escapeHtml(finding.source_documents.join(", ") || "none")}</span>
        <span><strong>Evidence strength:</strong> ${(finding.evidence_strength * 100).toFixed(1)}% · <strong>Weight:</strong> ${finding.weight.toFixed(1)}</span>
      </div>
      <ul>${renderList(finding.evidence)}</ul>
    </article>
  `).join("");

  popup.document.open();
  popup.document.write(`
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <title>Compliance result</title>
        <style>
          body {
            margin: 0;
            padding: 20px;
            font-family: Georgia, "Times New Roman", serif;
            color: #231b12;
            background: #f3efe4;
          }
          h1, h2, h3, h4, p { margin-top: 0; }
          h1 { white-space: nowrap; }
          .head, .score-grid, .findings { display: grid; gap: 12px; }
          .head { grid-template-columns: 1fr; align-items: start; }
          .head-main p { margin-bottom: 18px; }
          .case-line { white-space: nowrap; }
          .path-line { margin-top: -6px; }
          .code { color: #8a3b12; font-family: ui-monospace, Consolas, monospace; word-break: break-all; }
          .assessment, .score-card, .result-list, .finding {
            padding: 12px 14px;
            border: 1px solid #d8c8a8;
            border-radius: 8px;
            background: rgba(255,255,255,0.72);
          }
          .score-grid { grid-template-columns: repeat(5, minmax(0, 1fr)); }
          .score-grid { margin-bottom: 18px; }
          .score-card span, .result-list li, .finding p, .finding li { color: #6f6252; }
          .section-block { margin-top: 28px; }
          .gap-table-wrap {
            margin-top: 18px;
            overflow-x: auto;
          }
          .gap-table {
            width: 100%;
            border-collapse: collapse;
            border: 1px solid #d8c8a8;
            background: rgba(255,255,255,0.72);
            border-radius: 8px;
            overflow: hidden;
          }
          .gap-table th,
          .gap-table td {
            padding: 12px 14px;
            border-bottom: 1px solid #e6d8bd;
            text-align: left;
            vertical-align: top;
          }
          .gap-table th {
            background: #efe3c9;
            color: #5a3a16;
          }
          .gap-table td {
            color: #6f6252;
          }
          .gap-table tr:last-child td {
            border-bottom: none;
          }
          .finding header { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
          .requirement-label {
            margin-bottom: 6px;
            color: #8a3b12;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
          }
          .finding-metrics {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px 16px;
            margin-bottom: 12px;
            color: #6f6252;
          }
          .status { border-radius: 4px; padding: 3px 8px; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.05em; }
          .status-satisfied { background: #d7f1df; color: #2d6a4f; }
          .status-partial { background: #fff1cf; color: #b7791f; }
          .status-not_satisfied { background: #ffdedd; color: #9b2226; }
          @media (max-width: 900px) {
            .head, .score-grid { grid-template-columns: 1fr; }
            .finding-metrics { grid-template-columns: 1fr; }
          }
        </style>
      </head>
      <body>
        <div class="head">
          <div class="head-main">
            <h1>Compliance result</h1>
            <p class="code path-line">${escapeHtml(compliance.saved_at)}</p>
            ${caseTitle ? `<p class="case-line">Case: ${escapeHtml(caseTitle)}</p>` : ""}
            <p>${escapeHtml(`${compliance.method} · ${compliance.compliance_provider} · ${compliance.compliance_model}`)}</p>
            <p>Created: ${escapeHtml(formatDateTime(compliance.created_at))}</p>
          </div>
        </div>
        <p class="assessment">${escapeHtml(compliance.analysis.overall_assessment)}</p>
        <div class="score-grid">
          <div class="score-card"><span>M1 Binary</span><strong> ${compliance.scores.m1_binary_rule_score.toFixed(4)}</strong></div>
          <div class="score-card"><span>M2 Ordinal</span><strong> ${compliance.scores.m2_ordinal_score.toFixed(4)}</strong></div>
          <div class="score-card"><span>M3 Weighted</span><strong> ${compliance.scores.m3_evidence_weighted_score.toFixed(4)}</strong></div>
          <div class="score-card"><span>M4 Confidence</span><strong> ${compliance.scores.m4_confidence_aware_score.toFixed(4)}</strong></div>
          <div class="score-card"><span>M5 Grounding</span><strong> ${compliance.scores.m5_grounding_score.toFixed(4)}</strong></div>
        </div>
        <div class="gap-table-wrap">
          <table class="gap-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Requirement</th>
                <th>Confidence</th>
                <th>Status</th>
                <th>Gap</th>
                <th>Recommendation</th>
              </tr>
            </thead>
            <tbody>${renderLinkedRows(compliance.analysis.linked_rows ?? [], procedureToRecord)}</tbody>
          </table>
        </div>
        <section class="section-block">
          <h2>Grounded Evidence (Procedure → Record)</h2>
          <div class="findings">${requirementEvaluations}</div>
        </section>
      </body>
    </html>
  `);
  popup.document.close();
  popup.focus();
}

export default function App() {
  const views = [
    { id: "upload", label: "Upload" },
    { id: "documents", label: "Documents" },
    { id: "create-case", label: "Create case" },
    { id: "cases", label: "Cases" },
    { id: "run-compliance", label: "Run compliance" },
    { id: "compliances", label: "Compliances" },
  ] as const;

  type ActiveView = (typeof views)[number]["id"];

  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [cases, setCases] = useState<CaseRecord[]>([]);
  const [selectedProcedures, setSelectedProcedures] = useState<string[]>([]);
  const [selectedRecords, setSelectedRecords] = useState<string[]>([]);
  const [selectedReferences, setSelectedReferences] = useState<string[]>([]);
  const [title, setTitle] = useState("");
  const [notes, setNotes] = useState("");
  const [providers, setProviders] = useState<LLMProviderDescriptor[]>(FALLBACK_LLM_PROVIDERS);
  const [provider, setProvider] = useState(FALLBACK_LLM_PROVIDERS[0]?.key ?? "openai");
  const [model, setModel] = useState(FALLBACK_LLM_PROVIDERS[0]?.default_model ?? "gpt-5.4-nano");
  const [complianceMethod, setComplianceMethod] = useState<ComplianceMethod>("non_rag");
  const [instructions, setInstructions] = useState("");
  const [selectedCaseId, setSelectedCaseId] = useState("");
  const [caseDocuments, setCaseDocuments] = useState<CaseDocuments | null>(null);
  const [latestCompliance, setLatestCompliance] = useState<ComplianceResponse | null>(null);
  const [complianceHistory, setComplianceHistory] = useState<ComplianceSummary[]>([]);
  const [selectedComplianceFile, setSelectedComplianceFile] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadType, setUploadType] = useState<DocumentType>("procedure");
  const [uploadLanguage, setUploadLanguage] = useState<DocumentLanguage>("en");
  const [extractOnUpload, setExtractOnUpload] = useState(true);
  const [extractionProvider, setExtractionProvider] = useState(FALLBACK_LLM_PROVIDERS[0]?.key ?? "openai");
  const [extractionModel, setExtractionModel] = useState(FALLBACK_LLM_PROVIDERS[0]?.default_model ?? "gpt-5.4-nano");
  const [groupId, setGroupId] = useState("");
  const [selectedDeliverablesByDocument, setSelectedDeliverablesByDocument] = useState<SelectedDeliverablesByDocument>({});
  const [deliverablePicker, setDeliverablePicker] = useState<DeliverableExtractionResponse | null>(null);
  const [editableDeliverables, setEditableDeliverables] = useState<DeliverableItem[]>([]);
  const [extractionInfoByDocument, setExtractionInfoByDocument] = useState<Record<string, { provider: string; model: string } | null>>({});
  const [activeView, setActiveView] = useState<ActiveView>("documents");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    void refreshAll();
  }, []);

  useEffect(() => {
    void refreshComplianceHistory();
  }, []);

  useEffect(() => {
    void refreshProviders();
  }, []);

  async function refreshProviders() {
    try {
      const catalog = await listLLMProviders();
      if (catalog.length === 0) {
        return;
      }
      const nextProvider = catalog.some((item) => item.key === provider) ? provider : catalog[0].key;
      const nextExtractionProvider = catalog.some((item) => item.key === extractionProvider)
        ? extractionProvider
        : catalog[0].key;
      const nextProviderModel = catalog.find((item) => item.key === nextProvider)?.default_model ?? "";
      const nextExtractionModel = catalog.find((item) => item.key === nextExtractionProvider)?.default_model ?? "";

      setProviders(catalog);
      setProvider(nextProvider);
      setExtractionProvider(nextExtractionProvider);
      if (!model || nextProvider !== provider) {
        setModel(nextProviderModel);
      }
      if (!extractionModel || nextExtractionProvider !== extractionProvider) {
        setExtractionModel(nextExtractionModel);
      }
    } catch {
      setProviders(FALLBACK_LLM_PROVIDERS);
    }
  }

  async function refreshAll() {
    setError("");
    const [loadedDocuments, loadedCases] = await Promise.all([listDocuments(), listCases()]);
    setDocuments(loadedDocuments);
    setCases(loadedCases);
    await refreshExtractionInfo(loadedDocuments);
    await refreshComplianceHistory();
    if (!selectedCaseId && loadedCases[0]) {
      setSelectedCaseId(loadedCases[0].case_id);
    }
  }

  async function refreshExtractionInfo(loadedDocuments: DocumentRecord[]) {
    const eligibleDocuments = loadedDocuments.filter(
      (doc) => doc.document_type === "procedure" || doc.document_type === "reference",
    );

    const infoEntries = await Promise.all(
      eligibleDocuments.map(async (doc) => {
        try {
          const result = await getLatestDocumentDeliverables(doc.stored_filename);
          return [
            doc.stored_filename,
            {
              provider: result.extraction_provider,
              model: result.extraction_model,
            },
          ] as const;
        } catch {
          return [doc.stored_filename, null] as const;
        }
      }),
    );

    setExtractionInfoByDocument(Object.fromEntries(infoEntries));
  }

  async function refreshComplianceHistory() {
    try {
      const history = await listAllCompliances();
      setComplianceHistory(history);
      if (history.length > 0 && !history.some((item) => item.file_name === selectedComplianceFile)) {
        setSelectedComplianceFile(history[0].file_name);
      } else if (history.length === 0) {
        setSelectedComplianceFile("");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Compliance history failed");
    }
  }

  async function onSelectCompliance(caseId: string, fileName: string) {
    setBusy(`compliance-history:${fileName}`);
    setError("");
    try {
      const result = await getCaseComplianceResult(caseId, fileName);
      setLatestCompliance(result);
      setSelectedComplianceFile(fileName);
      setSelectedCaseId(caseId);
      openComplianceWindow(result, caseTitleById.get(caseId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Loading compliance failed");
    } finally {
      setBusy("");
    }
  }

  async function onShowCaseDocuments(caseId: string) {
    if (caseDocuments?.case_id === caseId) {
      setCaseDocuments(null);
      return;
    }
    setBusy(`case-docs:${caseId}`);
    setError("");
    try {
      const payload = await getCaseDocuments(caseId);
      setCaseDocuments(payload);
      setSelectedCaseId(caseId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Loading case documents failed");
    } finally {
      setBusy("");
    }
  }

  async function onUploadSubmit(event: FormEvent) {
    event.preventDefault();
    if (!uploadFile) return;
    setBusy("upload");
    setError("");
    try {
      const uploaded = await uploadDocument({
        file: uploadFile,
        documentType: uploadType,
        language: uploadLanguage,
        groupId: groupId.trim() || undefined,
      });
      if (uploadType === "procedure" && extractOnUpload) {
        await extractDocumentDeliverables({
          storedFilename: uploaded.stored_filename,
          provider: extractionProvider,
          model: extractionModel,
          method: "non_rag",
        });
      }
      setUploadFile(null);
      setGroupId("");
      await refreshAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy("");
    }
  }

  async function onCreateCase(event: FormEvent) {
    event.preventDefault();
    setBusy("case");
    setError("");
    try {
      const created = await createCase({
        title,
        procedureStoredFilenames: selectedProcedures,
        recordStoredFilenames: selectedRecords,
        referenceStoredFilenames: selectedReferences,
        notes,
      });
      setTitle("");
      setNotes("");
      setSelectedProcedures([]);
      setSelectedRecords([]);
      setSelectedReferences([]);
      setSelectedCaseId(created.case_id);
      setCaseDocuments(null);
      await refreshAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Case creation failed");
    } finally {
      setBusy("");
    }
  }

  async function onRunCompliance(event: FormEvent) {
    event.preventDefault();
    if (!selectedCaseId) return;
    setBusy("compliance");
    setError("");
    try {
      const result = await runCompliance({
        caseId: selectedCaseId,
        provider,
        model,
        method: complianceMethod,
        instructions,
        selectedDeliverablesByDocument,
      });
      setLatestCompliance(result);
      const fileName = result.saved_at.split("/").pop() ?? "";
      setSelectedComplianceFile(fileName);
      await refreshComplianceHistory();
      openComplianceWindow(result, caseTitleById.get(selectedCaseId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Compliance failed");
    } finally {
      setBusy("");
    }
  }

  async function onParseDocument(storedFilename: string) {
    setBusy(`parse:${storedFilename}`);
    setError("");
    try {
      const response = await fetch(getParsedDocumentUrl(storedFilename));
      if (!response.ok) {
        throw new Error(response.statusText || "Parse failed");
      }
      const parsed = await response.json();
      const previewWindow = window.open("", "_blank", "width=960,height=720");
      if (!previewWindow) {
        throw new Error("Popup blocked");
      }
      const parsedText = escapeHtml(JSON.stringify(parsed, null, 2));
      previewWindow.document.open();
      previewWindow.document.write(`
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <title>Parsed preview</title>
            <style>
              body {
                margin: 0;
                font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
                background: #1c1b1a;
                color: #f5f0e8;
              }
              pre {
                margin: 0;
                padding: 16px;
                white-space: pre-wrap;
                overflow-wrap: anywhere;
              }
            </style>
          </head>
          <body><pre>${parsedText}</pre></body>
        </html>
      `);
      previewWindow.document.close();
      previewWindow.focus();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Parse failed");
    } finally {
      setBusy("");
    }
  }

  async function onDeleteDocument(storedFilename: string) {
    setBusy(`delete-doc:${storedFilename}`);
    setError("");
    try {
      const document = documents.find((item) => item.stored_filename === storedFilename);
      await deleteDocument(storedFilename);
      setSelectedDeliverablesByDocument((current) => {
        const next = { ...current };
        delete next[storedFilename];
        if (document?.source_filename) {
          delete next[document.source_filename];
        }
        return next;
      });
      setCaseDocuments((current) =>
        current
          ? {
              ...current,
              procedure_documents: current.procedure_documents.filter((doc) => doc.stored_filename !== storedFilename),
              record_documents: current.record_documents.filter((doc) => doc.stored_filename !== storedFilename),
              reference_documents: current.reference_documents.filter((doc) => doc.stored_filename !== storedFilename),
            }
          : null,
      );
      await refreshAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    } finally {
      setBusy("");
    }
  }

  async function onCheckRequirements(storedFilename: string) {
    setBusy(`requirements:${storedFilename}`);
    setError("");
    try {
      const result = await getLatestDocumentDeliverables(storedFilename);
      setDeliverablePicker(result);
      setEditableDeliverables(result.deliverables);
      const documentKey = result.source_filename ?? storedFilename;
      setSelectedDeliverablesByDocument((current) => {
        if (current[documentKey]) {
          return current;
        }
        return {
          ...current,
          [documentKey]: result.deliverables.map((item) => item.requirement_text),
        };
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Loading deliverables failed");
    } finally {
      setBusy("");
    }
  }

  function onDeliverableFieldChange(
    index: number,
    field: keyof DeliverableItem,
    value: string | boolean | number,
  ) {
    setEditableDeliverables((current) =>
      current.map((item, itemIndex) => {
        if (itemIndex !== index) {
          return item;
        }
        const updated = {
          ...item,
          [field]: value,
        };
        return {
          ...updated,
          confidence: computeDeliverableConfidence(updated),
        };
      }),
    );
  }

  function onAddDeliverable() {
    const sourceDocument =
      deliverablePicker?.source_filename ??
      deliverablePicker?.document_stored_filename ??
      "";
    setEditableDeliverables((current) => [
      ...current,
      {
        section_label: "manual",
        heading_title: "Manual requirement",
        requirement_text: "",
        requirement_type: "recorded_information",
        mandatory: true,
        source_quote: "",
        source_document: sourceDocument,
        required_by_procedure: true,
        confidence: 0,
      },
    ]);
  }

  function onDeleteDeliverable(index: number) {
    setEditableDeliverables((current) => current.filter((_, itemIndex) => itemIndex !== index));
  }

  async function onSaveDeliverables() {
    if (!deliverablePicker?.document_stored_filename) {
      return;
    }

    setBusy("save-deliverables");
    setError("");
    try {
      const normalizedDeliverables = editableDeliverables.map((item) => {
        const requirementText = item.requirement_text.trim();
        const sourceQuote = item.source_quote.trim();
        return {
          ...item,
          section_label: item.section_label.trim() || "manual",
          heading_title: item.heading_title.trim() || "Manual requirement",
          requirement_text: requirementText,
          source_quote: sourceQuote === requirementText ? "" : sourceQuote,
          source_document: item.source_document.trim()
            || deliverablePicker.source_filename
            || deliverablePicker.document_stored_filename,
        };
      }).filter((item) => item.requirement_text);

      const result = await updateLatestDocumentDeliverables({
        storedFilename: deliverablePicker.document_stored_filename,
        deliverables: normalizedDeliverables,
      });
      setDeliverablePicker(result);
      setEditableDeliverables(result.deliverables);
      const documentKey = result.source_filename ?? result.document_stored_filename ?? "";
      setSelectedDeliverablesByDocument((current) => ({
        ...current,
        [documentKey]: result.deliverables.map((item) => item.requirement_text),
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Saving deliverables failed");
    } finally {
      setBusy("");
    }
  }

  function onViewOriginal(storedFilename: string) {
    window.open(getDocumentFileUrl(storedFilename), "_blank", "noopener,noreferrer");
  }

  async function onDeleteCase(caseId: string) {
    setBusy(`delete-case:${caseId}`);
    setError("");
    try {
      await deleteCase(caseId);
      if (selectedCaseId === caseId) {
        setSelectedCaseId("");
        setCaseDocuments(null);
        setLatestCompliance(null);
        setComplianceHistory([]);
        setSelectedComplianceFile("");
      }
      setCaseDocuments((current) => (current?.case_id === caseId ? null : current));
      await refreshAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Case delete failed");
    } finally {
      setBusy("");
    }
  }

  async function onDeleteCompliance(caseId: string, fileName: string) {
    setBusy(`delete-compliance:${fileName}`);
    setError("");
    try {
      await deleteCaseComplianceResult(caseId, fileName);
      if (selectedComplianceFile === fileName) {
        setSelectedComplianceFile("");
        if (latestCompliance?.saved_at.split("/").pop() === fileName) {
          setLatestCompliance(null);
        }
      }
      await refreshComplianceHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Compliance delete failed");
    } finally {
      setBusy("");
    }
  }

  const procedureDocs = documents.filter((doc) => doc.document_type === "procedure");
  const referenceDocs = documents.filter((doc) => doc.document_type === "reference");
  const recordDocs = documents.filter(
    (doc) => doc.document_type === "record" || (doc.document_type !== "procedure" && doc.document_type !== "reference"),
  );
  const caseTitleById = new Map(cases.map((item) => [item.case_id, item.title]));
  const providerDefaults = Object.fromEntries(providers.map((item) => [item.key, item.default_model]));

  function onExtractionProviderChange(value: string) {
    setExtractionProvider(value);
    setExtractionModel(providerDefaults[value] ?? "");
  }

  function onComplianceProviderChange(value: string) {
    setProvider(value);
    setModel(providerDefaults[value] ?? "");
  }

  function renderActivePanel() {
    switch (activeView) {
      case "upload":
        return (
          <UploadPanel
            uploadFile={uploadFile}
            providers={providers}
            uploadType={uploadType}
            uploadLanguage={uploadLanguage}
            extractOnUpload={extractOnUpload}
            extractionProvider={extractionProvider}
            extractionModel={extractionModel}
            groupId={groupId}
            busy={busy}
            onSubmit={onUploadSubmit}
            onFileChange={setUploadFile}
            onTypeChange={setUploadType}
            onLanguageChange={setUploadLanguage}
            onExtractOnUploadChange={setExtractOnUpload}
            onExtractionProviderChange={onExtractionProviderChange}
            onExtractionModelChange={setExtractionModel}
            onGroupIdChange={setGroupId}
          />
        );
      case "documents":
        return (
          <DocumentsPanel
            documents={documents}
            extractionInfoByDocument={extractionInfoByDocument}
            onRefresh={() => void refreshAll()}
            onParseDocument={(storedFilename) => void onParseDocument(storedFilename)}
            onViewOriginal={onViewOriginal}
            onCheckRequirements={(storedFilename) => void onCheckRequirements(storedFilename)}
            onDeleteDocument={(storedFilename) => void onDeleteDocument(storedFilename)}
          />
        );
      case "create-case":
        return (
          <CreateCasePanel
            title={title}
            notes={notes}
            procedureDocs={procedureDocs}
            recordDocs={recordDocs}
            referenceDocs={referenceDocs}
            extractionInfoByDocument={extractionInfoByDocument}
            selectedProcedures={selectedProcedures}
            selectedRecords={selectedRecords}
            selectedReferences={selectedReferences}
            busy={busy}
            onSubmit={onCreateCase}
            onTitleChange={setTitle}
            onNotesChange={setNotes}
            onSelectedProceduresChange={setSelectedProcedures}
            onSelectedRecordsChange={setSelectedRecords}
            onSelectedReferencesChange={setSelectedReferences}
          />
        );
      case "cases":
        return (
          <CasesPanel
            cases={cases}
            selectedCaseId={selectedCaseId}
            caseDocuments={caseDocuments}
            extractionInfoByDocument={extractionInfoByDocument}
            busy={busy}
            onShowDocuments={(caseId) => void onShowCaseDocuments(caseId)}
            onDeleteCase={(caseId) => void onDeleteCase(caseId)}
          />
        );
      case "run-compliance":
        return (
          <CompliancePanel
            cases={cases}
            providers={providers}
            selectedCaseId={selectedCaseId}
            provider={provider}
            model={model}
            method={complianceMethod}
            instructions={instructions}
            busy={busy}
            onSubmit={onRunCompliance}
            onSelectCase={setSelectedCaseId}
            onProviderChange={onComplianceProviderChange}
            onModelChange={setModel}
            onMethodChange={setComplianceMethod}
            onInstructionsChange={setInstructions}
          />
        );
      case "compliances":
        return (
          <ComplianceHistoryPanel
            cases={cases}
            complianceHistory={complianceHistory}
            selectedComplianceFile={selectedComplianceFile}
            busy={busy}
            onSelectCompliance={(caseId, fileName) => void onSelectCompliance(caseId, fileName)}
            onDeleteCompliance={(caseId, fileName) => void onDeleteCompliance(caseId, fileName)}
          />
        );
      default:
        return null;
    }
  }

  return (
    <div className="app-shell">
      <AppHeader
        documentCount={documents.length}
        caseCount={cases.length}
        hasLiveCompliance={latestCompliance !== null}
      />

      {error ? <div className="banner error">{error}</div> : null}

      <nav className="view-menu" aria-label="Windows">
        {views.map((view) => (
          <button
            key={view.id}
            className={`view-tab ${activeView === view.id ? "active" : ""}`}
            onClick={() => setActiveView(view.id)}
            type="button"
          >
            {view.label}
          </button>
        ))}
      </nav>

      <main className="workspace">
        {renderActivePanel()}
      </main>

      {deliverablePicker ? (
        <div className="overlay" role="dialog" aria-modal="true">
          <div className="modal-card">
            <div className="panel-head">
              <div>
                <h2>Requirements</h2>
                <p className="empty-state">{deliverablePicker.source_filename}</p>
              </div>
              <div className="actions">
                <button className="button button-ghost button-tiny" onClick={onAddDeliverable} type="button">
                  Add requirement
                </button>
                <button
                  className="button button-tiny"
                  onClick={() => void onSaveDeliverables()}
                  type="button"
                  disabled={busy === "save-deliverables"}
                >
                  {busy === "save-deliverables" ? "Saving..." : "Save"}
                </button>
                <button className="button button-ghost button-tiny" onClick={() => setDeliverablePicker(null)} type="button">
                  Close
                </button>
              </div>
            </div>
            <div className="stack">
              {editableDeliverables.map((item, index) => {
                return (
                  <article className="panel" key={`${index}:${item.requirement_text || "new"}`}>
                    <div className="requirement-row">
                      <strong>{index + 1}</strong>
                      <input
                        value={item.requirement_text}
                        onChange={(e) => onDeliverableFieldChange(index, "requirement_text", e.target.value)}
                      />
                      <span className="requirement-confidence">{(item.confidence * 100).toFixed(1)}%</span>
                      <button className="button button-ghost button-tiny" onClick={() => onDeleteDeliverable(index)} type="button">
                        Delete
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
