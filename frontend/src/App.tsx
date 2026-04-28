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
  setDocumentFrozen,
  updateCaseRecords,
  updateLatestDocumentDeliverables,
  uploadDocument,
} from "./api";
import { AppHeader } from "./components/AppHeader";
import { CasesPanel } from "./components/CasesPanel";
import { ComplianceHistoryPanel } from "./components/ComplianceHistoryPanel";
import { CompliancePanel } from "./components/CompliancePanel";
import { CreateCasePanel } from "./components/CreateCasePanel";
import { DocumentsPanel } from "./components/DocumentsPanel";
import { FrozenBadge } from "./components/FrozenBadge";
import { UploadPanel } from "./components/UploadPanel";
import { formatMethodLabel } from "./utils/formatMethodLabel";
import type {
  CaseRecord,
  CaseDocuments,
  ComplianceRunMethod,
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
      </tr>
    `;
  }

  return rows.map((row, index) => {
    const evidenceStrength = procedureToRecord?.[index]?.evidence_strength;
    const evidenceStrengthLabel = typeof evidenceStrength === "number"
      ? `${(evidenceStrength * 100).toFixed(1)}%`
      : "&mdash;";
    return `
    <tr>
      <td>${index + 1}</td>
      <td>
        <div>${escapeHtml(row.requirement) || "&mdash;"}</div>
        ${row.rationale ? `<div class="row-rationale">${escapeHtml(row.rationale)}</div>` : ""}
      </td>
      <td>${evidenceStrengthLabel}</td>
      <td>${escapeHtml(row.status) || "&mdash;"}</td>
    </tr>
  `;
  }).join("");
}

function getAnalysisFindings(
  analysis: ComplianceResponse["analysis"] | NonNullable<ComplianceResponse["stages"]>[number]["analysis"] | null | undefined,
) {
  if (!analysis) {
    return [];
  }
  return analysis.procedure_to_record?.length
    ? analysis.procedure_to_record
    : analysis.findings;
}

function formatRecallMetric(value: number): string {
  return `${value.toFixed(4)} / 1.0`;
}

function formatStrengthPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function summarizeStatuses(
  procedureToRecord: ComplianceResponse["analysis"]["procedure_to_record"] | ComplianceResponse["analysis"]["findings"],
): Record<"satisfied" | "partial" | "not_satisfied", number> {
  if (!procedureToRecord?.length) {
    return {
      satisfied: 0,
      partial: 0,
      not_satisfied: 0,
    };
  }
  return procedureToRecord.reduce(
    (counts, finding) => {
      counts[finding.status] += 1;
      return counts;
    },
    {
      satisfied: 0,
      partial: 0,
      not_satisfied: 0,
    } as Record<"satisfied" | "partial" | "not_satisfied", number>,
  );
}

function getComplianceStages(compliance: ComplianceResponse) {
  if (compliance.stages?.length) {
    return compliance.stages;
  }
  const fallbackStages = [];
  if (compliance.baseline_analysis) {
    fallbackStages.push({
      stage_key: "legacy_baseline",
      stage_label: formatMethodLabel(compliance.baseline_method ?? "non_rag"),
      method: compliance.baseline_method ?? "non_rag",
      analysis: compliance.baseline_analysis,
      retrieval_metrics: compliance.baseline_retrieval_metrics ?? null,
    });
  }
  fallbackStages.push({
    stage_key: "legacy_final",
    stage_label: formatMethodLabel(compliance.method),
    method: compliance.method,
    analysis: compliance.analysis,
    retrieval_metrics: compliance.retrieval_metrics ?? null,
  });
  return fallbackStages;
}

function applyFrozenDeliverableConfidence(
  items: DeliverableItem[],
  frozen: boolean,
): DeliverableItem[] {
  if (!frozen) {
    return items;
  }
  return items.map((item) => ({
    ...item,
    validated_confidence: item.validated_confidence,
  }));
}

function renderAssessmentBar({
  analysis,
  retrievalMetrics,
  statusSummary,
}: {
  analysis: ComplianceResponse["analysis"];
  retrievalMetrics?: ComplianceResponse["retrieval_metrics"] | ComplianceResponse["baseline_retrieval_metrics"] | null;
  statusSummary: Record<"satisfied" | "partial" | "not_satisfied", number>;
}): string {
  const completedPercent = analysis.completion_percent;
  return `
    <div class="assessment">
      <span class="assessment-item assessment-item-edge-left">Completed <strong>${completedPercent}%</strong></span>
      <span class="assessment-item">Weighted Completed <strong>${analysis.weighted_completion_percent}%</strong></span>
      <span class="assessment-item assessment-status-item">
        <span class="status status-satisfied">Satisfied</span>
        <span class="assessment-status-count">${statusSummary.satisfied}</span>
      </span>
      <span class="assessment-item assessment-status-item">
        <span class="status status-partial">Partial</span>
        <span class="assessment-status-count">${statusSummary.partial}</span>
      </span>
      <span class="assessment-item assessment-status-item">
        <span class="status status-not_satisfied">Not_Satisfied</span>
        <span class="assessment-status-count">${statusSummary.not_satisfied}</span>
      </span>
    </div>
    <div class="assessment">
      <span class="assessment-item">Coverage <strong>${analysis.overall_coverage_percent}%</strong></span>
      <span class="assessment-item">Weighted Coverage <strong>${analysis.weighted_coverage_percent}%</strong></span>
      <span class="assessment-item">Avg Evidence Strength <strong>${formatStrengthPercent(analysis.average_evidence_strength)}</strong></span>
      <span class="assessment-item">Weighted Avg Strength <strong>${formatStrengthPercent(analysis.weighted_average_evidence_strength)}</strong></span>
      ${(typeof retrievalMetrics?.record_k === "number"
        && typeof (retrievalMetrics?.average_record_recall_at_k ?? retrievalMetrics?.record_recall_at_k) === "number")
        ? `<span class="assessment-item">Avg Recall@${retrievalMetrics.record_k} <strong>${formatRecallMetric(retrievalMetrics.average_record_recall_at_k ?? retrievalMetrics.record_recall_at_k ?? 0)}</strong></span>`
        : ""}
    </div>
  `;
}

function renderEvidenceList(
  finding: ComplianceResponse["analysis"]["findings"][number],
): string {
  const evidenceItems = finding.evidence_items?.length
    ? finding.evidence_items
    : finding.evidence.map((text) => ({
        text,
        source_document: finding.source_document,
        stage_label: null,
      }));

  if (!evidenceItems.length) {
    return "<li>None</li>";
  }

  return evidenceItems.map((item) => `
    <li>
      ${escapeHtml(item.text)}
      ${item.stage_label ? ` <span class="evidence-stage">${escapeHtml(item.stage_label)}</span>` : ""}
    </li>
  `).join("");
}

function renderRequirementEvaluations(
  analysis: ComplianceResponse["analysis"] | NonNullable<ComplianceResponse["stages"]>[number]["analysis"],
  retrievalMetrics?: ComplianceResponse["retrieval_metrics"] | ComplianceResponse["baseline_retrieval_metrics"] | null,
): string {
  const findings = getAnalysisFindings(analysis);
  return findings.map((finding, index) => {
    const rowRecall = analysis.linked_rows?.[index]?.record_recall_at_k;
    const recallLabel = typeof rowRecall === "number" && retrievalMetrics?.record_k
      ? `Recall@${retrievalMetrics.record_k}: ${rowRecall.toFixed(1)} / 1.0`
      : "";
    return `
      <article class="finding requirement-evaluation">
        <header>
          <div class="finding-title">
            <div class="requirement-label">Requirement ${index + 1}</div>
            <strong>${escapeHtml(finding.requirement)}</strong>
          </div>
          <div class="finding-header-meta">
            <span class="status status-${escapeHtml(finding.status)}">${escapeHtml(finding.status)}</span>
            ${recallLabel ? `<span class="finding-recall">${escapeHtml(recallLabel)}</span>` : ""}
          </div>
        </header>
        <div class="finding-metrics">
          <span><strong>Source:</strong> ${escapeHtml(finding.source_document || "none")}</span>
          <span><strong>Evidence strength:</strong> ${(finding.evidence_strength * 100).toFixed(1)}% · <strong>Weight:</strong> ${finding.weight.toFixed(1)} · <strong>Coverage:</strong> ${finding.requirement_coverage_percent}%</span>
        </div>
        <ul>${renderEvidenceList(finding)}</ul>
      </article>
    `;
  }).join("");
}

function renderStagePanel(
  stage: NonNullable<ComplianceResponse["stages"]>[number] | {
    stage_key: string;
    stage_label: string;
    method: string;
    analysis: ComplianceResponse["analysis"];
    retrieval_metrics?: ComplianceResponse["retrieval_metrics"] | ComplianceResponse["baseline_retrieval_metrics"] | null;
  },
): string {
  const stageFindings = getAnalysisFindings(stage.analysis);
  const stageStatusSummary = summarizeStatuses(stageFindings);
  return `
    <section class="stage-panel">
      <h2 class="stage-panel-title">${escapeHtml(stage.stage_label)}</h2>
      ${renderAssessmentBar({
        analysis: stage.analysis,
        retrievalMetrics: stage.retrieval_metrics,
        statusSummary: stageStatusSummary,
      })}
      <details class="stage-panel-details">
        <summary>Requirement details</summary>
        <div class="stage-panel-body">
          <div class="gap-table-wrap">
            <table class="gap-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Requirement</th>
                  <th>Evidence Strength</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>${renderLinkedRows(
                stage.analysis.linked_rows ?? [],
                stageFindings,
              )}</tbody>
            </table>
          </div>
          <section class="section-block">
            <h3>Grounded Evidence (Procedure → Record)</h3>
            <div class="findings">${renderRequirementEvaluations(stage.analysis, stage.retrieval_metrics)}</div>
          </section>
        </div>
      </details>
    </section>
  `;
}

function openComplianceWindow(compliance: ComplianceResponse, caseTitle?: string): void {
  const popup = window.open("", "_blank", "width=1080,height=900");
  if (!popup) {
    throw new Error("Popup blocked");
  }

  const stages = getComplianceStages(compliance);
  const stagePanels = stages.map((stage) => renderStagePanel(stage)).join("");
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
            padding: 16px;
            font-family: Georgia, "Times New Roman", serif;
            font-size: 13px;
            line-height: 1.35;
            color: #231b12;
            background: #f3efe4;
          }
          .page {
            width: min(1080px, 100%);
            margin: 0 auto;
          }
          h1, h2, h3, h4, p { margin-top: 0; }
          h1 { white-space: nowrap; }
          .head, .findings { display: grid; gap: 8px; }
          .head { grid-template-columns: 1fr; align-items: start; }
          .head-main p { margin-bottom: 12px; font-size: 0.8rem; }
          .case-line { white-space: nowrap; }
          .path-line { margin-top: -6px; }
          .code { color: #8a3b12; font-family: ui-monospace, Consolas, monospace; word-break: break-all; font-size: 0.74rem; }
          .assessment, .score-card, .result-list, .finding {
            padding: 8px 10px;
            border: 1px solid #d8c8a8;
            border-radius: 7px;
            background: rgba(255,255,255,0.72);
          }
          .assessment {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            flex-wrap: wrap;
            row-gap: 8px;
            column-gap: 12px;
          }
          .assessment + .assessment {
            margin-top: 12px;
          }
          .assessment-item {
            color: #6f6252;
            text-align: center;
            white-space: nowrap;
            flex: 0 0 auto;
            font-size: 0.76rem;
          }
          .assessment-item-edge-left {
            text-align: left;
          }
          .assessment-item strong {
            color: #231b12;
          }
          .assessment-status-item {
            display: inline-flex;
            align-items: center;
            gap: 0;
          }
          .assessment-status-count {
            color: #231b12;
            font-weight: 700;
          }
          .score-card span, .result-list li, .finding p, .finding li { color: #6f6252; }
          .section-block { margin-top: 12px; }
          .gap-table-wrap {
            margin-top: 12px;
            overflow-x: auto;
          }
          .gap-table {
            width: 100%;
            border-collapse: collapse;
            border: 1px solid #d8c8a8;
            background: rgba(255,255,255,0.72);
            border-radius: 7px;
            overflow: hidden;
          }
          .gap-table th,
          .gap-table td {
            padding: 8px 10px;
            border-bottom: 1px solid #e6d8bd;
            text-align: left;
            vertical-align: top;
            font-size: 0.76rem;
          }
          .gap-table th {
            background: #efe3c9;
            color: #5a3a16;
          }
          .gap-table td {
            color: #6f6252;
          }
          .row-rationale {
            margin-top: 4px;
            color: #6f6252;
            font-size: 0.72rem;
          }
          .gap-table tr:last-child td {
            border-bottom: none;
          }
          .finding header {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 8px 14px;
            align-items: start;
          }
          .finding-title {
            min-width: 0;
          }
          .finding-title strong {
            display: block;
          }
          .finding-header-meta {
            display: inline-flex;
            align-items: center;
            justify-content: flex-end;
            gap: 8px;
            white-space: nowrap;
          }
          .finding-recall {
            white-space: nowrap;
          }
          .requirement-label {
            margin-bottom: 6px;
            color: #8a3b12;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
          }
          .finding-metrics {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 6px 12px;
            margin-bottom: 8px;
            color: #6f6252;
            font-size: 0.74rem;
          }
          .evidence-stage {
            display: inline-block;
            margin-left: 6px;
            padding: 1px 6px;
            border-radius: 999px;
            background: #efe3c9;
            color: #8a3b12;
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
          }
          .stage-panel {
            border: 1px solid #d8c8a8;
            border-radius: 7px;
            background: rgba(255,255,255,0.72);
            padding: 10px 12px 12px;
          }
          .stage-panel + .stage-panel {
            margin-top: 10px;
          }
          .stage-panel-title {
            margin-bottom: 10px;
            color: #5a3a16;
          }
          .stage-panel-details {
            margin-top: 10px;
          }
          .stage-panel-details summary {
            cursor: pointer;
            color: #5a3a16;
            font-weight: 700;
            list-style: none;
          }
          .stage-panel-details summary::-webkit-details-marker {
            display: none;
          }
          .stage-panel-body {
            padding-top: 10px;
          }
          .status { border-radius: 4px; padding: 2px 7px; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; }
          .status-satisfied { background: #d7f1df; color: #2d6a4f; }
          .status-partial { background: #fff1cf; color: #b7791f; }
          .status-not_satisfied { background: #ffdedd; color: #9b2226; }
          @media (max-width: 900px) {
            .head { grid-template-columns: 1fr; }
            .assessment {
              flex-direction: column;
              align-items: flex-start;
            }
            .finding header {
              grid-template-columns: 1fr;
            }
            .finding-header-meta {
              justify-content: flex-start;
            }
            .finding-metrics { grid-template-columns: 1fr; }
          }
        </style>
      </head>
      <body>
        <div class="page">
          <div class="head">
            <div class="head-main">
              <h1>Compliance result</h1>
              <p class="code path-line">${escapeHtml(compliance.saved_at)}</p>
              ${caseTitle ? `<p class="case-line">Case: ${escapeHtml(caseTitle)}</p>` : ""}
              <p>${escapeHtml(`${formatMethodLabel(compliance.method)} · ${compliance.compliance_provider} · ${compliance.compliance_model}`)}</p>
              <p>Created: ${escapeHtml(formatDateTime(compliance.created_at))}</p>
            </div>
          </div>
          <section>${stagePanels}</section>
        </div>
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
  const [title, setTitle] = useState("");
  const [notes, setNotes] = useState("");
  const [providers, setProviders] = useState<LLMProviderDescriptor[]>(FALLBACK_LLM_PROVIDERS);
  const [provider, setProvider] = useState(FALLBACK_LLM_PROVIDERS[0]?.key ?? "openai");
  const [model, setModel] = useState(FALLBACK_LLM_PROVIDERS[0]?.default_model ?? "gpt-5.4-nano");
  const complianceMethod: ComplianceRunMethod = "two_stage_rag";
  const [instructions, setInstructions] = useState("");
  const [selectedCaseId, setSelectedCaseId] = useState("");
  const [selectedAdditionalComplianceDocuments, setSelectedAdditionalComplianceDocuments] = useState<string[]>([]);
  const [caseDocuments, setCaseDocuments] = useState<CaseDocuments | null>(null);
  const [latestCompliance, setLatestCompliance] = useState<ComplianceResponse | null>(null);
  const [complianceHistory, setComplianceHistory] = useState<ComplianceSummary[]>([]);
  const [openComplianceDocumentsFile, setOpenComplianceDocumentsFile] = useState("");
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

  useEffect(() => {
    setSelectedAdditionalComplianceDocuments([]);
  }, [selectedCaseId]);

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
    const eligibleDocuments = loadedDocuments.filter((doc) => doc.document_type === "procedure");

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

  async function onShowCaseDocuments(caseId: string, fileName?: string) {
    if (fileName && openComplianceDocumentsFile === fileName) {
      setOpenComplianceDocumentsFile("");
      setCaseDocuments(null);
      return;
    }

    if (fileName && caseDocuments?.case_id === caseId) {
      setOpenComplianceDocumentsFile(fileName);
      return;
    }

    if (!fileName && caseDocuments?.case_id === caseId) {
      if (openComplianceDocumentsFile) {
        setOpenComplianceDocumentsFile("");
      } else {
        setCaseDocuments(null);
      }
      return;
    }

    setBusy(`case-docs:${caseId}`);
    setError("");
    try {
      const payload = await getCaseDocuments(caseId);
      setCaseDocuments(payload);
      setSelectedCaseId(caseId);
      setOpenComplianceDocumentsFile(fileName ?? "");
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
      const fallbackTitle = selectedProcedures
        .map((storedFilename) => documents.find((doc) => doc.stored_filename === storedFilename)?.source_filename)
        .find((value): value is string => Boolean(value))
        ?? "Case";
      const created = await createCase({
        title: title.trim() || fallbackTitle,
        procedureStoredFilenames: selectedProcedures,
        recordStoredFilenames: selectedRecords,
        referenceStoredFilenames: [],
        notes,
      });
      setTitle("");
      setNotes("");
      setSelectedProcedures([]);
      setSelectedRecords([]);
      setSelectedCaseId(created.case_id);
      setCaseDocuments(null);
      setOpenComplianceDocumentsFile("");
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
        additionalDocumentFilenames: selectedAdditionalComplianceDocuments,
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
                font-size: 12px;
                background: #1c1b1a;
                color: #f5f0e8;
              }
              pre {
                margin: 0;
                padding: 12px;
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
    const document = documents.find((item) => item.stored_filename === storedFilename);
    if ((document?.document_type === "procedure" || document?.document_type === "reference") && document.frozen) {
      const documentLabel = document.document_type === "procedure" ? "Procedure" : "Reference";
      setError(`${documentLabel} "${document.source_filename}" is frozen. Unfreeze it before deleting.`);
      return;
    }
    if (!window.confirm(`Delete document "${document?.source_filename ?? storedFilename}"?`)) {
      return;
    }

    setBusy(`delete-doc:${storedFilename}`);
    setError("");
    try {
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
      const document = documents.find((item) => item.stored_filename === storedFilename);
      const frozen = document?.document_type === "procedure" && document.frozen;
      const normalizedDeliverables = applyFrozenDeliverableConfidence(result.deliverables, frozen);
      setDeliverablePicker({
        ...result,
        deliverables: normalizedDeliverables,
      });
      setEditableDeliverables(normalizedDeliverables);
      const documentKey = result.source_filename ?? storedFilename;
      setSelectedDeliverablesByDocument((current) => ({
        ...current,
        [documentKey]: result.deliverables.map((item) => item.requirement_text),
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Loading deliverables failed");
    } finally {
      setBusy("");
    }
  }

  async function onToggleDocumentFreeze(storedFilename: string, frozen: boolean) {
    const document = documents.find((item) => item.stored_filename === storedFilename);
    if (!document || (document.document_type !== "procedure" && document.document_type !== "reference")) {
      return;
    }

    setBusy(`freeze-doc:${storedFilename}`);
    setError("");
    try {
      const updated = await setDocumentFrozen(storedFilename, frozen);
      setDocuments((current) =>
        current.map((item) => (item.stored_filename === storedFilename ? updated : item)),
      );
      setCaseDocuments((current) =>
        current
          ? {
              ...current,
              procedure_documents: current.procedure_documents.map((item) =>
                item.stored_filename === storedFilename ? updated : item,
              ),
              reference_documents: current.reference_documents.map((item) =>
                item.stored_filename === storedFilename ? updated : item,
              ),
            }
          : null,
      );
      if (deliverablePicker?.document_stored_filename === storedFilename) {
        const normalizedDeliverables = applyFrozenDeliverableConfidence(editableDeliverables, frozen);
        setDeliverablePicker((current) => (
          current
            ? {
                ...current,
                deliverables: normalizedDeliverables,
              }
            : current
        ));
        setEditableDeliverables(normalizedDeliverables);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Freeze update failed");
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
        return updated;
      }),
    );
  }

  function onDeliverableWeightChange(index: number, value: string) {
    const parsed = Number.parseFloat(value);
    onDeliverableFieldChange(
      index,
      "weight",
      Number.isFinite(parsed) && parsed > 0 ? Number(parsed.toFixed(2)) : 1,
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
        weight: 1,
        validated_confidence: 0,
      },
    ]);
  }

  function onDeleteDeliverable(index: number) {
    if (!window.confirm(`Delete requirement ${index + 1}?`)) {
      return;
    }
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
            || deliverablePicker.document_stored_filename
            || "",
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
    const caseRecord = cases.find((item) => item.case_id === caseId);
    if (!window.confirm(`Delete case "${caseRecord?.title ?? caseId}"?`)) {
      return;
    }

    setBusy(`delete-case:${caseId}`);
    setError("");
    try {
      await deleteCase(caseId);
      if (selectedCaseId === caseId) {
        setSelectedCaseId("");
        setCaseDocuments(null);
        setOpenComplianceDocumentsFile("");
        setLatestCompliance(null);
        setComplianceHistory([]);
        setSelectedComplianceFile("");
      }
      setCaseDocuments((current) => (current?.case_id === caseId ? null : current));
      setOpenComplianceDocumentsFile((current) => (caseDocuments?.case_id === caseId ? "" : current));
      await refreshAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Case delete failed");
    } finally {
      setBusy("");
    }
  }

  async function onUpdateCaseRecords(caseId: string, recordStoredFilenames: string[]) {
    setBusy(`update-case-records:${caseId}`);
    setError("");
    try {
      const updated = await updateCaseRecords({
        caseId,
        recordStoredFilenames,
      });
      setCases((current) =>
        current.map((item) => (item.case_id === caseId ? updated : item)),
      );
      if (caseDocuments?.case_id === caseId) {
        const payload = await getCaseDocuments(caseId);
        setCaseDocuments(payload);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Case update failed");
    } finally {
      setBusy("");
    }
  }

  async function onDeleteCompliance(caseId: string, fileName: string) {
    if (!window.confirm(`Delete compliance result "${fileName}"?`)) {
      return;
    }

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
      if (openComplianceDocumentsFile === fileName) {
        setOpenComplianceDocumentsFile("");
        setCaseDocuments(null);
      }
      await refreshComplianceHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Compliance delete failed");
    } finally {
      setBusy("");
    }
  }

  async function onDeleteAllCompliances() {
    if (complianceHistory.length === 0) {
      return;
    }
    if (!window.confirm(`Delete all ${complianceHistory.length} compliance results?`)) {
      return;
    }

    setBusy("delete-all-compliances");
    setError("");
    try {
      for (const item of complianceHistory) {
        await deleteCaseComplianceResult(item.case_id, item.file_name);
      }
      setSelectedComplianceFile("");
      setLatestCompliance(null);
      setOpenComplianceDocumentsFile("");
      setCaseDocuments(null);
      await refreshComplianceHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Bulk compliance delete failed");
    } finally {
      setBusy("");
    }
  }

  const procedureDocs = documents.filter((doc) => doc.document_type === "procedure");
  const recordDocs = documents.filter(
    (doc) => doc.document_type === "record" || (doc.document_type !== "procedure" && doc.document_type !== "reference"),
  );
  const activeDeliverableDocument = deliverablePicker
    ? documents.find((item) => item.stored_filename === deliverablePicker.document_stored_filename) ?? null
    : null;
  const deliverablePickerFrozen = activeDeliverableDocument?.document_type === "procedure"
    && activeDeliverableDocument.frozen;
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
            busy={busy}
            onRefresh={() => void refreshAll()}
            onParseDocument={(storedFilename) => void onParseDocument(storedFilename)}
            onViewOriginal={onViewOriginal}
            onCheckRequirements={(storedFilename) => void onCheckRequirements(storedFilename)}
            onToggleDocumentFreeze={(storedFilename, frozen) => void onToggleDocumentFreeze(storedFilename, frozen)}
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
            extractionInfoByDocument={extractionInfoByDocument}
            selectedProcedures={selectedProcedures}
            selectedRecords={selectedRecords}
            busy={busy}
            onSubmit={onCreateCase}
            onTitleChange={setTitle}
            onNotesChange={setNotes}
            onSelectedProceduresChange={setSelectedProcedures}
            onSelectedRecordsChange={setSelectedRecords}
          />
        );
      case "cases":
        return (
          <CasesPanel
            cases={cases}
            selectedCaseId={selectedCaseId}
            caseDocuments={caseDocuments}
            openDocumentsFile={openComplianceDocumentsFile}
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
            documents={documents}
            extractionInfoByDocument={extractionInfoByDocument}
            providers={providers}
            selectedCaseId={selectedCaseId}
            provider={provider}
            model={model}
            instructions={instructions}
            selectedAdditionalDocuments={selectedAdditionalComplianceDocuments}
            busy={busy}
            onSubmit={onRunCompliance}
            onSelectCase={setSelectedCaseId}
            onProviderChange={onComplianceProviderChange}
            onModelChange={setModel}
            onInstructionsChange={setInstructions}
            onSelectedAdditionalDocumentsChange={setSelectedAdditionalComplianceDocuments}
          />
        );
      case "compliances":
        return (
          <ComplianceHistoryPanel
            cases={cases}
            caseDocuments={caseDocuments}
            complianceHistory={complianceHistory}
            documents={documents}
            openDocumentsFile={openComplianceDocumentsFile}
            selectedComplianceFile={selectedComplianceFile}
            busy={busy}
            extractionInfoByDocument={extractionInfoByDocument}
            onSelectCompliance={(caseId, fileName) => void onSelectCompliance(caseId, fileName)}
            onShowDocuments={(caseId, fileName) => void onShowCaseDocuments(caseId, fileName)}
            onDeleteCompliance={(caseId, fileName) => void onDeleteCompliance(caseId, fileName)}
            onDeleteAllCompliances={() => void onDeleteAllCompliances()}
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
            <div className="panel-head modal-panel-head">
              <div>
                <h2>Requirements</h2>
                <p className="empty-state modal-document-status">
                  {deliverablePicker.source_filename}
                  {deliverablePickerFrozen && activeDeliverableDocument ? " " : null}
                  {deliverablePickerFrozen && activeDeliverableDocument ? (
                    <FrozenBadge document={activeDeliverableDocument} />
                  ) : null}
                </p>
              </div>
              <div className="actions">
                {!deliverablePickerFrozen ? (
                  <button className="button button-ghost button-tiny" onClick={onAddDeliverable} type="button">
                    Add requirement
                  </button>
                ) : null}
                {!deliverablePickerFrozen ? (
                  <button
                    className="button button-tiny"
                    onClick={() => void onSaveDeliverables()}
                    type="button"
                    disabled={busy === "save-deliverables"}
                  >
                    {busy === "save-deliverables" ? "Saving..." : "Save"}
                  </button>
                ) : null}
                <button className="button button-ghost button-tiny" onClick={() => setDeliverablePicker(null)} type="button">
                  Close
                </button>
              </div>
            </div>
            <div className="stack">
              {editableDeliverables.map((item, index) => {
                return (
                  <article className="panel" key={index}>
                    <div className={`requirement-row ${deliverablePickerFrozen ? "requirement-row-frozen" : "requirement-row-editable"}`}>
                      <strong>{index + 1}</strong>
                      <input
                        value={item.requirement_text}
                        onChange={(e) => onDeliverableFieldChange(index, "requirement_text", e.target.value)}
                        disabled={deliverablePickerFrozen}
                      />
                      {deliverablePickerFrozen ? (
                        <span className="requirement-weight">W {item.weight.toFixed(1)}</span>
                      ) : (
                        <input
                          className="requirement-weight-input"
                          type="number"
                          min="0.1"
                          step="0.1"
                          value={item.weight}
                          onChange={(e) => onDeliverableWeightChange(index, e.target.value)}
                        />
                      )}
                      {deliverablePickerFrozen ? (
                        <span className="requirement-confidence">
                          {(item.validated_confidence * 100).toFixed(1)}%
                        </span>
                      ) : null}
                      {!deliverablePickerFrozen ? (
                        <button
                          className="button button-ghost button-tiny"
                          onClick={() => onDeleteDeliverable(index)}
                          type="button"
                        >
                          Delete
                        </button>
                      ) : null}
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
