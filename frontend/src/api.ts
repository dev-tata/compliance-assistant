import type {
  CaseDocuments,
  CaseRecord,
  ComplianceMethod,
  ComplianceResponse,
  ComplianceSummary,
  DeliverableExtractionMethod,
  DeliverableExtractionResponse,
  DocumentLanguage,
  DocumentRecord,
  DocumentType,
  LLMProviderDescriptor,
  SelectedDeliverablesByDocument,
} from "./types";

const API_BASE = "http://127.0.0.1:8000";

async function handle<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

export async function listDocuments(): Promise<DocumentRecord[]> {
  return handle<DocumentRecord[]>(await fetch(`${API_BASE}/documents`));
}

export async function listLLMProviders(): Promise<LLMProviderDescriptor[]> {
  return handle<LLMProviderDescriptor[]>(await fetch(`${API_BASE}/llm/providers`));
}

export async function uploadDocument(input: {
  file: File;
  documentType: DocumentType;
  language: DocumentLanguage;
  groupId?: string;
}): Promise<DocumentRecord> {
  const formData = new FormData();
  formData.append("file", input.file);
  formData.append("document_type", input.documentType);
  formData.append("language", input.language);
  if (input.groupId) {
    formData.append("group_id", input.groupId);
  }

  return handle<DocumentRecord>(
    await fetch(`${API_BASE}/documents/upload`, {
      method: "POST",
      body: formData,
    }),
  );
}

export async function parseDocument(storedFilename: string): Promise<unknown> {
  return handle<unknown>(
    await fetch(`${API_BASE}/documents/parse/${encodeURIComponent(storedFilename)}`),
  );
}

export function getParsedDocumentUrl(storedFilename: string): string {
  return `${API_BASE}/documents/parse/${encodeURIComponent(storedFilename)}`;
}

export function getDocumentFileUrl(storedFilename: string): string {
  return `${API_BASE}/documents/file/${encodeURIComponent(storedFilename)}`;
}

export async function deleteDocument(storedFilename: string): Promise<DocumentRecord> {
  return handle<DocumentRecord>(
    await fetch(`${API_BASE}/documents/${encodeURIComponent(storedFilename)}`, {
      method: "DELETE",
    }),
  );
}

export async function listCases(): Promise<CaseRecord[]> {
  return handle<CaseRecord[]>(await fetch(`${API_BASE}/cases`));
}

export async function getCaseDocuments(caseId: string): Promise<CaseDocuments> {
  return handle<CaseDocuments>(
    await fetch(`${API_BASE}/cases/${encodeURIComponent(caseId)}/documents`),
  );
}

export async function listAllCompliances(): Promise<ComplianceSummary[]> {
  return handle<ComplianceSummary[]>(await fetch(`${API_BASE}/cases/compliances`));
}

export async function createCase(input: {
  title: string;
  procedureStoredFilenames: string[];
  recordStoredFilenames: string[];
  referenceStoredFilenames: string[];
  notes?: string;
}): Promise<CaseRecord> {
  return handle<CaseRecord>(
    await fetch(`${API_BASE}/cases`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: input.title,
        procedure_stored_filenames: input.procedureStoredFilenames,
        record_stored_filenames: input.recordStoredFilenames,
        reference_stored_filenames: input.referenceStoredFilenames,
        notes: input.notes || null,
      }),
    }),
  );
}

export async function deleteCase(caseId: string): Promise<CaseRecord> {
  return handle<CaseRecord>(
    await fetch(`${API_BASE}/cases/${encodeURIComponent(caseId)}`, {
      method: "DELETE",
    }),
  );
}

export async function listCaseCompliances(caseId: string): Promise<ComplianceSummary[]> {
  return handle<ComplianceSummary[]>(
    await fetch(`${API_BASE}/cases/${encodeURIComponent(caseId)}/compliances`),
  );
}

export async function getCaseComplianceResult(
  caseId: string,
  fileName: string,
): Promise<ComplianceResponse> {
  return handle<ComplianceResponse>(
    await fetch(
      `${API_BASE}/cases/${encodeURIComponent(caseId)}/compliances/${encodeURIComponent(fileName)}`,
    ),
  );
}

export function getCaseComplianceResultUrl(caseId: string, fileName: string): string {
  return `${API_BASE}/cases/${encodeURIComponent(caseId)}/compliances/${encodeURIComponent(fileName)}`;
}

export async function deleteCaseComplianceResult(
  caseId: string,
  fileName: string,
): Promise<ComplianceSummary> {
  return handle<ComplianceSummary>(
    await fetch(
      `${API_BASE}/cases/${encodeURIComponent(caseId)}/compliances/${encodeURIComponent(fileName)}`,
      { method: "DELETE" },
    ),
  );
}

export async function runCompliance(input: {
  caseId: string;
  provider: string;
  model: string;
  method: ComplianceMethod;
  instructions?: string;
  selectedDeliverablesByDocument?: SelectedDeliverablesByDocument;
}): Promise<ComplianceResponse> {
  return handle<ComplianceResponse>(
    await fetch(`${API_BASE}/cases/${encodeURIComponent(input.caseId)}/compliance`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: input.provider,
        model: input.model,
        method: input.method,
        instructions: input.instructions || null,
        selected_deliverables_by_document: input.selectedDeliverablesByDocument ?? {},
      }),
    }),
  );
}

export async function extractDeliverables(input: {
  caseId: string;
  provider: string;
  model: string;
  method: DeliverableExtractionMethod;
  instructions?: string;
}): Promise<DeliverableExtractionResponse> {
  return handle<DeliverableExtractionResponse>(
    await fetch(`${API_BASE}/cases/${encodeURIComponent(input.caseId)}/deliverables/extract`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: input.provider,
        model: input.model,
        method: input.method,
        instructions: input.instructions || null,
      }),
    }),
  );
}

export async function extractDocumentDeliverables(input: {
  storedFilename: string;
  provider: string;
  model: string;
  method: DeliverableExtractionMethod;
  instructions?: string;
}): Promise<DeliverableExtractionResponse> {
  return handle<DeliverableExtractionResponse>(
    await fetch(
      `${API_BASE}/documents/${encodeURIComponent(input.storedFilename)}/deliverables/extract`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: input.provider,
          model: input.model,
          method: input.method,
          instructions: input.instructions || null,
        }),
      },
    ),
  );
}

export async function getLatestDocumentDeliverables(
  storedFilename: string,
): Promise<DeliverableExtractionResponse> {
  return handle<DeliverableExtractionResponse>(
    await fetch(
      `${API_BASE}/documents/${encodeURIComponent(storedFilename)}/deliverables/latest`,
    ),
  );
}

export async function updateLatestDocumentDeliverables(input: {
  storedFilename: string;
  deliverables: DeliverableExtractionResponse["deliverables"];
}): Promise<DeliverableExtractionResponse> {
  return handle<DeliverableExtractionResponse>(
    await fetch(
      `${API_BASE}/documents/${encodeURIComponent(input.storedFilename)}/deliverables/latest`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          deliverables: input.deliverables,
        }),
      },
    ),
  );
}
