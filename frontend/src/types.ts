export type DocumentType =
  | "procedure"
  | "record"
  | "template"
  | "registry"
  | "risk_assessment"
  | "requirement_specification"
  | "validation_plan"
  | "validation_report"
  | "test_plan"
  | "test_execution"
  | "change_request"
  | "reference";

export type DocumentLanguage = "en" | "sv" | "mixed";

export type DocumentRecord = {
  source_filename: string;
  created_at: string | null;
  stored_filename: string;
  stored_at: string;
  document_type: DocumentType | null;
  language: DocumentLanguage | null;
  group_id: string | null;
  parsed_json_at: string | null;
  content_hash: string | null;
};

export type CaseRecord = {
  case_id: string;
  created_at: string | null;
  title: string;
  procedure_stored_filenames: string[];
  record_stored_filenames: string[];
  reference_stored_filenames: string[];
  notes: string | null;
};

export type CaseDocuments = {
  case_id: string;
  title: string;
  procedure_documents: DocumentRecord[];
  record_documents: DocumentRecord[];
  reference_documents: DocumentRecord[];
};

export type LLMProviderDescriptor = {
  key: string;
  label: string;
  default_model: string;
  description: string;
  endpoint_mode: string;
};

export type ComplianceSummary = {
  case_id: string;
  file_name: string;
  created_at: string;
  saved_at: string;
  provider: string;
  model: string;
  method: ComplianceMethod;
  overall_assessment: string;
};

export type ComplianceMethod = "non_rag" | "simple_rag" | "nested_rag";
export type DeliverableExtractionMethod = "non_rag" | "simple_rag" | "nested_rag";

export type ComplianceStatus = "satisfied" | "partial" | "not_satisfied";

export type ComplianceFinding = {
  requirement: string;
  status: ComplianceStatus;
  evidence: string[];
  source_documents: string[];
  evidence_strength: number;
  confidence: number;
  weight: number;
};

export type ComplianceLinkedRow = {
  requirement: string;
  status: ComplianceStatus;
  gap: string;
  recommendation: string;
};

export type ComplianceAnalysis = {
  overall_assessment: string;
  gaps: string[];
  linked_rows: ComplianceLinkedRow[];
  findings: ComplianceFinding[];
  procedure_to_record?: ComplianceFinding[];
  recommended_actions: string[];
};

export type ComplianceScores = {
  m1_binary_rule_score: number;
  m2_ordinal_score: number;
  m3_evidence_weighted_score: number;
  m4_confidence_aware_score: number;
  m5_grounding_score: number;
};

export type SectionMatch = {
  procedure_document: string;
  procedure_section_label: string | null;
  procedure_heading_title: string | null;
  record_document: string | null;
  record_section_label: string | null;
  record_heading_title: string | null;
  match_percent: number;
  match_basis: string;
};

export type ComplianceResponse = {
  case_id: string;
  compliance_provider: string;
  compliance_model: string;
  extraction_provider?: string | null;
  extraction_model?: string | null;
  method: ComplianceMethod;
  created_at: string;
  saved_at: string;
  analysis: ComplianceAnalysis;
  scores: ComplianceScores;
  section_matches: SectionMatch[];
};

export type DeliverableItem = {
  section_label: string;
  heading_title: string;
  requirement_text: string;
  requirement_type:
    | "document_output"
    | "recorded_information"
    | "approval_or_signoff"
    | "update_or_notification"
    | "archival_or_storage"
    | "change_control"
    | "validation_activity";
  mandatory: boolean;
  source_quote: string;
  source_document: string;
  required_by_procedure: boolean;
  confidence: number;
};

export type DeliverableExtractionResponse = {
  case_id: string | null;
  document_stored_filename: string | null;
  source_filename: string | null;
  extraction_provider: string;
  extraction_model: string;
  prompt_version?: string;
  parser_version?: string;
  method: DeliverableExtractionMethod;
  created_at: string;
  saved_at: string;
  deliverables: DeliverableItem[];
};

export type SelectedDeliverablesByDocument = Record<string, string[]>;
