import type { DocumentLanguage, DocumentType } from "./types";

export const documentTypes: DocumentType[] = [
  "procedure",
  "template",
  "registry",
  "risk_assessment",
  "requirement_specification",
  "validation_plan",
  "validation_report",
  "test_plan",
  "test_execution",
  "change_request",
  "reference",
];

export const languages: DocumentLanguage[] = ["en", "sv"];

export const documentTypeLabels: Record<DocumentType, string> = {
  procedure: "Procedure",
  record: "Record",
  template: "Template",
  registry: "Registry",
  risk_assessment: "Risk assessment",
  requirement_specification: "Requirement specification",
  validation_plan: "Validation plan",
  validation_report: "Validation report",
  test_plan: "Test plan",
  test_execution: "Test execution",
  change_request: "Change request",
  reference: "Reference",
};

export const languageLabels: Record<DocumentLanguage, string> = {
  en: "English",
  sv: "Swedish",
  mixed: "Mixed",
};
