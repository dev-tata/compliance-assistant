import type { ComplianceMethod } from "../types";

type MethodValue = ComplianceMethod | string | null | undefined;

export function formatMethodLabel(method: MethodValue): string {
  switch (method) {
    case "non_rag":
      return "Non-RAG";
    case "single_call_two_stage_rag":
    case "two_stage_rag":
      return "Two-Stage RAG";
    case "record_retrieval_stage":
      return "Record Retrieval Stage";
    default:
      return String(method ?? "");
  }
}
