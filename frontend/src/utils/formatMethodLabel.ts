import type { ComplianceMethod } from "../types";

type MethodValue = ComplianceMethod | string | null | undefined;

export function formatMethodLabel(method: MethodValue): string {
  switch (method) {
    case "non_rag":
      return "Non-RAG";
    case "single_source_rag":
    case "simple_rag":
      return "Single-source RAG";
    case "multi_source_rag":
    case "nested_rag":
      return "Multi-source RAG";
    default:
      return String(method ?? "");
  }
}
