import type { FormEvent } from "react";

import type { CaseRecord, ComplianceMethod, LLMProviderDescriptor } from "../types";

type CompliancePanelProps = {
  cases: CaseRecord[];
  providers: LLMProviderDescriptor[];
  selectedCaseId: string;
  provider: string;
  model: string;
  method: ComplianceMethod;
  instructions: string;
  busy: string;
  onSubmit: (event: FormEvent) => void | Promise<void>;
  onSelectCase: (caseId: string) => void;
  onProviderChange: (value: string) => void;
  onModelChange: (value: string) => void;
  onMethodChange: (value: ComplianceMethod) => void;
  onInstructionsChange: (value: string) => void;
};

export function CompliancePanel({
  cases,
  providers,
  selectedCaseId,
  provider,
  model,
  method,
  instructions,
  busy,
  onSubmit,
  onSelectCase,
  onProviderChange,
  onModelChange,
  onMethodChange,
  onInstructionsChange,
}: CompliancePanelProps) {
  return (
    <section className="panel panel-wide">
      <h2>Run compliance</h2>
      <form onSubmit={onSubmit} className="stack">
        <label className="field">
          <span>Case</span>
          <select value={selectedCaseId} onChange={(e) => onSelectCase(e.target.value)}>
            <option value="">Select case</option>
            {cases.map((item) => <option key={item.case_id} value={item.case_id}>{item.title}</option>)}
          </select>
        </label>
        <div className="row">
          <label className="field">
            <span>Compliance provider</span>
            <select value={provider} onChange={(e) => onProviderChange(e.target.value)}>
              {providers.map((item) => (
                <option key={item.key} value={item.key}>{item.label}</option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Compliance model</span>
            <input value={model} onChange={(e) => onModelChange(e.target.value)} />
          </label>
        </div>
        <label className="field">
          <span>Method</span>
          <select value={method} onChange={(e) => onMethodChange(e.target.value as ComplianceMethod)}>
            <option value="non_rag">Non-RAG</option>
            <option value="simple_rag">Simple RAG</option>
          </select>
        </label>
        <label className="field">
          <span>Instructions</span>
          <textarea value={instructions} onChange={(e) => onInstructionsChange(e.target.value)} rows={5} />
        </label>
        <button className="button" disabled={!selectedCaseId || busy === "compliance"}>
          {busy === "compliance" ? "Running..." : "Run compliance"}
        </button>
      </form>
    </section>
  );
}
