import { useEffect, useRef } from "react";
import type { FormEvent } from "react";

import { documentTypeLabels, documentTypes, languageLabels, languages } from "../constants";
import type { DocumentLanguage, DocumentType, LLMProviderDescriptor } from "../types";

type UploadPanelProps = {
  uploadFile: File | null;
  providers: LLMProviderDescriptor[];
  uploadType: DocumentType;
  uploadLanguage: DocumentLanguage;
  extractOnUpload: boolean;
  extractionProvider: string;
  extractionModel: string;
  groupId: string;
  busy: string;
  onSubmit: (event: FormEvent) => void | Promise<void>;
  onFileChange: (file: File | null) => void;
  onTypeChange: (value: DocumentType) => void;
  onLanguageChange: (value: DocumentLanguage) => void;
  onExtractOnUploadChange: (value: boolean) => void;
  onExtractionProviderChange: (value: string) => void;
  onExtractionModelChange: (value: string) => void;
  onGroupIdChange: (value: string) => void;
};

export function UploadPanel({
  uploadFile,
  providers,
  uploadType,
  uploadLanguage,
  extractOnUpload,
  extractionProvider,
  extractionModel,
  groupId,
  busy,
  onSubmit,
  onFileChange,
  onTypeChange,
  onLanguageChange,
  onExtractOnUploadChange,
  onExtractionProviderChange,
  onExtractionModelChange,
  onGroupIdChange,
}: UploadPanelProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const extractionEnabled = uploadType === "procedure" || uploadType === "reference";
  const extractionTargetLabel = uploadType === "reference" ? "reference" : "procedure";

  useEffect(() => {
    if (!uploadFile && fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, [uploadFile]);

  return (
    <section className="panel">
      <h2>Upload</h2>
      <form onSubmit={onSubmit} className="stack">
        <label className="field">
          <span>File</span>
          <input
            ref={fileInputRef}
            type="file"
            onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
          />
        </label>
        <label className="field">
          <span>Document type</span>
          <select value={uploadType} onChange={(e) => onTypeChange(e.target.value as DocumentType)}>
            {documentTypes.map((type) => <option key={type} value={type}>{documentTypeLabels[type]}</option>)}
          </select>
        </label>
        <div className={`field field-inline ${extractionEnabled ? "" : "field-disabled"}`}>
          <span>Extraction</span>
          <div className="check">
            <input
              type="checkbox"
              checked={extractionEnabled && extractOnUpload}
              disabled={!extractionEnabled}
              onChange={(e) => onExtractOnUploadChange(e.target.checked)}
            />
            <span>Run extraction after upload for {extractionTargetLabel} files</span>
          </div>
        </div>
        <div className="row">
          <label className={`field ${extractionEnabled ? "" : "field-disabled"}`}>
            <span>Extraction provider</span>
            <select
              value={extractionProvider}
              disabled={!extractionEnabled || !extractOnUpload}
              onChange={(e) => onExtractionProviderChange(e.target.value)}
            >
              {providers.map((item) => (
                <option key={item.key} value={item.key}>{item.label}</option>
              ))}
            </select>
          </label>
          <label className={`field ${extractionEnabled ? "" : "field-disabled"}`}>
            <span>Extraction model</span>
            <input
              value={extractionModel}
              disabled={!extractionEnabled || !extractOnUpload}
              onChange={(e) => onExtractionModelChange(e.target.value)}
            />
          </label>
        </div>
        <label className="field">
          <span>Language</span>
          <select value={uploadLanguage} onChange={(e) => onLanguageChange(e.target.value as DocumentLanguage)}>
            {languages.map((language) => <option key={language} value={language}>{languageLabels[language]}</option>)}
          </select>
        </label>
        <label className="field">
          <span>Group ID</span>
          <input value={groupId} onChange={(e) => onGroupIdChange(e.target.value)} placeholder="Optional" />
        </label>
        <button className="button" disabled={!uploadFile || busy === "upload"}>
          {busy === "upload" ? "Uploading..." : extractOnUpload && extractionEnabled ? "Upload and extract" : "Upload document"}
        </button>
      </form>
    </section>
  );
}
