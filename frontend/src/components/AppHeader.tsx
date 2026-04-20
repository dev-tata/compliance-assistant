type AppHeaderProps = {
  documentCount: number;
  caseCount: number;
  hasLiveCompliance: boolean;
};

export function AppHeader({
  documentCount,
  caseCount,
  hasLiveCompliance,
}: AppHeaderProps) {
  return (
    <header className="hero">
      <div className="hero-copy">
        <p className="eyebrow">compliance assistant</p>
        <h1>Quality Assurance Workbench</h1>
        <p className="lede">
          Upload documents, assemble cases, and run compliance analysis.
        </p>
      </div>
      <div className="hero-card">
        <div><strong>{documentCount}</strong><span>Documents</span></div>
        <div><strong>{caseCount}</strong><span>Cases</span></div>
        <div><strong>{hasLiveCompliance ? "Live" : "Idle"}</strong><span>Compliance</span></div>
      </div>
    </header>
  );
}
