import type { DocumentRecord } from "../types";

export function getFrozenBadgeLabel(doc: DocumentRecord): string | null {
  if (!doc.frozen) return null;
  return "Frozen";
}

type FrozenBadgeProps = {
  document: DocumentRecord;
};

export function FrozenBadge({ document }: FrozenBadgeProps) {
  const label = getFrozenBadgeLabel(document);
  if (!label) {
    return null;
  }

  return <span className="status-badge status-badge-frozen">{label}</span>;
}
