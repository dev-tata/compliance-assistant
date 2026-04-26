export function formatOverallAssessment(value: string): string {
  const normalized = value.trim();
  const match = normalized.match(/^Completed_(\d+)_(\d+)$/);
  if (match) {
    return `Completed ${match[1]}-${match[2]}`;
  }
  return normalized.replace(/_/g, " ");
}
