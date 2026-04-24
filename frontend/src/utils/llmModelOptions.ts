export const MODEL_OPTIONS_BY_PROVIDER: Record<string, string[]> = {
  openai: ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"],
  gemini: ["gemini-3.1-flash-lite-preview", "gemini-2.5-flash", "gemini-3.1-pro-preview"],
};

export function getModelOptions(provider: string, currentModel?: string): string[] {
  const configured = MODEL_OPTIONS_BY_PROVIDER[provider] ?? [];
  return [...new Set(currentModel ? [currentModel, ...configured] : configured)];
}
