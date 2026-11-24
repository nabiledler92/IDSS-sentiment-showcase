/*
 * Pushes a custom event to Google Tag Manager's dataLayer. Event names and
 * parameters match the ones used on nabil-takiyuddin.dev so one set of GTM
 * triggers covers both sites.
 *
 * Labels only: never pass review text, AI questions or answers, file names
 * or file contents. Does nothing when GTM isn't loaded (local development,
 * previews, or a blocker).
 */
type Params = Record<string, string | number | boolean>;

export function track(event: string, params: Params = {}): void {
  if (typeof window === "undefined") return;
  const dl = (window as unknown as { dataLayer?: unknown[] }).dataLayer;
  if (!Array.isArray(dl)) return;
  dl.push({ event, ...params });
}
