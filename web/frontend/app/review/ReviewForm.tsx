"use client";

import { useState } from "react";
import { SentimentCard } from "@/components/Sentiment";
import { predict, type Prediction } from "@/lib/api";

const EXAMPLES = [
  "Order sampai lambat sejam, air dah cair. Sangat kecewa dengan servis",
  "Kopi sedap gila, the aren latte memang best. Will order again!",
  "Kopi ok but a bit pricey compared to other kedai",
];

export function ReviewForm() {
  const [text, setText] = useState("");
  const [stopwords, setStopwords] = useState(true);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await predict(text, stopwords));
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Couldn't analyse that review.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="stack section" onSubmit={submit}>
      <label className="field">
        Paste a customer review
        <textarea
          value={text}
          maxLength={2000}
          onChange={(e) => setText(e.target.value)}
          placeholder="e.g. Aplikasi nya bagus, tapi sering error saat checkout..."
        />
      </label>
      <div className="row caption">
        Try an example:
        {EXAMPLES.map((ex) => (
          <button key={ex} type="button" className="btn btn-ghost" style={{ minHeight: 36, padding: "6px 12px", fontSize: 14 }} onClick={() => setText(ex)}>
            {ex.length > 34 ? ex.slice(0, 32) + "…" : ex}
          </button>
        ))}
      </div>
      <label className="check">
        <input type="checkbox" checked={stopwords} onChange={(e) => setStopwords(e.target.checked)} />
        Text is Indonesian/Malay (apply stop-word removal)
      </label>
      <div>
        <button className="btn" type="submit" disabled={busy || !text.trim()}>
          {busy ? (
            <>
              <span className="spinner" aria-hidden="true" />
              Analysing…
            </>
          ) : (
            "Analyze sentiment"
          )}
        </button>
      </div>
      <div aria-live="polite">
        {error ? <p className="notice notice-error">{error}</p> : null}
        {result ? <SentimentCard result={result} /> : null}
      </div>
    </form>
  );
}
