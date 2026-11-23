"use client";

import { useEffect, useState } from "react";
import { ask, getBrief, getMeta, type Meta, type Stats } from "@/lib/api";
import { Prose } from "./Prose";

export function useMeta() {
  const [meta, setMeta] = useState<Meta | null>(null);
  useEffect(() => {
    let live = true;
    getMeta()
      .then((m) => live && setMeta(m))
      .catch(() => live && setMeta(null));
    return () => {
      live = false;
    };
  }, []);
  return meta;
}

function Off({ meta }: { meta: Meta | null }) {
  return (
    <p className="caption">
      The AI features aren&apos;t switched on right now. When they are, an open-source model (
      <code>{meta?.aiModel ?? "Qwen/Qwen2.5-72B-Instruct"}</code>, through Hugging Face) reads only the
      numbers on this page, never the reviews themselves.
    </p>
  );
}

export function AiBrief({ meta, batch }: { meta: Meta | null; batch: { stats: Stats; signature: string } | null }) {
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // State resets for a new batch because the parent keys this component by
  // the batch signature.
  async function run() {
    if (!batch) return;
    setBusy(true);
    setError(null);
    try {
      setText((await getBrief(batch.stats, batch.signature)).text);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't generate a brief right now.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section aria-labelledby="ai-brief-title" className="stack">
      <h3 id="ai-brief-title" className="section-title" style={{ fontFamily: "var(--font-body)", marginBottom: 0 }}>
        AI insights brief
      </h3>
      {meta && !meta.aiEnabled ? (
        <Off meta={meta} />
      ) : (
        <>
          <div className="row">
            <button className="btn btn-ghost" onClick={run} disabled={!batch || busy}>
              {busy ? (
                <>
                  <span className="spinner" aria-hidden="true" />
                  Asking the AI model…
                </>
              ) : text ? (
                "Regenerate AI insights"
              ) : (
                "Generate AI insights"
              )}
            </button>
            {!batch ? <span className="caption">Upload a file and run the batch analysis to enable this.</span> : null}
          </div>
          <div aria-live="polite">
            {error ? <p className="notice notice-error">{error}</p> : null}
            {text ? (
              <div className="ai-card">
                <span className="kt-badge" style={{ color: "var(--ink)" }}>
                  AI-generated
                </span>
                <div style={{ marginTop: 10 }}>
                  <Prose text={text} />
                </div>
              </div>
            ) : null}
          </div>
          {text ? (
            <p className="caption">
              Written by an open-source model ({meta?.aiModel}) through Hugging Face: a narrative read on the numbers
              below, not a substitute for them.
            </p>
          ) : null}
        </>
      )}
    </section>
  );
}

type Turn = { q: string; a?: string; error?: string };

export function QaChat({ meta, batch }: { meta: Meta | null; batch: { stats: Stats; signature: string } }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;
    setBusy(true);
    setQuestion("");
    setTurns((t) => [...t, { q }]);
    try {
      const { text } = await ask(batch.stats, batch.signature, q);
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, a: text } : x)));
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Couldn't answer right now.";
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, error: msg } : x)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section aria-labelledby="qa-title" className="stack">
      <h3 id="qa-title" className="section-title" style={{ fontFamily: "var(--font-body)", marginBottom: 0 }}>
        Ask a follow-up question
      </h3>
      {meta && !meta.aiEnabled ? (
        <Off meta={meta} />
      ) : (
        <>
          <p className="caption">Answers use only the statistics from this batch, so the model can&apos;t invent figures.</p>
          {turns.length ? (
            <div className="chat" aria-live="polite">
              {turns.map((t, i) => (
                <div key={i} className="stack" style={{ gap: 8 }}>
                  <div className="q">{t.q}</div>
                  {t.a ? (
                    <div className="a">
                      <Prose text={t.a} />
                    </div>
                  ) : t.error ? (
                    <p className="notice notice-error">{t.error}</p>
                  ) : (
                    <div className="a caption">
                      <span className="spinner" aria-hidden="true" />
                      Thinking…
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : null}
          <form className="chat-form" onSubmit={submit}>
            <label className="field" style={{ gap: 0 }}>
              <span className="skip-link">Your question</span>
              <input
                type="text"
                value={question}
                maxLength={300}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="e.g. Which category should we fix first?"
              />
            </label>
            <button className="btn" type="submit" disabled={busy || !question.trim()}>
              Ask
            </button>
          </form>
        </>
      )}
    </section>
  );
}
