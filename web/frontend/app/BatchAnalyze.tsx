"use client";

import { useRef, useState } from "react";
import { AiBrief, QaChat, useMeta } from "@/components/AI";
import { BarChart } from "@/components/Charts";
import { KpiRow } from "@/components/Sentiment";
import {
  LABELS,
  PALETTE,
  downloadBatchCsv,
  readColumns,
  runBatch,
  type BatchResult,
  type Columns,
  type Term,
} from "@/lib/api";

const CATEGORY_ORDER = ["Service", "Product Quality", "Price"];

export function BatchAnalyze() {
  const meta = useMeta();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [cols, setCols] = useState<Columns | null>(null);
  const [textCol, setTextCol] = useState("");
  const [stopwords, setStopwords] = useState(true);
  const [result, setResult] = useState<BatchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<"read" | "run" | "csv" | null>(null);
  const [drag, setDrag] = useState(false);

  async function choose(f: File | null | undefined) {
    if (!f) return;
    setFile(f);
    setResult(null);
    setCols(null);
    setError(null);
    setBusy("read");
    try {
      const c = await readColumns(f);
      setCols(c);
      setTextCol(c.guess);
    } catch (e) {
      setFile(null);
      setError(e instanceof Error ? e.message : "Couldn't read that file.");
    } finally {
      setBusy(null);
    }
  }

  async function useSample() {
    const res = await fetch("/sample_reviews_synthetic.csv");
    const blob = await res.blob();
    await choose(new File([blob], "sample_reviews_synthetic.csv", { type: "text/csv" }));
  }

  async function run() {
    if (!file || !textCol) return;
    setBusy("run");
    setError(null);
    try {
      setResult(await runBatch(file, textCol, stopwords));
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : "Couldn't analyse that file.");
    } finally {
      setBusy(null);
    }
  }

  async function download() {
    if (!file) return;
    setBusy("csv");
    try {
      const blob = await downloadBatchCsv(file, textCol, stopwords);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "sentiment_results.csv";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't prepare the download.");
    } finally {
      setBusy(null);
    }
  }

  const batchKey = result ? { stats: result.stats, signature: result.signature } : null;

  return (
    <div className="section">
      <AiBrief key={batchKey?.signature ?? "none"} meta={meta} batch={batchKey} />
      <hr className="divider" />

      <div className="stack">
        <label
          className={`dropzone${drag ? " drag" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            choose(e.dataTransfer.files?.[0]);
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv,.xlsx"
            onChange={(e) => choose(e.target.files?.[0])}
          />
          <strong>{file ? file.name : "Upload CSV or XLSX"}</strong>
          <span className="caption">
            Drag a file here or click to choose · up to {meta?.limits.maxFileMB ?? 4} MB and{" "}
            {(meta?.limits.maxRows ?? 5000).toLocaleString("en-MY")} rows · files are analysed in memory and never stored
          </span>
        </label>
        <div className="row caption">
          No file to hand?
          <button type="button" className="btn btn-ghost" style={{ minHeight: 36, padding: "6px 12px", fontSize: 14 }} onClick={useSample} disabled={busy !== null}>
            Try the sample file
          </button>
          <span>40 invented code-switched reviews</span>
        </div>

        {busy === "read" ? (
          <p className="caption">
            <span className="spinner" aria-hidden="true" />
            Reading the file…
          </p>
        ) : null}

        {cols ? (
          <>
            {cols.truncated ? (
              <p className="notice notice-warn">
                This file has more than {(meta?.limits.maxRows ?? 5000).toLocaleString("en-MY")} rows. Only the first{" "}
                {(meta?.limits.maxRows ?? 5000).toLocaleString("en-MY")} will be analysed.
              </p>
            ) : null}
            <label className="field">
              Which column contains the review text?
              <select value={textCol} onChange={(e) => setTextCol(e.target.value)}>
                {cols.columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
            <label className="check">
              <input type="checkbox" checked={stopwords} onChange={(e) => setStopwords(e.target.checked)} />
              Text is Indonesian/Malay (apply stop-word removal)
            </label>
            <div>
              <button className="btn" onClick={run} disabled={busy !== null}>
                {busy === "run" ? (
                  <>
                    <span className="spinner" aria-hidden="true" />
                    Scoring reviews…
                  </>
                ) : (
                  "Run batch analysis"
                )}
              </button>
            </div>
          </>
        ) : null}

        <div aria-live="polite">{error ? <p className="notice notice-error">{error}</p> : null}</div>
      </div>

      {result ? <Results result={result} meta={meta} onDownload={download} downloading={busy === "csv"} batch={batchKey!} /> : null}
    </div>
  );
}

function Results({
  result,
  meta,
  onDownload,
  downloading,
  batch,
}: {
  result: BatchResult;
  meta: ReturnType<typeof useMeta>;
  onDownload: () => void;
  downloading: boolean;
  batch: { stats: BatchResult["stats"]; signature: string };
}) {
  const present = new Set(LABELS.filter((l) => result.counts[l] > 0));
  const cats = CATEGORY_ORDER.filter((c) => c in result.categoriesAll);
  const anyCategory = cats.some((c) => result.categoriesAll[c] > 0);

  return (
    <div className="section stack" style={{ gap: 28 }}>
      <p className="notice notice-ok" role="status">
        Scored {result.total.toLocaleString("en-MY")} reviews.
      </p>
      <KpiRow total={result.total} counts={result.counts} />
      <QaChat key={batch.signature} meta={meta} batch={batch} />

      <div className="chart-grid">
        <BarChart
          title="Sentiment breakdown"
          categories={LABELS}
          series={[{ name: "Reviews", color: "var(--primary)", values: LABELS.map((l) => result.counts[l]) }]}
        />
        <div>
          <p className="section-title">Class breakdown</p>
          <table className="data">
            <thead>
              <tr>
                <th>Predicted sentiment</th>
                <th className="num">Reviews</th>
                <th className="num">Share</th>
              </tr>
            </thead>
            <tbody>
              {LABELS.map((l) => (
                <tr key={l}>
                  <td>
                    <span aria-hidden="true" style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: PALETTE[l].accent, marginRight: 8 }} />
                    {l}
                  </td>
                  <td className="num">{result.counts[l]}</td>
                  <td className="num">{result.total ? Math.round((100 * result.counts[l]) / result.total) : 0}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {anyCategory ? (
        <div className="chart-grid">
          <BarChart
            title="Review categories mentioned"
            categories={cats}
            series={[
              { name: "All reviews", color: "var(--series-all)", values: cats.map((c) => result.categoriesAll[c] ?? 0) },
              { name: "Negative reviews", color: "var(--negative)", values: cats.map((c) => result.categoriesNegative[c] ?? 0) },
            ]}
          />
          <div className="stack">
            <p className="caption" style={{ marginTop: 34 }}>
              Service, Product Quality and Price, tagged by bilingual keyword rules on the cleaned review text. A review can
              mention more than one category, so these counts don&apos;t add up to the total. A Negative bar that&apos;s tall
              relative to its All-reviews bar marks a category tied to complaints.
            </p>
            <table className="data">
              <thead>
                <tr>
                  <th>Category</th>
                  <th className="num">All reviews</th>
                  <th className="num">Negative</th>
                </tr>
              </thead>
              <tbody>
                {cats.map((c) => (
                  <tr key={c}>
                    <td>{c}</td>
                    <td className="num">{result.categoriesAll[c] ?? 0}</td>
                    <td className="num">{result.categoriesNegative[c] ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      {result.negativeCount ? (
        <div>
          <p className="section-title">
            {result.negativeCount} reviews flagged Negative: priority follow-up list
          </p>
          {result.negativeCount > result.negatives.length ? (
            <p className="caption" style={{ marginBottom: 8 }}>
              Showing the {result.negatives.length} most confident. Download the CSV for all of them.
            </p>
          ) : null}
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th>Review</th>
                  <th className="num">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {result.negatives.map((n, i) => (
                  <tr key={i}>
                    <td>{n.text}</td>
                    <td className="num">{Math.round(n.confidence * 100)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      <div>
        <button className="btn btn-ghost" onClick={onDownload} disabled={downloading}>
          {downloading ? (
            <>
              <span className="spinner" aria-hidden="true" />
              Preparing CSV…
            </>
          ) : (
            "Download full results (CSV)"
          )}
        </button>
      </div>

      {present.has("Negative") || present.has("Positive") ? (
        <>
          <hr className="divider" style={{ margin: 0 }} />
          <div>
            <h2 style={{ fontSize: 24 }}>Theme highlights from this batch</h2>
            <p className="caption" style={{ marginTop: 6 }}>
              Recurring words and phrases that show up disproportionately in one sentiment class compared with the rest of
              the batch: concrete issues to fix and strengths to promote.
            </p>
          </div>
          <div className="themes">
            <ThemeList title="Top issues to improve" from="Negative" present={present.has("Negative")} terms={result.negTerms} />
            <ThemeList title="Top strengths to scale" from="Positive" present={present.has("Positive")} terms={result.posTerms} />
          </div>
        </>
      ) : null}
    </div>
  );
}

function ThemeList({ title, from, present, terms }: { title: string; from: "Negative" | "Positive"; present: boolean; terms: Term[] }) {
  return (
    <div>
      <p className="theme-head">
        <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: PALETTE[from].accent, marginRight: 8 }} aria-hidden="true" />
        {title} <span className="caption">(from {from} reviews)</span>
      </p>
      {!present ? (
        <p className="notice notice-info">No {from} reviews in this batch.</p>
      ) : !terms.length ? (
        <p className="notice notice-info">Not enough {from} reviews to extract reliable themes.</p>
      ) : (
        terms.map((t) => (
          <details className="term" key={t.term}>
            <summary>
              &ldquo;{t.term}&rdquo;: {t.count} reviews ({t.pct_of_class}% of {from})
            </summary>
            <ul>
              {t.examples.map((ex, i) => (
                <li key={i}>{ex}</li>
              ))}
            </ul>
          </details>
        ))
      )}
    </div>
  );
}
