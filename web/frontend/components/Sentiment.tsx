import { LABELS, PALETTE, type Label, type Prediction } from "@/lib/api";

const ORDER: Label[] = ["Positive", "Neutral", "Negative"];

export function SentimentCard({ result }: { result: Prediction }) {
  const pal = PALETTE[result.label];
  return (
    <div
      className="kt-card"
      role="status"
      style={{ "--accent": pal.accent, "--tint": pal.bg, "--text": pal.text } as React.CSSProperties}
    >
      <span className="kt-badge">{result.label}</span>
      <p className="kt-conf">
        {Math.round(result.confidence * 100)}% <small>confidence</small>
      </p>
      <p className="kt-rec">{result.recommendation}</p>
      {ORDER.map((label) => {
        const pct = (result.probabilities[label] ?? 0) * 100;
        return (
          <div className="prob-row" key={label}>
            <span>{label}</span>
            <div
              className="prob-track"
              role="meter"
              aria-label={`${label} probability`}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(pct)}
            >
              <div className="prob-fill" style={{ width: `${pct}%`, background: PALETTE[label].accent }} />
            </div>
            <span>{Math.round(pct)}%</span>
          </div>
        );
      })}
    </div>
  );
}

export function KpiRow({ total, counts }: { total: number; counts: Record<Label, number> }) {
  return (
    <div className="kpis">
      <Kpi label="Total reviews" value={total} sub="scored this batch" accent="var(--primary)" />
      {LABELS.map((l) => (
        <Kpi
          key={l}
          label={l}
          value={counts[l]}
          sub={`${total ? Math.round((100 * counts[l]) / total) : 0}% of batch`}
          accent={PALETTE[l].accent}
        />
      ))}
    </div>
  );
}

function Kpi({ label, value, sub, accent }: { label: string; value: number; sub: string; accent: string }) {
  return (
    <div className="kpi" style={{ "--accent": accent } as React.CSSProperties}>
      <p className="kpi-label">{label}</p>
      <p className="kpi-value">{value.toLocaleString("en-MY")}</p>
      <p className="kpi-sub">{sub}</p>
    </div>
  );
}
