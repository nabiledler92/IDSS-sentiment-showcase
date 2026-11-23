"use client";

import { useEffect, useRef, useState } from "react";

/*
 * Small SVG bar charts. Following the dataviz rules: one axis, thin bars with
 * rounded data ends on a shared baseline, a 2px gap between adjacent bars,
 * direct value labels, a legend for two series, a hover tooltip with a hit
 * target larger than the bar, and a table alongside for exact numbers.
 * Text uses ink colours, never the series colour.
 */

type Series = { name: string; color: string; values: number[] };

const DEFAULT_W = 560;
const PAD = { top: 28, right: 8, bottom: 34, left: 8 };

function niceMax(max: number) {
  if (max <= 5) return 5;
  const p = 10 ** Math.floor(Math.log10(max));
  return Math.ceil(max / p) * p;
}

function roundedTopBar(x: number, y: number, w: number, h: number, r: number) {
  if (h <= 0) return "";
  const rr = Math.min(r, w / 2, h);
  return `M${x},${y + h}V${y + rr}Q${x},${y} ${x + rr},${y}H${x + w - rr}Q${x + w},${y} ${x + w},${y + rr}V${y + h}Z`;
}

export function BarChart({
  title,
  categories,
  series,
  height = 260,
  unit = "reviews",
}: {
  title: string;
  categories: string[];
  series: Series[];
  height?: number;
  unit?: string;
}) {
  const [hover, setHover] = useState<{ ci: number; si: number } | null>(null);
  // Draw at the container's real width so text stays at its true pixel size
  // on phones instead of shrinking with a scaled viewBox.
  const ref = useRef<HTMLElement>(null);
  const [W, setW] = useState(DEFAULT_W);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setW(Math.max(260, Math.round(entry.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const innerW = W - PAD.left - PAD.right;
  const innerH = height - PAD.top - PAD.bottom;
  const max = niceMax(Math.max(1, ...series.flatMap((s) => s.values)));
  const groupW = innerW / categories.length;
  const gap = 2;
  const barW = Math.min(64, (groupW * 0.6 - gap * (series.length - 1)) / series.length);
  const groupBarsW = barW * series.length + gap * (series.length - 1);
  const yOf = (v: number) => PAD.top + innerH - (v / max) * innerH;

  const tip =
    hover &&
    (() => {
      const s = series[hover.si];
      const v = s.values[hover.ci];
      const x = PAD.left + hover.ci * groupW + (groupW - groupBarsW) / 2 + hover.si * (barW + gap) + barW / 2;
      return { left: `${(x / W) * 100}%`, top: `${(yOf(v) / height) * 100}%`, text: `${categories[hover.ci]}${series.length > 1 ? ` · ${s.name}` : ""}: ${v} ${unit}` };
    })();

  return (
    <figure ref={ref} className={`chart${hover ? " hovering" : ""}`} style={{ margin: 0 }}>
      <figcaption className="section-title">{title}</figcaption>
      {series.length > 1 ? (
        <div className="legend">
          {series.map((s) => (
            <span key={s.name}>
              <i style={{ background: s.color }} />
              {s.name}
            </span>
          ))}
        </div>
      ) : null}
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`${title}. ${categories.map((c, ci) => `${c}: ${series.map((s) => `${series.length > 1 ? s.name + " " : ""}${s.values[ci]}`).join(", ")}`).join("; ")}`}>
        {[0, 0.5, 1].map((f) => (
          <line key={f} className="grid-line" x1={PAD.left} x2={W - PAD.right} y1={yOf(max * f)} y2={yOf(max * f)} />
        ))}
        {categories.map((c, ci) => {
          const gx = PAD.left + ci * groupW + (groupW - groupBarsW) / 2;
          return (
            <g key={c}>
              {series.map((s, si) => {
                const v = s.values[ci];
                const x = gx + si * (barW + gap);
                const y = yOf(v);
                const active = hover?.ci === ci && hover?.si === si;
                return (
                  <g key={s.name}>
                    <path className={`bar${active ? " active" : ""}`} d={roundedTopBar(x, y, barW, PAD.top + innerH - y, 4)} style={{ fill: s.color }} />
                    <text className="value-label" x={x + barW / 2} y={y - 7} textAnchor="middle">
                      {v}
                    </text>
                    <rect
                      className="hit"
                      x={x - gap / 2}
                      y={PAD.top}
                      width={barW + gap}
                      height={innerH}
                      onMouseEnter={() => setHover({ ci, si })}
                      onMouseLeave={() => setHover(null)}
                    />
                  </g>
                );
              })}
              <text className="axis-label" x={PAD.left + ci * groupW + groupW / 2} y={height - 10} textAnchor="middle">
                {c}
              </text>
            </g>
          );
        })}
        <line x1={PAD.left} x2={W - PAD.right} y1={PAD.top + innerH} y2={PAD.top + innerH} stroke="var(--muted)" />
      </svg>
      {tip ? (
        <div className="tooltip" style={{ left: tip.left, top: tip.top }}>
          {tip.text}
        </div>
      ) : null}
    </figure>
  );
}
