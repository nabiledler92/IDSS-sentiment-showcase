import type { Metadata } from "next";
import Image from "next/image";
import info from "@/data/model-info.json";

export const metadata: Metadata = {
  title: "Model info",
  description:
    "How the deployed Logistic Regression with SMOTE model was chosen: 6 classifiers, 5 experimental conditions, 20 model-condition results.",
};

const pct = (v: number | null) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const f3 = (v: number | null) => (v == null ? "—" : v.toFixed(3));

export default function ModelPage() {
  const b = info.best;
  const conditions = Array.from(new Set(info.twentyResults.map((r) => r.condition)));
  return (
    <>
      <h2>About the active model</h2>
      <div className="section stack">
        <p>
          <strong>{info.deployedModel}</strong>, trained on 1,197 cleaned Kopi Kenangan app reviews with TF-IDF features
          (unigrams and bigrams) on a SMOTE-balanced training set, using default hyperparameters. In the dissertation&apos;s
          20-result comparison this configuration beat the GridSearchCV-tuned version and both Random Forest + XGBoost
          ensembles.
        </p>
        <div className="metric-grid">
          <Metric label="Accuracy" value={pct(b.accuracy)} note="held-out test set, 240 reviews" />
          <Metric label="Weighted F1" value={f3(b.f1Weighted)} note="main selection criterion" />
          <Metric label="Macro F1" value={f3(b.f1Macro)} note="treats each class equally" />
        </div>
      </div>

      <section className="section">
        <h3 className="section-title" style={{ fontFamily: "var(--font-body)" }}>
          Per-class results
        </h3>
        <div className="chart-grid">
          <table className="data">
            <thead>
              <tr>
                <th>Class</th>
                <th className="num">Precision</th>
                <th className="num">Recall</th>
                <th className="num">F1</th>
                <th className="num">Test reviews</th>
              </tr>
            </thead>
            <tbody>
              {(["Negative", "Neutral", "Positive"] as const).map((c) => (
                <tr key={c}>
                  <td>{c}</td>
                  <td className="num">{f3(b.report[c].precision)}</td>
                  <td className="num">{f3(b.report[c].recall)}</td>
                  <td className="num">{f3(b.report[c]["f1-score"])}</td>
                  <td className="num">{b.report[c].support}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <figure className="figure" style={{ margin: 0 }}>
            <Image
              src="/figures/confusion_matrix_smote_only_lr.png"
              alt="Confusion matrix on the 240-review test set. Negative: 14 correct, 1 predicted Neutral, 9 predicted Positive. Neutral: 1 predicted Negative, 3 correct, 1 predicted Positive. Positive: 5 predicted Negative, 206 correct."
              width={1000}
              height={1000}
              sizes="(max-width: 820px) 100vw, 480px"
            />
          </figure>
        </div>
        <p className="caption" style={{ marginTop: 10 }}>
          Neutral is the weakest class: only 26 of the 1,197 reviews were 3-star, so the model saw very few examples.
        </p>
      </section>

      <section className="section">
        <h3 className="section-title" style={{ fontFamily: "var(--font-body)" }}>
          All 20 model-condition results
        </h3>
        <p className="caption" style={{ marginBottom: 10 }}>
          Six classifiers under baseline, SMOTE-only and SMOTE + GridSearchCV conditions, plus two Random Forest + XGBoost
          ensembles. The deployed model is highlighted.
        </p>
        <div className="table-scroll" style={{ maxHeight: "none" }}>
          <table className="data">
            <thead>
              <tr>
                <th>Condition</th>
                <th>Model</th>
                <th className="num">Accuracy</th>
                <th className="num">Macro F1</th>
                <th className="num">Weighted F1</th>
              </tr>
            </thead>
            <tbody>
              {conditions.flatMap((cond) =>
                info.twentyResults
                  .filter((r) => r.condition === cond)
                  .map((r, i) => (
                    <tr
                      key={`${cond}-${r.model}`}
                      className={cond === "(b) SMOTE only" && r.model === b.name ? "highlight" : undefined}
                    >
                      <td>{i === 0 ? cond : ""}</td>
                      <td>{r.model}</td>
                      <td className="num">{pct(r.accuracy)}</td>
                      <td className="num">{f3(r.f1Macro)}</td>
                      <td className="num">{f3(r.f1Weighted)}</td>
                    </tr>
                  )),
              )}
            </tbody>
          </table>
        </div>
        <figure className="figure section" style={{ marginInline: 0 }}>
          <Image
            src="/figures/full_20_results_comparison.png"
            alt="Bar charts of test accuracy and weighted F1 for all 20 results. Logistic Regression with SMOTE only is the tallest bar on both."
            width={3581}
            height={1472}
            sizes="(max-width: 1180px) 100vw, 1100px"
          />
        </figure>
      </section>

      <section className="section">
        <h3 className="section-title" style={{ fontFamily: "var(--font-body)" }}>
          Single model vs ensembles (McNemar&apos;s test)
        </h3>
        <table className="data">
          <thead>
            <tr>
              <th>Comparison</th>
              <th className="num">p-value</th>
              <th>Significant at 0.05?</th>
            </tr>
          </thead>
          <tbody>
            {info.mcnemar.map((m) => (
              <tr key={m.comparison}>
                <td>{m.comparison}</td>
                <td className="num">{m.pValue?.toFixed(3)}</td>
                <td>{m.significant ? "Yes" : "No"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <p className="caption section">
        Want to try your own data? The{" "}
        <a href="https://kopi-sentiment-idss.streamlit.app/">Streamlit version</a> can retrain all six classifiers on any
        review export. Full method and code are on{" "}
        <a href="https://github.com/nabiledler92/IDSS-sentiment-showcase">GitHub</a>.
      </p>
    </>
  );
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="kpi" style={{ "--accent": "var(--primary)" } as React.CSSProperties}>
      <p className="kpi-label">{label}</p>
      <p className="kpi-value">{value}</p>
      <p className="kpi-sub">{note}</p>
    </div>
  );
}
