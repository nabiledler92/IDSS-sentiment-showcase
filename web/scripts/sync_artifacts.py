"""Copy the deployed model and its published results into the web app.

The Vercel project is rooted at web/, so it can't read files elsewhere in the
repository. This script copies what the web app needs from the repo root:

- web/backend/model/      the three files the Python API loads to predict
- web/frontend/public/figures/   charts shown on the Model Info page
- web/frontend/public/sample_reviews_synthetic.csv   the "Try the sample file" data
- web/frontend/data/model-info.json   metrics and the 20-result table

Run it from the repository root after re-exporting the models
(notebooks/extended_20_results/stage6_export_deployed.py):

    python web/scripts/sync_artifacts.py

`python web/scripts/sync_artifacts.py --check` exits non-zero if the copies
are out of date, without changing anything.
"""
import csv
import filecmp
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"

MODEL_FILES = ["best_model.pkl", "tfidf_vectorizer.pkl", "label_encoder.pkl", "best_model_name.txt"]
FIGURES = ["confusion_matrix_smote_only_lr.png", "full_20_results_comparison.png"]


def model_info() -> dict:
    results = json.loads((ROOT / "outputs/tables/model_results_all.json").read_text())
    best = max(results, key=lambda n: results[n]["test_f1_weighted"])
    with open(ROOT / "outputs/tables/extended_20_results/master_20_results.csv", newline="") as f:
        master = list(csv.DictReader(f))
    with open(ROOT / "outputs/tables/extended_20_results/mcnemar_20_results.csv", newline="") as f:
        mcnemar = list(csv.DictReader(f))

    def num(v):
        return float(v) if v not in ("", None) else None

    return {
        "deployedModel": (ROOT / "models/best_model_name.txt").read_text().strip(),
        "best": {
            "name": best,
            "accuracy": results[best]["test_accuracy"],
            "f1Macro": results[best]["test_f1_macro"],
            "f1Weighted": results[best]["test_f1_weighted"],
            "report": {
                k: v for k, v in results[best]["classification_report"].items()
                if k in ("Negative", "Neutral", "Positive")
            },
            "confusionMatrix": results[best]["confusion_matrix"],
        },
        "smoteOnly": [
            {
                "model": name,
                "accuracy": r["test_accuracy"],
                "f1Macro": r["test_f1_macro"],
                "f1Weighted": r["test_f1_weighted"],
            }
            for name, r in sorted(results.items(), key=lambda kv: -kv[1]["test_f1_weighted"])
        ],
        "twentyResults": [
            {
                "condition": row["Condition"],
                "model": row["Model"],
                "accuracy": num(row["Accuracy"]),
                "f1Macro": num(row["F1 (macro)"]),
                "f1Weighted": num(row["F1 (weighted)"]),
            }
            for row in master
        ],
        "mcnemar": [
            {
                "comparison": row["Comparison"],
                "pValue": num(row["p-value"]),
                "significant": row["Significant at alpha=0.05"] == "Yes",
            }
            for row in mcnemar
        ],
    }


def planned_copies():
    for name in MODEL_FILES:
        yield ROOT / "models" / name, WEB / "backend/model" / name
    for name in FIGURES:
        yield ROOT / "outputs/figures" / name, WEB / "frontend/public/figures" / name
    yield ROOT / "data/sample_reviews_synthetic.csv", WEB / "frontend/public/sample_reviews_synthetic.csv"


def main() -> int:
    check = "--check" in sys.argv
    stale = []
    for src, dst in planned_copies():
        if not dst.exists() or not filecmp.cmp(src, dst, shallow=False):
            stale.append(dst)
            if not check:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    info_path = WEB / "frontend/data/model-info.json"
    info_text = json.dumps(model_info(), indent=2) + "\n"
    if not info_path.exists() or info_path.read_text() != info_text:
        stale.append(info_path)
        if not check:
            info_path.parent.mkdir(parents=True, exist_ok=True)
            info_path.write_text(info_text)

    for path in stale:
        print(("out of date: " if check else "updated: ") + str(path.relative_to(ROOT)))
    if check and stale:
        print("Run: python web/scripts/sync_artifacts.py")
        return 1
    if not stale:
        print("web/ artifacts are up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
