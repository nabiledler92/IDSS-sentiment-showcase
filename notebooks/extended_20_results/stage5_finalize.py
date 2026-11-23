import warnings
warnings.filterwarnings("ignore")
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from statsmodels.stats.contingency_tables import mcnemar

BASE_DIR = Path(__file__).resolve().parent          # .../notebooks/extended_20_results
PROJECT_DIR = BASE_DIR.parent.parent                # repo root
CKPT_DIR = BASE_DIR / "checkpoints"
OUT_TABLES = PROJECT_DIR / "outputs" / "tables" / "extended_20_results"
OUT_FIGS = PROJECT_DIR / "outputs" / "figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

d = joblib.load(CKPT_DIR / "prep.pkl")
y_test = d["y_test"]
le = d["le"]

s2 = joblib.load(CKPT_DIR / "stage2_results.pkl")
s3 = joblib.load(CKPT_DIR / "stage3_tuned_partial.pkl")
s4 = joblib.load(CKPT_DIR / "stage4_ensembles.pkl")

all_rows = list(s2["rows"]) + list(s3["rows"]) + list(s4["rows"])
all_preds = {}
all_preds.update(s2["preds"])
all_preds.update(s3["preds"])
all_preds.update(s4["preds"])

master = pd.DataFrame(all_rows)
master.to_csv(OUT_TABLES / "master_20_results.csv", index=False)

print("=" * 100)
print(f"MASTER TABLE -- {len(master)} results")
print("=" * 100)
print(master[["Condition", "Model", "Accuracy", "F1 (macro)", "F1 (weighted)"]].to_string(index=False))

best_row = master.sort_values("F1 (weighted)", ascending=False).iloc[0]
print("\nBEST OVERALL BY WEIGHTED F1:", best_row["Condition"], "-", best_row["Model"], "| weighted F1 =", best_row["F1 (weighted)"])

lr_key = "(c) Logistic Regression"
lr_pred = all_preds[lr_key]
mcnemar_rows = []
for label, key in [("RF+XGBoost Voting (bagging-style)", "(d) RF+XGB Voting"),
                    ("RF+XGBoost Stacking (boosting-style)", "(e) RF+XGB Stacking")]:
    ens_pred = all_preds[key]
    lr_correct = (lr_pred == y_test)
    ens_correct = (ens_pred == y_test)
    both_correct = int(np.sum(lr_correct & ens_correct))
    lr_only = int(np.sum(lr_correct & ~ens_correct))
    ens_only = int(np.sum(~lr_correct & ens_correct))
    both_wrong = int(np.sum(~lr_correct & ~ens_correct))
    table = [[both_correct, lr_only], [ens_only, both_wrong]]
    result = mcnemar(table, exact=(lr_only + ens_only) < 25, correction=True)
    mcnemar_rows.append({
        "Comparison": f"Logistic Regression (tuned) vs {label}",
        "LR-only correct (n)": lr_only, "Ensemble-only correct (n)": ens_only,
        "Statistic": round(float(result.statistic), 4), "p-value": round(float(result.pvalue), 5),
        "Significant at alpha=0.05": "Yes" if result.pvalue < 0.05 else "No",
    })

mcnemar_df = pd.DataFrame(mcnemar_rows)
mcnemar_df.to_csv(OUT_TABLES / "mcnemar_20_results.csv", index=False)
print("\n" + "=" * 100)
print("McNEMAR'S TEST -- Logistic Regression (tuned) vs both ensemble combinations")
print("=" * 100)
print(mcnemar_df.to_string(index=False))

summary = {
    "n_results": len(master),
    "best_overall": {"condition": best_row["Condition"], "model": best_row["Model"], "weighted_f1": float(best_row["F1 (weighted)"])},
    "logistic_regression_tuned": master[(master.Condition == "(c) SMOTE + GridSearchCV") & (master.Model == "Logistic Regression")][["Accuracy","F1 (macro)","F1 (weighted)"]].iloc[0].to_dict(),
    "voting_bagging_style": master[master.Condition == "(d) Ensemble - Bagging-style"][["Accuracy","F1 (macro)","F1 (weighted)"]].iloc[0].to_dict(),
    "stacking_boosting_style": master[master.Condition == "(e) Ensemble - Boosting-style"][["Accuracy","F1 (macro)","F1 (weighted)"]].iloc[0].to_dict(),
    "app_best_model_changes": bool(best_row["Model"] != "Logistic Regression"),
}
with open(OUT_TABLES / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("\nSUMMARY:", json.dumps(summary, indent=2, default=str))

# ---- figure: full 20-result comparison, grouped by condition ----
cond_order = ["(a) Baseline", "(b) SMOTE only", "(c) SMOTE + GridSearchCV",
              "(d) Ensemble - Bagging-style", "(e) Ensemble - Boosting-style"]
colors_map = {"(a) Baseline": "#9AA5B1", "(b) SMOTE only": "#5B8DB8",
              "(c) SMOTE + GridSearchCV": "#1E2761", "(d) Ensemble - Bagging-style": "#B85042",
              "(e) Ensemble - Boosting-style": "#E0A96D"}

fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
x_labels = []
acc_vals, f1w_vals, colors = [], [], []
for cond in cond_order:
    sub = master[master.Condition == cond]
    for _, r in sub.iterrows():
        x_labels.append(f"{r['Model']}\n{cond.split(') ')[1] if ') ' in cond else cond}")
        acc_vals.append(r["Accuracy"]); f1w_vals.append(r["F1 (weighted)"]); colors.append(colors_map[cond])

x = np.arange(len(x_labels))
axes[0].bar(x, acc_vals, color=colors)
axes[0].set_xticks(x); axes[0].set_xticklabels(x_labels, rotation=75, ha="right", fontsize=7)
axes[0].set_ylabel("Accuracy"); axes[0].set_title("Test Accuracy -- all 20 results", fontweight="bold")
axes[0].set_ylim(0.75, 1.0)

axes[1].bar(x, f1w_vals, color=colors)
axes[1].set_xticks(x); axes[1].set_xticklabels(x_labels, rotation=75, ha="right", fontsize=7)
axes[1].set_ylabel("Weighted F1"); axes[1].set_title("Weighted F1-score -- all 20 results", fontweight="bold")
axes[1].set_ylim(0.75, 1.0)

legend_handles = [Patch(color=colors_map[c], label=c) for c in cond_order]
fig.legend(handles=legend_handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06), frameon=False, fontsize=9)
fig.suptitle("Full Comparison: 20 Results Across 5 Experimental Conditions", fontsize=14, fontweight="bold", y=1.14)
plt.tight_layout()
plt.savefig(OUT_FIGS / "full_20_results_comparison.png", dpi=200, bbox_inches="tight")
print("\nSaved figure:", OUT_FIGS / "full_20_results_comparison.png")
print("DONE")
