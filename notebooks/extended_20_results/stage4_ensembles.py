import warnings
warnings.filterwarnings("ignore")
import time
from pathlib import Path
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "checkpoints"

d = joblib.load(CKPT_DIR / "prep.pkl")
X_test, y_test = d["X_test"], d["y_test"]
X_train_res, y_train_res = d["X_train_res"], d["y_train_res"]

s3 = joblib.load(CKPT_DIR / "stage3_tuned_partial.pkl")
rf_tuned = s3["models"]["Random Forest"]
xgb_tuned = s3["models"]["XGBoost"]

cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

def eval_and_report(name, model, condition, fit_time):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    row = {"Condition": condition, "Model": name, "Accuracy": round(acc, 4),
           "Precision (macro)": round(prec_m, 4), "Recall (macro)": round(rec_m, 4),
           "F1 (macro)": round(f1_m, 4), "F1 (weighted)": round(f1_w, 4),
           "Train Time (s)": round(fit_time, 4)}
    print(f"{name:40s} acc={acc:.4f}  f1_macro={f1_m:.4f}  f1_weighted={f1_w:.4f}  fit={fit_time:.2f}s")
    return row, y_pred

print("=" * 90); print("(d) ENSEMBLE (bagging-style) -- RF + XGBoost soft-voting"); print("=" * 90)
t0 = time.time()
voting = VotingClassifier(estimators=[("random_forest", rf_tuned), ("xgboost", xgb_tuned)], voting="soft")
voting.fit(X_train_res, y_train_res)
voting_time = time.time() - t0
row_d, pred_d = eval_and_report("RF + XGBoost (Voting / Bagging-style)", voting, "(d) Ensemble - Bagging-style", voting_time)

print("\n" + "=" * 90); print("(e) ENSEMBLE (boosting-style) -- RF + XGBoost stacking"); print("=" * 90)
t0 = time.time()
stacking = StackingClassifier(
    estimators=[("random_forest", rf_tuned), ("xgboost", xgb_tuned)],
    final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    cv=cv3, n_jobs=-1,
)
stacking.fit(X_train_res, y_train_res)
stacking_time = time.time() - t0
row_e, pred_e = eval_and_report("RF + XGBoost (Stacking / Boosting-style)", stacking, "(e) Ensemble - Boosting-style", stacking_time)

joblib.dump({
    "rows": [row_d, row_e],
    "preds": {"(d) RF+XGB Voting": pred_d, "(e) RF+XGB Stacking": pred_e},
    "voting_model": voting, "stacking_model": stacking,
}, CKPT_DIR / "stage4_ensembles.pkl")
print("\nSAVED checkpoints/stage4_ensembles.pkl")
