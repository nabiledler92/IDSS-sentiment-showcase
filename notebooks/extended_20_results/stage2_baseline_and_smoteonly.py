import warnings
warnings.filterwarnings("ignore")
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "checkpoints"

d = joblib.load(CKPT_DIR / "prep.pkl")
X_train, y_train = d["X_train"], d["y_train"]
X_test, y_test = d["X_test"], d["y_test"]
X_train_res, y_train_res = d["X_train_res"], d["y_train_res"]

def make_base_models():
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=1),
    }

def eval_model(name, model, condition, X_te, y_te, extra=None):
    t0 = time.time()
    y_pred = model.predict(X_te)
    pred_time = time.time() - t0
    acc = accuracy_score(y_te, y_pred)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_te, y_pred, average="weighted", zero_division=0)
    row = {"Condition": condition, "Model": name, "Accuracy": round(acc, 4),
           "Precision (macro)": round(prec_m, 4), "Recall (macro)": round(rec_m, 4),
           "F1 (macro)": round(f1_m, 4), "F1 (weighted)": round(f1_w, 4),
           "Prediction Time (s)": round(pred_time, 4)}
    if extra: row.update(extra)
    return row, y_pred

rows = []
preds = {}

print("=" * 90); print("(a) BASELINE -- 6 classifiers, default hyperparameters, no SMOTE"); print("=" * 90)
t_all = time.time()
for name, model in make_base_models().items():
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    row, y_pred = eval_model(name, model, "(a) Baseline", X_test, y_test, extra={"Train Time (s)": round(fit_time, 4)})
    rows.append(row); preds[f"(a) {name}"] = y_pred
    print(f"{name:22s} acc={row['Accuracy']:.4f}  f1_macro={row['F1 (macro)']:.4f}  f1_weighted={row['F1 (weighted)']:.4f}  fit={fit_time:.2f}s")
print(f"Stage (a) total: {time.time()-t_all:.1f}s")

print("\n" + "=" * 90); print("(b) SMOTE ONLY -- 6 classifiers, default hyperparameters, SMOTE-balanced training data"); print("=" * 90)
t_all = time.time()
for name, model in make_base_models().items():
    t0 = time.time()
    model.fit(X_train_res, y_train_res)
    fit_time = time.time() - t0
    row, y_pred = eval_model(name, model, "(b) SMOTE only", X_test, y_test, extra={"Train Time (s)": round(fit_time, 4)})
    rows.append(row); preds[f"(b) {name}"] = y_pred
    print(f"{name:22s} acc={row['Accuracy']:.4f}  f1_macro={row['F1 (macro)']:.4f}  f1_weighted={row['F1 (weighted)']:.4f}  fit={fit_time:.2f}s")
print(f"Stage (b) total: {time.time()-t_all:.1f}s")

joblib.dump({"rows": rows, "preds": preds}, CKPT_DIR / "stage2_results.pkl")
print("SAVED checkpoints/stage2_results.pkl")
