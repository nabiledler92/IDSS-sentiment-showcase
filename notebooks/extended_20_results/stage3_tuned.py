import warnings
warnings.filterwarnings("ignore")
import sys, time, json
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
X_test, y_test = d["X_test"], d["y_test"]
X_train_res, y_train_res = d["X_train_res"], d["y_train_res"]

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

configs = {
    "Naive Bayes": (MultinomialNB(), {"alpha": [0.1, 0.5, 1.0, 2.0]}, cv5),
    "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                             {"C": [0.1, 1, 10], "solver": ["lbfgs"]}, cv5),
    "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                       {"max_depth": [10, 20, None], "min_samples_leaf": [1, 2, 5]}, cv5),
    "SVM": (SVC(kernel="linear", probability=False, random_state=RANDOM_STATE), {"C": [0.1, 1, 10]}, cv3),
    "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
                       {"n_estimators": [100, 200], "max_depth": [20, None]}, cv3),
    "XGBoost": (XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=1),
                {"n_estimators": [100, 200], "max_depth": [4, 6]}, cv3),
}

which = sys.argv[1] if len(sys.argv) > 1 else None
names = [which] if which else list(configs.keys())

try:
    state = joblib.load(CKPT_DIR / "stage3_tuned_partial.pkl")
except FileNotFoundError:
    state = {"rows": [], "preds": {}, "models": {}, "params": {}, "done": []}

for name in names:
    if name in state["done"]:
        print(f"SKIP (already done): {name}")
        continue
    model, grid, cv = configs[name]
    t0 = time.time()
    gs = GridSearchCV(model, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_train_res, y_train_res)
    train_time = time.time() - t0
    best = gs.best_estimator_
    if name == "SVM":
        best = SVC(kernel="linear", C=gs.best_params_["C"], probability=True, random_state=RANDOM_STATE)
        best.fit(X_train_res, y_train_res)

    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    row = {"Condition": "(c) SMOTE + GridSearchCV", "Model": name, "Accuracy": round(acc, 4),
           "Precision (macro)": round(prec_m, 4), "Recall (macro)": round(rec_m, 4),
           "F1 (macro)": round(f1_m, 4), "F1 (weighted)": round(f1_w, 4),
           "Train Time (s)": round(train_time, 4), "Best Params": str(gs.best_params_)}

    state["rows"].append(row)
    state["preds"][f"(c) {name}"] = y_pred
    state["models"][name] = best
    state["params"][name] = gs.best_params_
    state["done"].append(name)
    joblib.dump(state, CKPT_DIR / "stage3_tuned_partial.pkl")
    print(f"{name:22s} acc={acc:.4f}  f1_macro={f1_m:.4f}  f1_weighted={f1_w:.4f}  time={train_time:5.1f}s  params={gs.best_params_}")

print("done so far:", state["done"])
