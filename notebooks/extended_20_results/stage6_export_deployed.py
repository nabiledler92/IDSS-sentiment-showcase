"""Stage 6: export the deployed model artifacts the Streamlit app loads.

Refits the six condition (b) models (SMOTE-balanced training data, default
hyperparameters) on the checkpointed split from stage 1, then writes:

- models/model_<Name>.pkl            all six condition (b) classifiers
- models/best_model.pkl              the best of them by weighted F1
- models/best_model_name.txt         e.g. "Logistic Regression with SMOTE"
- models/tfidf_vectorizer.pkl        the stage 1 TF-IDF vectorizer
- models/label_encoder.pkl           the stage 1 label encoder
- outputs/tables/model_results_all.json   per-model metrics (read by the app's Model Info tab)
- outputs/figures/confusion_matrix_smote_only_lr.png  confusion matrix of the deployed model

Run after stage1_prep.py (stages 2-5 are not required, but run them first if
you want the 20-result tables to match the exported models).
"""
import warnings
warnings.filterwarnings("ignore")
import json
import time
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42


def make_base_models():
    """Same definitions as stage2_baseline_and_smoteonly.py (condition b)."""
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=1),
    }


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
CKPT_DIR = BASE_DIR / "checkpoints"
MODELS_DIR = PROJECT_DIR / "models"
OUT_TABLES = PROJECT_DIR / "outputs" / "tables"
OUT_FIGS = PROJECT_DIR / "outputs" / "figures"

d = joblib.load(CKPT_DIR / "prep.pkl")
X_train_res, y_train_res = d["X_train_res"], d["y_train_res"]
X_test, y_test = d["X_test"], d["y_test"]
tfidf, le = d["tfidf"], d["le"]
classes = list(le.classes_)

results, fitted = {}, {}
for name, model in make_base_models().items():
    t0 = time.time()
    model.fit(X_train_res, y_train_res)
    train_time = time.time() - t0
    y_pred = model.predict(X_test)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    results[name] = {
        "best_params": {}, "cv_best_f1_macro": None,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision_macro": p_m, "test_recall_macro": r_m, "test_f1_macro": f_m,
        "test_precision_weighted": p_w, "test_recall_weighted": r_w, "test_f1_weighted": f_w,
        "train_time_sec": train_time,
        "classification_report": classification_report(
            y_test, y_pred, target_names=classes, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    fitted[name] = model
    print(f"{name:22s} acc={results[name]['test_accuracy']:.4f}  "
          f"f1_macro={f_m:.4f}  f1_weighted={f_w:.4f}")

best_name = max(results, key=lambda n: results[n]["test_f1_weighted"])
display_name = f"{best_name} with SMOTE"
print(f"\nBest by weighted F1: {display_name}")

MODELS_DIR.mkdir(exist_ok=True)
for name, model in fitted.items():
    joblib.dump(model, MODELS_DIR / f"model_{name.replace(' ', '_')}.pkl")
joblib.dump(fitted[best_name], MODELS_DIR / "best_model.pkl")
joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.pkl")
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
(MODELS_DIR / "best_model_name.txt").write_text(display_name)

with open(OUT_TABLES / "model_results_all.json", "w") as f:
    json.dump(results, f, indent=2)

cm = np.array(results[best_name]["confusion_matrix"])
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
ax.set_title(f"Confusion Matrix — {best_name}\n(SMOTE only, test set, n = {len(y_test)})",
             fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_FIGS / "confusion_matrix_smote_only_lr.png", dpi=200, bbox_inches="tight")
print("Saved models/, outputs/tables/model_results_all.json and the confusion matrix.")
