"""Loads the deployed model once per function instance and scores text."""
from functools import lru_cache
from pathlib import Path

import joblib

from .pipeline import clean_text

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

RECOMMENDATION = {
    "Negative": "Flag for immediate follow-up — reach out to the customer and check for a "
                "recurring operational issue.",
    "Neutral": "Monitor — track whether this sentiment trend grows across similar reviews.",
    "Positive": "No action needed — consider highlighting in testimonials/marketing.",
}


@lru_cache(maxsize=1)
def load():
    """Returns (model, vectorizer, label_encoder, display_name)."""
    model = joblib.load(MODEL_DIR / "best_model.pkl")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    name = (MODEL_DIR / "best_model_name.txt").read_text().strip()
    return model, vectorizer, label_encoder, name


def classes() -> list[str]:
    return [str(c) for c in load()[2].classes_]


def predict_many(texts, use_indonesian_stopwords: bool = True):
    """Score a list of raw texts. Returns (cleaned, labels, probas), where each
    proba is a {class: probability} dict. Same result as calling the Streamlit
    app's predict_sentiment() on each text, but vectorised in one pass."""
    model, vectorizer, label_encoder, _ = load()
    cleaned = [clean_text(t, use_indonesian_stopwords) for t in texts]
    if not cleaned:
        return [], [], []
    X = vectorizer.transform(cleaned)
    idx = model.predict(X)
    labels = [str(lbl) for lbl in label_encoder.inverse_transform(idx)]
    proba_rows = model.predict_proba(X)
    names = [str(c) for c in label_encoder.classes_]
    probas = [{n: float(p) for n, p in zip(names, row)} for row in proba_rows]
    return cleaned, labels, probas
