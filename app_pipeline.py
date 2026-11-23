"""
Reusable sentiment-classification pipeline shared by the notebook and the
Streamlit app. Generalized so it can be pointed at ANY text dataset, not just
the Kopi Kenangan reviews - the app.py UI lets a user upload their own
CSV/XLSX and reuses these exact functions to train fresh models on it.
"""
import re
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42


def score_to_3class(score, neg_max=2, neu_value=3):
    """Map a 1-5 star rating to Negative / Neutral / Positive. Generalizes to
    any rating scale by adjusting neg_max / neu_value."""
    if score <= neg_max:
        return "Negative"
    elif score == neu_value:
        return "Neutral"
    else:
        return "Positive"


def clean_text(text, use_indonesian_stopwords=True, _cache={}):
    """Lower-case, strip URLs/punctuation/numbers, optionally remove
    Indonesian/Malay stop-words. Works reasonably on any language even with
    stop-word removal off (plain bag-of-words / TF-IDF still applies)."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if use_indonesian_stopwords:
        if "remover" not in _cache:
            from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
            _cache["remover"] = StopWordRemoverFactory().create_stop_word_remover()
        text = _cache["remover"].remove(text)
    return text


def prepare_dataset(df, text_col, use_indonesian_stopwords=True):
    """Clean a raw dataframe's text column in place, dropping empty rows."""
    df = df.copy()
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.strip().str.len() > 1]
    df = df.drop_duplicates(subset=text_col).reset_index(drop=True)
    df["clean_text"] = df[text_col].apply(lambda t: clean_text(t, use_indonesian_stopwords))
    df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)
    return df


MODEL_REGISTRY = {
    "Naive Bayes": lambda: MultinomialNB(),
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
    "SVM": lambda: SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
    "XGBoost": lambda: XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=1),
}

TUNING_GRIDS = {
    "Naive Bayes": {"alpha": [0.1, 0.5, 1.0, 2.0]},
    "Logistic Regression": {"C": [0.1, 1, 10], "solver": ["lbfgs"]},
    "Decision Tree": {"max_depth": [10, 20, None], "min_samples_leaf": [1, 2, 5]},
    "SVM": {"C": [0.1, 1, 10]},
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [20, None]},
    "XGBoost": {"n_estimators": [100, 200], "max_depth": [4, 6]},
}


def train_and_evaluate(
    texts, labels,
    test_size=0.2,
    max_features=3000,
    use_smote=True,
    use_tuning=True,
    models_to_run=None,
    progress_callback=None,
):
    """
    Full train/evaluate pipeline generalized for any (texts, labels) pair.

    Returns a dict with: le, tfidf, results (per-model metrics), fitted_models
    (per-model estimator), best_model_name, X_test/y_test (for confusion
    matrices), and split info for reporting.
    """
    models_to_run = models_to_run or list(MODEL_REGISTRY.keys())

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(set(y))

    if n_classes < 2:
        raise ValueError(
            "Only one sentiment class is present after cleaning — need at least 2 "
            "distinct classes to train a classifier. Check your label/rating column choice."
        )
    class_counts = np.bincount(y)
    if class_counts.min() < 2:
        smallest = le.inverse_transform([int(np.argmin(class_counts))])[0]
        raise ValueError(
            f"The '{smallest}' class only has {class_counts.min()} example(s) after "
            "cleaning. Each class needs at least 2 examples (ideally many more) to "
            "train and evaluate a model — upload a larger sample or merge sparse classes."
        )
    min_test_rows = max(n_classes, 5)
    if len(texts) * test_size < min_test_rows:
        raise ValueError(
            f"This dataset ({len(texts)} usable rows) is too small for a reliable "
            f"train/test split across {n_classes} classes. Aim for at least "
            f"{int(min_test_rows / test_size)} rows (ideally hundreds) before training."
        )

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=RANDOM_STATE,
        stratify=y if n_classes > 1 else None,
    )

    # min_df=2 is fine at Kopi-Kenangan scale (~1,000 rows) but silently
    # strips almost every word for small uploaded datasets, since it requires
    # a term to appear in >=2 documents. Adapt it to dataset size so the
    # pipeline generalizes to small SME datasets too.
    min_df = 2 if len(X_train_text) >= 50 else 1
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=min_df)
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)

    if use_smote and len(set(y_train)) > 1:
        min_class_count = np.bincount(y_train).min()
        if min_class_count >= 2:
            k_neighbors = max(1, min(5, min_class_count - 1))
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
            X_train_fit, y_train_fit = smote.fit_resample(X_train, y_train)
        else:
            X_train_fit, y_train_fit = X_train, y_train
    else:
        X_train_fit, y_train_fit = X_train, y_train

    n_classes = len(set(y_train))
    cv_splits = min(5, int(np.bincount(y_train_fit).min())) if use_tuning else None
    cv_splits = max(cv_splits, 2) if cv_splits else None

    results = {}
    fitted_models = {}

    for i, name in enumerate(models_to_run):
        if progress_callback:
            progress_callback(i, len(models_to_run), name)

        t0 = time.time()
        model = MODEL_REGISTRY[name]()

        if use_tuning and cv_splits and n_classes > 1:
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
            grid = TUNING_GRIDS[name]
            try:
                gs = GridSearchCV(model, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
                gs.fit(X_train_fit, y_train_fit)
                model = gs.best_estimator_
                best_params = gs.best_params_
                cv_score = gs.best_score_
            except ValueError:
                model.fit(X_train_fit, y_train_fit)
                best_params, cv_score = {}, None
        else:
            model.fit(X_train_fit, y_train_fit)
            best_params, cv_score = {}, None

        train_time = time.time() - t0
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "best_params": best_params,
            "cv_f1_macro": cv_score,
            "test_accuracy": acc,
            "test_f1_macro": f1,
            "test_f1_weighted": f1_w,
            "test_precision_macro": prec,
            "test_recall_macro": rec,
            "train_time_sec": train_time,
            "classification_report": classification_report(
                y_test, y_pred, target_names=le.classes_, zero_division=0, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        fitted_models[name] = model

    best_model_name = max(results, key=lambda k: results[k]["test_f1_weighted"])

    return {
        "label_encoder": le,
        "vectorizer": tfidf,
        "results": results,
        "fitted_models": fitted_models,
        "best_model_name": best_model_name,
        "classes": list(le.classes_),
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
    }


# Common informal Indonesian/Malay connector and filler words that survive
# the Sastrawi stop-word remover (which targets formal spelling) but carry no
# thematic meaning on their own - filtered out of the *displayed* top terms
# only, not from the TF-IDF features used for model training, so this has no
# effect on the trained models or Chapter 4 results.
_NOISE_TERMS = {
    "aplikasi", "aplikasinya", "app", "mau", "udah", "sudah", "padahal", "baru",
    "sama", "saya", "saja", "aja", "dgn", "dan", "yg", "yang", "itu", "ini",
    "ada", "ya", "ga", "gak", "nggak", "tdk", "tidak", "akan", "jadi", "bikin",
    "banget", "terlalu", "kalau", "karena", "juga", "buat", "untuk", "utk",
    "dari", "ke", "di", "pas", "kalo", "gitu", "gini", "kok", "nya", "kalian",
    "gue", "gua", "gw", "kita", "kami", "nih", "deh", "dong", "sih", "lah",
    "banyak", "sangat", "up",
}


def _is_noise_term(term):
    return all(w in _NOISE_TERMS for w in term.split())


def top_terms_by_sentiment(
    raw_texts, clean_texts, sentiments, target,
    top_n=10, ngram_range=(1, 2), min_df=2, n_examples=2,
):
    """
    Surface the most distinctive recurring words/phrases for one sentiment
    class within a batch of scored reviews, so a product manager gets
    actionable themes (e.g. "checkout error", "delivery delay") instead of
    just a raw list of individual reviews.

    Distinctiveness = how much more often a term appears in `target`-class
    reviews than in the rest of the batch (by document frequency). This
    filters out words that are common everywhere (e.g. the product name)
    and surfaces terms concentrated in one sentiment class. Generalizes to
    any class label, not just "Negative"/"Positive".

    Returns a DataFrame with columns: term, count, pct_of_class, examples
    (a short list of raw review snippets containing that term).
    """
    raw_texts = pd.Series(raw_texts).reset_index(drop=True)
    clean_texts = pd.Series(clean_texts).reset_index(drop=True)
    sentiments = pd.Series(sentiments).reset_index(drop=True)

    empty = pd.DataFrame(columns=["term", "count", "pct_of_class", "examples"])

    mask_target = (sentiments == target).values
    n_target = int(mask_target.sum())
    if n_target < 3:
        return empty

    try:
        vec = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
        X = vec.fit_transform(clean_texts)
    except ValueError:
        # Vocabulary ended up empty (e.g. min_df too strict for this batch size).
        return empty

    if X.shape[1] == 0:
        return empty

    vocab = vec.get_feature_names_out()
    X_bin = (X > 0).astype(int)

    target_doc_count = np.asarray(X_bin[mask_target].sum(axis=0)).ravel()
    other_mask = ~mask_target
    n_other = int(other_mask.sum())
    if n_other > 0:
        other_doc_count = np.asarray(X_bin[other_mask].sum(axis=0)).ravel()
    else:
        other_doc_count = np.zeros_like(target_doc_count)

    target_rate = target_doc_count / max(n_target, 1)
    other_rate = other_doc_count / max(n_other, 1)
    distinctiveness = target_rate - other_rate

    eligible = np.where(target_doc_count >= min_df)[0]
    eligible = eligible[np.argsort(-(distinctiveness[eligible] * 1000 + target_doc_count[eligible]))]

    rows = []
    for idx in eligible:
        if len(rows) >= top_n:
            break
        term = vocab[idx]
        if _is_noise_term(term):
            continue
        col = np.asarray(X[:, idx].todense()).ravel()
        example_positions = np.where((col > 0) & mask_target)[0][:n_examples]
        examples = [raw_texts.iloc[i] for i in example_positions]
        rows.append({
            "term": term,
            "count": int(target_doc_count[idx]),
            "pct_of_class": round(100 * target_rate[idx], 1),
            "examples": examples,
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Fixed-category classification (Service / Product Quality / Price).
#
# Deliberately keyword-rule-based rather than an LLM call per review: for a
# batch of a few hundred reviews, an LLM-per-review pass would mean a few
# hundred extra Hugging Face Inference API calls, which is slow and easy to
# rate-limit on the free tier during a live demo. Keyword matching is local,
# free, and instant regardless of batch size -- the tradeoff is that it will
# miss reviews that raise a category without using any of the listed words.
# Keyword lists are bilingual (English + Malay/Indonesian) to match the
# code-switched review text this app is built for (see Chapter 1, Section
# 1.5.3 / Chapter 2, Section 2.7.2).
#
# Matched against `clean_text` output (lower-cased, punctuation/number-
# stripped, optionally Malay-stopword-removed) -- so keywords are plain,
# unaccented lowercase words, and Malay stopword removal only strips
# function words (e.g. "yang", "dan"), never the content words listed here.
# ----------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "Service": [
        # English
        "service", "staff", "slow", "wait", "waiting", "queue", "rude",
        "friendly", "unfriendly", "attitude", "crowded", "line", "delay",
        "delayed",
        # Malay / Indonesian
        "lambat", "lama", "layanan", "pelayan", "ramah", "tunggu", "antre",
        "antrian", "sabar", "kasar", "lewat",
    ],
    "Product Quality": [
        # English
        "taste", "flavor", "flavour", "quality", "fresh", "stale", "cold",
        "warm", "spill", "spilled", "spilling", "leak", "leaking", "wrong",
        "burnt", "sweet", "bland", "watery", "delicious", "yummy", "melted",
        # Malay / Indonesian
        "rasa", "sedap", "enak", "segar", "basi", "kualiti", "kualitas",
        "tumpah", "bocor", "salah", "hambar", "manis", "pahit",
    ],
    "Price": [
        # English
        "price", "prices", "pricing", "expensive", "cheap", "value",
        "worth", "cost", "overpriced", "affordable", "pricey",
        # Malay / Indonesian
        "mahal", "murah", "harga", "berbaloi", "sepadan",
    ],
}


def categorize_reviews(clean_texts, keywords=None):
    """
    Bilingual keyword-rule categorization of already-cleaned review text into
    fixed SME-relevant business categories (Service, Product Quality, Price).

    Multi-label: a single review can match more than one category (e.g. a
    review can complain about both slow service AND high price), or none
    (falls into "Other").

    Returns a boolean DataFrame indexed like `clean_texts`, one column per
    category in `keywords` plus an "Other" column that is True only where no
    category matched.
    """
    keywords = keywords or CATEGORY_KEYWORDS
    clean_texts = pd.Series(clean_texts).reset_index(drop=True).fillna("")

    result = pd.DataFrame(index=clean_texts.index)
    for cat, words in keywords.items():
        pattern = r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"
        result[cat] = clean_texts.str.contains(pattern, regex=True, na=False)

    result["Other"] = ~result[list(keywords.keys())].any(axis=1)
    return result


def category_counts(clean_texts, sentiments=None, target=None, keywords=None):
    """
    Review counts per category (excluding "Other"), sorted descending.
    If `sentiments` and `target` are given, restricts to that sentiment
    class first (e.g. category counts within just the Negative reviews) --
    useful for surfacing which category is driving complaints specifically,
    rather than mentions across all sentiment.
    """
    keywords = keywords or CATEGORY_KEYWORDS
    cats = categorize_reviews(clean_texts, keywords)
    if sentiments is not None and target is not None:
        mask = (pd.Series(sentiments).reset_index(drop=True) == target).values
        cats = cats.loc[mask]
    return cats[list(keywords.keys())].sum().sort_values(ascending=False)


# Short, model-type-specific notes used to flesh out the generated
# suggestion text with a plain-language sense of *why* a model behaves the
# way it does, not just its numbers.
MODEL_NOTES = {
    "Naive Bayes": "a fast, simple probabilistic baseline",
    "Logistic Regression": "a linear model whose per-word coefficients are directly interpretable",
    "Decision Tree": "a single interpretable tree that is easy to visualize and explain to stakeholders",
    "SVM": "a strong margin-based linear classifier, though less directly interpretable than a plain linear model",
    "Random Forest": "an ensemble of decision trees - robust, but harder to interpret and slower to retrain",
    "XGBoost": "a gradient-boosted ensemble - often powerful, but typically the slowest to tune and least interpretable of the six",
}


def generate_model_suggestion(results):
    """
    Turn a `results` dict (in the same shape train_and_evaluate() returns
    under results["results"]: per-model test_accuracy / test_f1_macro /
    test_f1_weighted / train_time_sec) into a plain-language recommendation
    of which model to deploy.

    Mirrors the reasoning used to select the best model in Chapter 4 -
    weighted F1 as the primary criterion (reflects reliability under the
    natural, imbalanced class distribution a deployed system will actually
    see), with macro F1 and training time considered as secondary factors -
    but is fully generic, so it works for any dataset's results, not just
    the Kopi Kenangan case.
    """
    if not results:
        return "No trained models to compare yet."

    by_weighted = sorted(results.items(), key=lambda kv: kv[1]["test_f1_weighted"], reverse=True)
    by_macro = sorted(results.items(), key=lambda kv: kv[1]["test_f1_macro"], reverse=True)
    by_speed = sorted(results.items(), key=lambda kv: kv[1]["train_time_sec"])

    best_name, best = by_weighted[0]
    best_macro_name, best_macro_r = by_macro[0]
    fastest_name, fastest_r = by_speed[0]

    parts = []

    parts.append(
        f"**{best_name}** is the recommended model here, with the highest weighted F1-score "
        f"({best['test_f1_weighted']:.3f}) and {best['test_accuracy'] * 100:.1f}% accuracy on the held-out "
        f"test set. Weighted F1 is used as the primary criterion because it reflects overall reliability "
        f"under the natural, often imbalanced class distribution a deployed system will actually see."
    )

    if best_macro_name == best_name:
        parts.append(
            f"It also has the best macro F1-score ({best['test_f1_macro']:.3f}) among the "
            f"{len(results)} models compared, meaning it is the most consistent across all classes, "
            f"including any minority ones - not just the majority class."
        )
    else:
        parts.append(
            f"Note that **{best_macro_name}** scores higher on macro F1 "
            f"({best_macro_r['test_f1_macro']:.3f} vs {best['test_f1_macro']:.3f} here), meaning it may "
            f"handle minority-class reviews a little better - worth a look if correctly catching rare "
            f"classes matters more than overall accuracy for this particular use case."
        )

    if fastest_name == best_name:
        parts.append(
            f"It is also the fastest of the {len(results)} models to tune "
            f"({best['train_time_sec']:.1f}s), which keeps periodic retraining on new reviews cheap."
        )
    else:
        parts.append(
            f"Tuning it took {best['train_time_sec']:.1f}s, versus {fastest_r['train_time_sec']:.1f}s for "
            f"the fastest model ({fastest_name}) - still practical for periodic retraining at typical SME "
            f"data volumes."
        )

    note = MODEL_NOTES.get(best_name)
    if note:
        parts.append(f"As a model type, {best_name} is {note}.")

    return " ".join(parts)


def predict_sentiment(text, model, vectorizer, label_encoder, use_indonesian_stopwords=True):
    """Predict sentiment (+ per-class probability if available) for one raw text."""
    cleaned = clean_text(text, use_indonesian_stopwords)
    X = vectorizer.transform([cleaned])
    pred_idx = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba(X)[0]
        proba = dict(zip(label_encoder.classes_, proba_arr))
    return pred_label, proba
