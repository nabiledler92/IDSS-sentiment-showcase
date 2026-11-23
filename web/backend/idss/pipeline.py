"""Text cleaning, theme extraction and category tagging.

These mirror the functions in the repository's app_pipeline.py (used by the
Streamlit app) so the web app gives the same answers. They're copied rather
than imported because the Vercel project is rooted at web/, and they avoid
pandas to keep the serverless function small. tests/test_parity.py checks
them against app_pipeline.py.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

_remover = None


def clean_text(text: str, use_indonesian_stopwords: bool = True) -> str:
    """Lower-case, strip URLs/punctuation/numbers, optionally remove
    Indonesian/Malay stop-words (Sastrawi)."""
    global _remover
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if use_indonesian_stopwords:
        if _remover is None:
            from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

            _remover = StopWordRemoverFactory().create_stop_word_remover()
        text = _remover.remove(text)
    return text


# Common filler words with no thematic meaning on their own. Filtered out of the
# displayed top terms only, exactly as in app_pipeline.py.
_NOISE_TERMS = {
    "aplikasi", "aplikasinya", "app", "mau", "udah", "sudah", "padahal", "baru",
    "sama", "saya", "saja", "aja", "dgn", "dan", "yg", "yang", "itu", "ini",
    "ada", "ya", "ga", "gak", "nggak", "tdk", "tidak", "akan", "jadi", "bikin",
    "banget", "terlalu", "kalau", "karena", "juga", "buat", "untuk", "utk",
    "dari", "ke", "di", "pas", "kalo", "gitu", "gini", "kok", "nya", "kalian",
    "gue", "gua", "gw", "kita", "kami", "nih", "deh", "dong", "sih", "lah",
    "banyak", "sangat", "up",
}


def _is_noise_term(term: str) -> bool:
    return all(w in _NOISE_TERMS for w in term.split())


def top_terms_by_sentiment(raw_texts, clean_texts, sentiments, target,
                           top_n=10, ngram_range=(1, 2), min_df=2, n_examples=2):
    """Words and phrases that appear disproportionately often in `target`-class
    reviews compared with the rest of the batch. Returns a list of dicts with
    term, count, pct_of_class and examples (raw review snippets)."""
    mask_target = np.array([s == target for s in sentiments])
    n_target = int(mask_target.sum())
    if n_target < 3:
        return []
    try:
        vec = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
        X = vec.fit_transform(list(clean_texts))
    except ValueError:
        return []
    if X.shape[1] == 0:
        return []

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
        positions = np.where((col > 0) & mask_target)[0][:n_examples]
        rows.append({
            "term": str(term),
            "count": int(target_doc_count[idx]),
            "pct_of_class": round(100 * float(target_rate[idx]), 1),
            "examples": [str(raw_texts[i]) for i in positions],
        })
    return rows


CATEGORY_KEYWORDS = {
    "Service": [
        "service", "staff", "slow", "wait", "waiting", "queue", "rude",
        "friendly", "unfriendly", "attitude", "crowded", "line", "delay",
        "delayed",
        "lambat", "lama", "layanan", "pelayan", "ramah", "tunggu", "antre",
        "antrian", "sabar", "kasar", "lewat",
    ],
    "Product Quality": [
        "taste", "flavor", "flavour", "quality", "fresh", "stale", "cold",
        "warm", "spill", "spilled", "spilling", "leak", "leaking", "wrong",
        "burnt", "sweet", "bland", "watery", "delicious", "yummy", "melted",
        "rasa", "sedap", "enak", "segar", "basi", "kualiti", "kualitas",
        "tumpah", "bocor", "salah", "hambar", "manis", "pahit",
    ],
    "Price": [
        "price", "prices", "pricing", "expensive", "cheap", "value",
        "worth", "cost", "overpriced", "affordable", "pricey",
        "mahal", "murah", "harga", "berbaloi", "sepadan",
    ],
}

_CATEGORY_PATTERNS = {
    cat: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b")
    for cat, words in CATEGORY_KEYWORDS.items()
}


def category_counts(clean_texts, sentiments=None, target=None) -> dict:
    """Review counts per business category (a review can match several),
    optionally restricted to one sentiment class. Sorted by count, descending,
    with ties kept in CATEGORY_KEYWORDS order."""
    counts = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for i, text in enumerate(clean_texts):
        if target is not None and sentiments[i] != target:
            continue
        for cat, pattern in _CATEGORY_PATTERNS.items():
            if pattern.search(text or ""):
                counts[cat] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
