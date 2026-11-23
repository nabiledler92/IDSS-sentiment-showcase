"""API and parity tests. Run from web/backend:  python -m pytest -q"""
import csv
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parents[1]
SAMPLE = REPO / "data" / "sample_reviews_synthetic.csv"
sys.path.insert(0, str(BACKEND))

import main  # noqa: E402
from idss import ai, guard, model, pipeline  # noqa: E402

client = TestClient(main.app)


def sample_texts():
    with open(SAMPLE, newline="", encoding="utf-8") as f:
        return [r["review_text"] for r in csv.DictReader(f)]


# ---------------------------------------------------------------- parity ----
# The web API must give the same answers as the Streamlit app's pipeline.

@pytest.fixture(scope="module")
def streamlit_pipeline():
    sys.path.insert(0, str(REPO))
    import app_pipeline

    return app_pipeline


EXTRA = [
    "Order sampai lambat sejam, air dah cair. Sangat kecewa dengan servis",
    "Aplikasi nya bagus, tapi sering error saat checkout...",
    "https://example.com 123 !!! Kopi SEDAP gila",
    "biasa je",
]


@pytest.mark.parametrize("stopwords", [True, False])
def test_predictions_match_streamlit(streamlit_pipeline, stopwords):
    m, vec, le, _ = model.load()
    texts = sample_texts() + EXTRA
    cleaned, labels, probas = model.predict_many(texts, stopwords)
    for text, c, lbl, p in zip(texts, cleaned, labels, probas):
        assert c == streamlit_pipeline.clean_text(text, stopwords)
        exp_lbl, exp_p = streamlit_pipeline.predict_sentiment(text, m, vec, le, stopwords)
        assert lbl == exp_lbl
        for k in exp_p:
            assert p[k] == pytest.approx(float(exp_p[k]), abs=1e-12)


def test_themes_and_categories_match_streamlit(streamlit_pipeline):
    texts = sample_texts() * 2  # enough rows for min_df=2 themes in every class
    cleaned, labels, _ = model.predict_many(texts, True)
    for target in ("Negative", "Positive"):
        ours = pipeline.top_terms_by_sentiment(texts, cleaned, labels, target)
        theirs = streamlit_pipeline.top_terms_by_sentiment(texts, cleaned, labels, target)
        assert [(r["term"], r["count"], r["pct_of_class"], r["examples"]) for r in ours] == [
            (r.term, int(r.count), float(r.pct_of_class), list(r.examples)) for r in theirs.itertuples()
        ]
    assert pipeline.category_counts(cleaned) == {
        k: int(v) for k, v in streamlit_pipeline.category_counts(cleaned).items()
    }
    assert pipeline.category_counts(cleaned, labels, "Negative") == {
        k: int(v) for k, v in streamlit_pipeline.category_counts(cleaned, labels, "Negative").items()
    }


# ----------------------------------------------------------------- routes ---

def test_meta():
    r = client.get("/api/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["modelName"] == "Logistic Regression with SMOTE"
    assert body["classes"] == ["Negative", "Neutral", "Positive"]


def test_predict_negative_review():
    r = client.post("/api/predict", json={"text": EXTRA[0]})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "Negative"
    assert set(body["probabilities"]) == {"Negative", "Neutral", "Positive"}
    assert body["recommendation"].startswith("Flag for immediate follow-up")


def test_predict_rejects_blank_and_huge_text():
    assert client.post("/api/predict", json={"text": "   "}).status_code == 400
    assert client.post("/api/predict", json={"text": "x" * 2001}).status_code == 422


def upload(name="reviews.csv", data=None):
    return {"file": (name, data if data is not None else SAMPLE.read_bytes(), "text/csv")}


def test_columns_guesses_review_text():
    r = client.post("/api/columns", files=upload())
    assert r.status_code == 200
    assert r.json() == {"columns": ["review_text", "rating"], "guess": "review_text", "rows": 40, "truncated": False}


def test_columns_rejects_other_file_types():
    r = client.post("/api/columns", files=upload("notes.txt", b"hello"))
    assert r.status_code == 400
    assert "csv" in r.json()["detail"]


def test_batch_summary():
    r = client.post("/api/batch", files=upload(), data={"text_col": "review_text", "stopwords": "true"})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 40
    assert sum(body["counts"].values()) == 40
    assert body["negativeCount"] == body["counts"]["Negative"]
    assert set(body["categoriesAll"]) == {"Service", "Product Quality", "Price"}
    assert guard.verify(body["stats"], body["signature"])


def test_batch_unknown_column():
    r = client.post("/api/batch", files=upload(), data={"text_col": "nope"})
    assert r.status_code == 400


def test_batch_xlsx(tmp_path):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["id", "ulasan"])
    for i, t in enumerate(sample_texts()[:10]):
        ws.append([i, t])
    path = tmp_path / "r.xlsx"
    wb.save(path)
    files = {"file": ("r.xlsx", path.read_bytes(),
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    cols = client.post("/api/columns", files=files).json()
    assert cols["guess"] == "ulasan"
    r = client.post("/api/batch", files=files, data={"text_col": "ulasan"})
    assert r.status_code == 200 and r.json()["total"] == 10


def test_batch_csv_download():
    r = client.post("/api/batch/csv", files=upload(), data={"text_col": "review_text"})
    assert r.status_code == 200
    rows = list(csv.reader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert rows[0] == ["review_text", "rating", "predicted_sentiment", "confidence"]
    assert len(rows) == 41


# --------------------------------------------------------------------- AI ---

def signed_stats():
    body = client.post("/api/batch", files=upload(), data={"text_col": "review_text"}).json()
    return body["stats"], body["signature"]


def test_ai_off_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    stats, sig = signed_stats()
    r = client.post("/api/brief", json={"stats": stats, "signature": sig})
    assert r.status_code == 503


@pytest.fixture
def fake_hf(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_not_a_real_token")
    calls = []

    def fake_chat(system_prompt, user_prompt, max_tokens):
        calls.append((system_prompt, user_prompt))
        return "**Short-term: test**\n- do a thing"

    monkeypatch.setattr(ai, "_chat", fake_chat)
    guard._hits.clear()
    yield calls
    guard._hits.clear()


def test_brief_and_ask_with_valid_signature(fake_hf):
    stats, sig = signed_stats()
    r = client.post("/api/brief", json={"stats": stats, "signature": sig})
    assert r.status_code == 200 and r.json()["text"].startswith("**Short-term")
    assert "Batch of 40 customer reviews" in fake_hf[0][1]

    r = client.post("/api/ask", json={"stats": stats, "signature": sig, "question": "What should I fix first?"})
    assert r.status_code == 200
    assert "DATA:\nTotal reviews analyzed: 40." in fake_hf[1][0]


def test_tampered_stats_are_rejected(fake_hf):
    stats, sig = signed_stats()
    stats["total"] = 999
    r = client.post("/api/brief", json={"stats": stats, "signature": sig})
    assert r.status_code == 400
    assert not fake_hf


def test_rate_limit(fake_hf):
    stats, sig = signed_stats()
    for _ in range(guard.AI_REQUESTS_PER_WINDOW):
        assert client.post("/api/brief", json={"stats": stats, "signature": sig}).status_code == 200
    r = client.post("/api/brief", json={"stats": stats, "signature": sig})
    assert r.status_code == 429


def test_hf_errors_are_logged_and_explained(monkeypatch, caplog):
    """A Hugging Face failure is logged with its status (never the token) and
    turned into a clear message for the visitor."""
    import huggingface_hub

    class FakeResponse:
        status_code = 403

    class FakeError(Exception):
        response = FakeResponse()

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def chat_completion(self, **k):
            raise FakeError("403 Forbidden: insufficient permissions")

    monkeypatch.setenv("HF_TOKEN", "hf_test_not_a_real_token")
    monkeypatch.setattr(huggingface_hub, "InferenceClient", FakeClient)
    guard._hits.clear()
    stats, sig = signed_stats()
    with caplog.at_level("WARNING", logger="idss.ai"):
        r = client.post("/api/brief", json={"stats": stats, "signature": sig})
    assert r.status_code == 502
    assert "isn't set up correctly" in r.json()["detail"]
    assert "status=403" in caplog.text and "hf_test_not_a_real_token" not in caplog.text
    guard._hits.clear()
