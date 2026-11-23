"""IDSS web API (FastAPI on Vercel Functions).

Routes, all under /api (web/vercel.json sends /api/* here):
  GET  /api/meta          model name, classes, whether AI features are on
  POST /api/predict       classify one review
  POST /api/columns       read an uploaded file's columns and guess the text column
  POST /api/batch         classify every row of an uploaded file and summarise it
  POST /api/batch/csv     the same, returned as a CSV of per-row results
  POST /api/brief         AI insights brief for a batch (Hugging Face)
  POST /api/ask           grounded follow-up question about a batch (Hugging Face)
"""
import csv
import io
from collections import Counter
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from idss import ai, guard, model
from idss.files import MAX_ROWS, FileError, guess_text_column, read_table
from idss.pipeline import category_counts, top_terms_by_sentiment

app = FastAPI(title="IDSS sentiment API", docs_url=None, redoc_url=None, openapi_url=None)

LABELS = ("Negative", "Neutral", "Positive")
MAX_REVIEW_CHARS = 2000
MAX_QUESTION_CHARS = 300
NEGATIVE_LIST_LIMIT = 100


@app.exception_handler(FileError)
async def _file_error(_: Request, exc: FileError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(ai.AIError)
async def _ai_error(_: Request, exc: ai.AIError):
    return JSONResponse({"detail": str(exc)}, status_code=502)


@app.get("/api/meta")
def meta():
    _, _, _, name = model.load()
    return {
        "modelName": name,
        "classes": model.classes(),
        "aiEnabled": ai.enabled(),
        "aiModel": ai.HF_MODEL_ID,
        "limits": {"maxRows": MAX_ROWS, "maxFileMB": 4},
    }


class PredictIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_REVIEW_CHARS)
    stopwords: bool = True


@app.post("/api/predict")
def predict(body: PredictIn):
    if not body.text.strip():
        raise HTTPException(400, "Enter a review to analyse.")
    _, labels, probas = model.predict_many([body.text], body.stopwords)
    label = labels[0]
    return {
        "label": label,
        "probabilities": probas[0],
        "confidence": max(probas[0].values()),
        "recommendation": model.RECOMMENDATION.get(label, ""),
    }


async def _read_upload(file: UploadFile):
    data = await file.read()
    return read_table(file.filename or "", data)


@app.post("/api/columns")
async def columns(file: Annotated[UploadFile, File()]):
    headers, rows, truncated = await _read_upload(file)
    return {
        "columns": headers,
        "guess": guess_text_column(headers, rows),
        "rows": len(rows),
        "truncated": truncated,
    }


async def _score_upload(file: UploadFile, text_col: str, stopwords: bool):
    headers, rows, truncated = await _read_upload(file)
    if text_col not in headers:
        raise FileError(f"Column \"{text_col}\" isn't in this file.")
    i = headers.index(text_col)
    kept = [r for r in rows if r[i].strip()]
    if not kept:
        raise FileError(f"Column \"{text_col}\" has no text in it.")
    texts = [r[i] for r in kept]
    cleaned, labels, probas = model.predict_many(texts, stopwords)
    return headers, kept, texts, cleaned, labels, probas, truncated


@app.post("/api/batch")
async def batch(
    file: Annotated[UploadFile, File()],
    text_col: Annotated[str, Form()],
    stopwords: Annotated[bool, Form()] = True,
):
    _, _, texts, cleaned, labels, probas, truncated = await _score_upload(file, text_col, stopwords)
    total = len(labels)
    counts = Counter(labels)
    present = set(labels)
    neg_terms = top_terms_by_sentiment(texts, cleaned, labels, "Negative") if "Negative" in present else []
    pos_terms = top_terms_by_sentiment(texts, cleaned, labels, "Positive") if "Positive" in present else []
    cat_all = category_counts(cleaned)
    cat_neg = category_counts(cleaned, labels, "Negative") if "Negative" in present else {}

    negatives = sorted(
        ({"text": t, "confidence": max(p.values())}
         for t, lbl, p in zip(texts, labels, probas) if lbl == "Negative"),
        key=lambda r: -r["confidence"],
    )

    stats = {
        "total": total,
        "counts": {k: counts.get(k, 0) for k in LABELS},
        "negTerms": [{"term": t["term"], "count": t["count"]} for t in neg_terms[:3]],
        "posTerms": [{"term": t["term"], "count": t["count"]} for t in pos_terms[:3]],
        "negCategories": cat_neg,
    }
    return {
        "total": total,
        "truncated": truncated,
        "maxRows": MAX_ROWS,
        "counts": stats["counts"],
        "negTerms": neg_terms,
        "posTerms": pos_terms,
        "categoriesAll": cat_all,
        "categoriesNegative": cat_neg,
        "negatives": negatives[:NEGATIVE_LIST_LIMIT],
        "negativeCount": len(negatives),
        "stats": stats,
        "signature": guard.sign(stats),
    }


@app.post("/api/batch/csv")
async def batch_csv(
    file: Annotated[UploadFile, File()],
    text_col: Annotated[str, Form()],
    stopwords: Annotated[bool, Form()] = True,
):
    headers, kept, _, _, labels, probas, _ = await _score_upload(file, text_col, stopwords)
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(headers + ["predicted_sentiment", "confidence"])
    for row, lbl, p in zip(kept, labels, probas):
        w.writerow(row + [lbl, f"{max(p.values()):.4f}"])
    return Response(
        out.getvalue().encode("utf-8-sig"),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="sentiment_results.csv"'},
    )


class SignedStats(BaseModel):
    stats: dict
    signature: str


class AskIn(SignedStats):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


def _check_ai(request: Request, body: SignedStats):
    if not ai.enabled():
        raise HTTPException(503, "The AI features aren't switched on for this site.")
    if not guard.verify(body.stats, body.signature):
        raise HTTPException(400, "Run the batch analysis again, then retry.")
    ok, retry_after = guard.allow(guard.visitor_id(request.headers))
    if not ok:
        raise HTTPException(
            429,
            f"You've reached the limit of {guard.AI_REQUESTS_PER_WINDOW} AI requests per "
            f"{guard.WINDOW_SECONDS // 60} minutes. Try again in about {max(1, retry_after // 60)} minutes.",
        )


@app.post("/api/brief")
def brief(request: Request, body: SignedStats):
    _check_ai(request, body)
    return {"text": ai.batch_brief(body.stats), "model": ai.HF_MODEL_ID}


@app.post("/api/ask")
def ask(request: Request, body: AskIn):
    _check_ai(request, body)
    if not body.question.strip():
        raise HTTPException(400, "Type a question first.")
    return {"text": ai.answer(body.stats, body.question.strip()), "model": ai.HF_MODEL_ID}
