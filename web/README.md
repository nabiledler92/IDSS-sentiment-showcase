# IDSS web app (Vercel)

The Vercel version of the Streamlit app, live at **https://idss.nabil-takiyuddin.dev**. It has three tabs, Batch Analyze, Analyze a Review and Model Info, plus the Hugging Face AI brief and Q&A. To retrain on your own data, use the [Streamlit version](https://kopi-sentiment-idss.streamlit.app/).

```
web/
  vercel.json          Vercel Services: /api/* → backend, everything else → frontend
  backend/             FastAPI on Vercel Functions (Python 3.13)
    main.py            routes
    idss/              model loading, text pipeline, file reading, Hugging Face, abuse guard
    model/             copies of the deployed model files (see "Keeping in sync")
    tests/             parity tests against app_pipeline.py, plus API tests
  frontend/            Next.js (App Router), kopitiam theme
  scripts/sync_artifacts.py
```

## How it works

- **Predictions** use the same pickled model files as the Streamlit app (`best_model.pkl`, `tfidf_vectorizer.pkl`, `label_encoder.pkl`). `backend/tests` checks that the cleaned text, labels, probabilities, themes and category counts match the Streamlit pipeline exactly.
- **Uploads** are read in memory and never stored. Limits: 4 MB and 5,000 rows per file (Vercel caps request bodies at 4.5 MB).
- **AI brief and Q&A** call `Qwen/Qwen2.5-72B-Instruct` through Hugging Face with the `HF_TOKEN` environment variable, server-side only. The model sees only the aggregated batch statistics, never review text. To keep the endpoints from being used as a free LLM proxy:
  - `/api/batch` signs the statistics it returns (HMAC), and `/api/brief` and `/api/ask` reject any statistics without a valid signature.
  - Each visitor gets 8 AI requests per 10 minutes. The limit is held in memory per function instance, so it isn't global; add a Vercel Firewall rate-limit rule on `/api/brief` and `/api/ask` for a hard cap.
  - Without `HF_TOKEN`, the AI sections say the features are switched off. Everything else works.

## Run it locally

Two terminals, from `web/`:

```bash
# 1. API on http://127.0.0.1:8000
cd backend
python -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn main:app --port 8000      # set HF_TOKEN first to try the AI features

# 2. Web app on http://localhost:3000 (proxies /api to the API in dev)
cd frontend
npm install
npm run dev
```

Tests: `cd backend && python -m pytest -q`. They import the repository's `app_pipeline.py` to compare results, which is why `requirements-dev.txt` also installs pandas, XGBoost and imbalanced-learn.

## Keeping in sync

After re-running the experiments (`notebooks/extended_20_results/stage6_export_deployed.py`), copy the new model files and results into `web/` from the repository root:

```bash
python web/scripts/sync_artifacts.py          # copy
python web/scripts/sync_artifacts.py --check  # fail if anything is out of date
```

## Deploying

The Vercel project uses `web/` as its root directory, with `vercel.json` defining the two services. Set `HF_TOKEN` in the project's environment variables (Production). `IDSS_SIGNING_SECRET` is optional: without it, the signing key is derived from `HF_TOKEN`.
