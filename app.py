"""
Streamlit app for the Knowledge-Driven IDSS sentiment module.

Ships with the pre-trained Kopi Kenangan Logistic Regression with SMOTE model
loaded by default, but is NOT hard-coded to that dataset: the "Train on Your Own Data"
tab lets any SME upload a different CSV/XLSX of reviews (with a text column
and either a star rating or an existing label column) and train a fresh set
of 6 models on it, right inside the app, using the exact same pipeline
(app_pipeline.py) used to build the default model.

Visual theme: "Kopitiam Modern" — warm espresso/cream palette (see
.streamlit/config.toml) with a small custom-CSS layer for sentiment cards,
KPI tiles and term-highlight styling. Purely presentational; no change to
the underlying prediction/training logic below.

Run with:  streamlit run app.py   (from the repository root)
"""
import io
import os
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import InferenceClient

from app_pipeline import (
    prepare_dataset, score_to_3class, train_and_evaluate, predict_sentiment,
    MODEL_REGISTRY, clean_text, top_terms_by_sentiment, generate_model_suggestion,
    category_counts,
)
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(page_title="Knowledge-Driven IDSS for Malaysian Kopi F&B SME", layout="wide")

# ----------------------------------------------------------------------
# Kopitiam Modern theme: sentiment palette + custom CSS layer.
# Colors mirror .streamlit/config.toml (primary #8B5E34 / cream bg) so the
# hand-built cards below feel native to Streamlit's own theme rather than
# bolted on.
# ----------------------------------------------------------------------
PALETTE = {
    "Negative": {"accent": "#B4432F", "bg": "#F6E3DF", "text": "#6B2417"},
    "Neutral":  {"accent": "#C99A3C", "bg": "#F7EDD8", "text": "#6B4F16"},
    "Positive": {"accent": "#4F7C4A", "bg": "#E4EDE3", "text": "#2C4A29"},
}
NEUTRAL_ACCENT = "#8B5E34"


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"], .stMarkdown, p, div, span { font-family: 'Inter', sans-serif; }
        h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 600 !important; }

        /* ---- sentiment result card (Analyze a Review tab) ---- */
        .kt-card {
            border-radius: 10px;
            border-left: 6px solid var(--accent);
            background: var(--bg);
            padding: 18px 22px;
            margin: 6px 0 18px 0;
        }
        .kt-badge {
            display: inline-block;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: .04em;
            text-transform: uppercase;
            color: var(--text);
            background: rgba(255,255,255,.55);
            border-radius: 999px;
            padding: 3px 12px;
            margin-bottom: 8px;
        }
        .kt-conf {
            font-family: 'Fraunces', serif;
            font-weight: 600;
            font-size: 34px;
            color: #2E2620;
            margin: 2px 0 4px 0;
            line-height: 1.1;
        }
        .kt-rec { font-size: 14px; color: #2E2620; margin-bottom: 14px; }
        .kt-probrow { display: flex; align-items: center; gap: 10px; margin: 5px 0; }
        .kt-problabel { width: 64px; font-size: 12px; color: #5A5147; flex-shrink: 0; }
        .kt-probtrack { flex: 1; height: 8px; border-radius: 5px; background: rgba(0,0,0,.08); overflow: hidden; }
        .kt-probfill { height: 100%; border-radius: 5px; }
        .kt-probpct { width: 42px; font-size: 12px; color: #5A5147; text-align: right; flex-shrink: 0; }

        /* ---- KPI tiles (Batch Analyze tab) ---- */
        .kt-kpi {
            border-radius: 10px;
            background: #F0E7DA;
            padding: 14px 16px;
            border-bottom: 3px solid var(--accent);
        }
        .kt-kpi-label { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #6B5E4F; margin: 0; }
        .kt-kpi-value { font-family: 'Fraunces', serif; font-weight: 600; font-size: 26px; color: #2E2620; margin: 2px 0 0 0; }
        .kt-kpi-sub { font-size: 11px; color: #8A7C6B; margin: 0; }

        /* ---- term-highlight expanders ---- */
        div[data-testid="stExpander"] {
            border: 1px solid #E6DCCB;
            border-radius: 8px;
            background: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sentiment_card_html(pred, proba, rec_text):
    pal = PALETTE.get(pred, {"accent": NEUTRAL_ACCENT, "bg": "#F0E7DA", "text": "#2E2620"})
    conf = max(proba.values()) * 100 if proba else None
    conf_html = f"{conf:.0f}%" if conf is not None else "&mdash;"

    rows = ""
    if proba:
        for label in ["Positive", "Neutral", "Negative"]:
            if label not in proba:
                continue
            pct = proba[label] * 100
            accent = PALETTE.get(label, {}).get("accent", NEUTRAL_ACCENT)
            rows += (
                f'<div class="kt-probrow">'
                f'<div class="kt-problabel">{label}</div>'
                f'<div class="kt-probtrack"><div class="kt-probfill" '
                f'style="width:{pct:.1f}%; background:{accent};"></div></div>'
                f'<div class="kt-probpct">{pct:.0f}%</div>'
                f'</div>'
            )

    return (
        f'<div class="kt-card" style="--accent:{pal["accent"]}; --bg:{pal["bg"]}; --text:{pal["text"]};">'
        f'<span class="kt-badge">{pred}</span>'
        f'<div class="kt-conf">{conf_html}<span style="font-size:14px; font-weight:400; color:#5A5147;"> confidence</span></div>'
        f'<div class="kt-rec">{rec_text}</div>'
        f'{rows}'
        f'</div>'
    )


def kpi_tile_html(label, value, sub, accent):
    return (
        f'<div class="kt-kpi" style="--accent:{accent};">'
        f'<p class="kt-kpi-label">{label}</p>'
        f'<p class="kt-kpi-value">{value}</p>'
        f'<p class="kt-kpi-sub">{sub}</p>'
        f'</div>'
    )


# ----------------------------------------------------------------------
# AI insights brief — an open-source LLM (via Hugging Face's serverless
# Inference API) turns the already-computed sentiment stats into a short,
# plain-language narrative. This is presentational/explanatory only: the
# exact numbers shown elsewhere on the page are the authoritative source of
# truth, not this brief. Generated on-demand (button click) rather than
# automatically, since the free-tier API is rate-limited.
#
# Swap HF_MODEL_ID for any other instruction-tuned model on the Hub if this
# one becomes unavailable via Inference Providers.
# ----------------------------------------------------------------------
HF_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
# Where visitors are sent to learn how to enable the AI features (README's
# "Run it locally" section, which covers adding HF_TOKEN).
AI_SETUP_URL = "https://github.com/nabiledler92/IDSS-sentiment-showcase#run-it-locally"


@st.cache_resource(show_spinner=False)
def get_hf_client(token):
    return InferenceClient(model=HF_MODEL_ID, provider="auto", token=token)


def get_hf_token():
    """Returns the configured HF_TOKEN from Streamlit secrets, falling back to
    the HF_TOKEN environment variable, or "" if neither is set. st.secrets
    raises when no secrets.toml exists at all, so it is wrapped in a
    try/except rather than a plain .get()."""
    try:
        token = str(st.secrets.get("HF_TOKEN", "")).strip()
    except Exception:
        token = ""
    return token or os.environ.get("HF_TOKEN", "").strip()


def hf_token_available():
    return bool(get_hf_token())


def call_hf_brief(system_prompt, user_prompt, max_tokens=220):
    """Call the HF-hosted open-source model and return plain text, or raise."""
    token = get_hf_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face API token configured. Add HF_TOKEN to "
            ".streamlit/secrets.toml (local) or the app's Secrets settings (Streamlit Cloud)."
        )
    client = get_hf_client(token)
    completion = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.4,
    )
    text = completion.choices[0].message.content.strip()
    if not text:
        raise RuntimeError(
            "The model returned an empty response. This can happen if the free "
            "Inference API is warming up — try clicking the button again in a few seconds."
        )
    return text


def generate_batch_insights_brief(total, counts, neg_terms, pos_terms, neg_categories=None):
    pct = {label: round(100 * int(counts.get(label, 0)) / total, 1) if total else 0
           for label in ["Negative", "Neutral", "Positive"]}
    neg_list = ", ".join(f"\"{r.term}\" ({r.count} reviews)" for r in neg_terms.itertuples()) or "none identified"
    pos_list = ", ".join(f"\"{r.term}\" ({r.count} reviews)" for r in pos_terms.itertuples()) or "none identified"
    if neg_categories is not None and len(neg_categories):
        cat_list = ", ".join(f"{cat} ({int(n)} reviews)" for cat, n in neg_categories.items() if n > 0) or "none identified"
    else:
        cat_list = "not available"

    system_prompt = (
        "You are a concise business analyst writing for a Malaysian Kopi F&B SME product "
        "manager who has no data-science background. Based ONLY on the numbers and themes "
        "given to you — never invent details, review content, categories, or statistics not "
        "present in the input — write a short insights brief in this exact structure:\n"
        "1. One-sentence executive summary of the headline finding.\n"
        "2. Exactly 3 recommendations, each as a bold markdown header (e.g. **Short-term: "
        "...**) followed by one bullet point of concrete, practical, low-cost action a coffee "
        "shop manager could act on directly. Prioritize the categories/themes with the most "
        "reviews first.\n"
        "Keep the whole brief under 150 words total."
    )
    user_prompt = (
        f"Batch of {total} customer reviews just analyzed.\n"
        f"Sentiment breakdown: Negative {counts.get('Negative', 0)} ({pct['Negative']}%), "
        f"Neutral {counts.get('Neutral', 0)} ({pct['Neutral']}%), "
        f"Positive {counts.get('Positive', 0)} ({pct['Positive']}%).\n"
        f"Top recurring negative themes: {neg_list}.\n"
        f"Top recurring positive themes: {pos_list}.\n"
        f"Negative reviews by business category (keyword-rule classified — Service, Product "
        f"Quality, Price): {cat_list}.\n"
        "Write the structured insights brief now."
    )
    return call_hf_brief(system_prompt, user_prompt, max_tokens=320)


def generate_training_insights_brief(comp_df, best_model_name, n_train, n_test):
    table_text = "; ".join(
        f"{row.Model}: accuracy {row.Accuracy:.3f}, weighted F1 {row['F1 (weighted)']:.3f}, "
        f"training time {row['Train time (s)']:.1f}s"
        for row in comp_df.itertuples()
    )
    system_prompt = (
        "You are a data science assistant explaining a model comparison to a non-technical "
        "SME product manager. Write a short plain-English summary (max 100 words) of which "
        "model performed best and why, based ONLY on the metrics given — do not invent "
        "numbers. End with exactly one concrete, practical next action."
    )
    user_prompt = (
        f"Trained on {n_train} rows, tested on {n_test} rows. Model comparison results: "
        f"{table_text}. Selected best model: {best_model_name}. Write the summary now."
    )
    return call_hf_brief(system_prompt, user_prompt)


def ai_brief_section(cache_key, generate_fn, *, button_label="Generate AI insights",
                      enabled=True, disabled_caption=None):
    """Shared on-demand AI-brief UI block: button -> spinner -> styled result
    card, with graceful handling when no HF token is configured yet.

    `enabled=False` renders the same header + a disabled button (so the
    feature is visible/advertised up front, before any data has been
    uploaded/analyzed) with `disabled_caption` explaining what's needed,
    instead of the functional button. `generate_fn` is never called in that
    state, so it's safe to pass one that would error on missing data."""
    st.markdown("**🤖 AI insights brief**")
    if not hf_token_available():
        st.caption(
            "Switched off in this public demo. When you run the app yourself with a free "
            "Hugging Face token, it writes a short, plain-language brief of the results "
            f"using the open-source `{HF_MODEL_ID}` model. "
            f"[How to turn it on]({AI_SETUP_URL})"
        )
        return
    if not enabled:
        st.button(button_label, key=f"{cache_key}_btn", disabled=True)
        st.caption(disabled_caption or "Upload and analyze data below to enable this.")
        return
    error_key = f"{cache_key}_error"
    if st.button(button_label, key=f"{cache_key}_btn"):
        with st.spinner("Asking the AI model for a quick read on this data..."):
            try:
                st.session_state[cache_key] = generate_fn()
                st.session_state[error_key] = None
            except Exception as e:
                st.session_state[cache_key] = None
                st.session_state[error_key] = str(e)
    if st.session_state.get(error_key):
        # Persisted (not a transient st.error inside the button block above) so it
        # stays visible across reruns — e.g. scrolling — instead of flashing once
        # and disappearing, which is what made earlier failures look like nothing
        # happened at all.
        st.error(f"Couldn't generate an AI brief right now: {st.session_state[error_key]}")
    if st.session_state.get(cache_key):
        st.markdown(
            f'<div class="kt-card" style="--accent:{NEUTRAL_ACCENT}; --bg:#F0E7DA; --text:#2E2620;">'
            f'<span class="kt-badge">AI-generated</span>'
            f'<div style="font-size:14px; color:#2E2620; margin-top:6px;">{st.session_state[cache_key]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Written by an open-source model ({HF_MODEL_ID}) via Hugging Face — a narrative "
            "read on the data above, not a substitute for the exact figures."
        )


def build_qa_context(total, counts, neg_terms, pos_terms, cat_counts):
    """Fixed text block of already-computed stats, re-sent as the system
    prompt on every chat turn (see qa_chat_section) so answers stay grounded
    in the same numbers shown elsewhere on the page."""
    pct = {label: round(100 * int(counts.get(label, 0)) / total, 1) if total else 0
           for label in ["Negative", "Neutral", "Positive"]}
    neg_list = ", ".join(f"\"{r.term}\" ({r.count} reviews)" for r in neg_terms.itertuples()) or "none identified"
    pos_list = ", ".join(f"\"{r.term}\" ({r.count} reviews)" for r in pos_terms.itertuples()) or "none identified"
    cat_list = ", ".join(f"{cat} ({int(n)} reviews)" for cat, n in cat_counts.items() if n > 0) or "none identified"
    return (
        f"Total reviews analyzed: {total}.\n"
        f"Sentiment breakdown: Negative {counts.get('Negative', 0)} ({pct['Negative']}%), "
        f"Neutral {counts.get('Neutral', 0)} ({pct['Neutral']}%), "
        f"Positive {counts.get('Positive', 0)} ({pct['Positive']}%).\n"
        f"Top recurring negative themes: {neg_list}.\n"
        f"Top recurring positive themes: {pos_list}.\n"
        f"Negative reviews by business category (Service, Product Quality, Price; "
        f"keyword-rule classified, a review can match more than one category): {cat_list}."
    )


def generate_qa_answer(question, context):
    """Answer a follow-up question grounded strictly in `context` (the same
    fixed stats shown elsewhere on the page) — the model never sees the raw
    review text, only the aggregated numbers, matching the same
    presentational/non-authoritative pattern as the AI insights brief."""
    system_prompt = (
        "You are a business analyst assistant for a Malaysian Kopi F&B SME product manager. "
        "You may ONLY use the DATA block below to answer — never invent numbers, review "
        "content, or details not present in it. If the question cannot be answered from this "
        "data, say so plainly (e.g. \"I don't have that information in the current batch "
        "data\") instead of guessing. Keep answers under 100 words, plain English, practical.\n\n"
        f"DATA:\n{context}"
    )
    return call_hf_brief(system_prompt, question, max_tokens=220)


def qa_chat_section(cache_key, context):
    """On-demand, grounded follow-up Q&A. Every turn re-sends the same fixed
    `context` string as the system prompt (not prior chat turns re-used as
    additional "facts", and not the raw review text) so answers can't drift
    into invented numbers as the conversation goes on."""
    st.markdown("**💬 Ask a follow-up question**")
    if not hf_token_available():
        st.caption(
            "Switched off in this public demo. When you run the app yourself with a free "
            "Hugging Face token, you can ask questions about the results and get answers "
            f"from `{HF_MODEL_ID}` based only on the numbers shown here. "
            f"[How to turn it on]({AI_SETUP_URL})"
        )
        return

    history_key = f"{cache_key}_history"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    for role, msg in st.session_state[history_key]:
        with st.chat_message(role):
            st.write(msg)

    question = st.chat_input(
        "e.g. How can I lower my drink prep time without hiring more staff?",
        key=f"{cache_key}_input",
    )
    if question:
        st.session_state[history_key].append(("user", question))
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = generate_qa_answer(question, context)
                except Exception as e:
                    answer = f"Couldn't get an answer right now: {e}"
            st.write(answer)
        st.session_state[history_key].append(("assistant", answer))

    st.caption(
        f"Answers are grounded only in the stats already shown above (via {HF_MODEL_ID}) — "
        "the model never sees the raw review text, so it will say when something falls "
        "outside what it was given rather than guessing."
    )


inject_css()


# ----------------------------------------------------------------------
# Active-model state: defaults to the pre-trained Kopi Kenangan artifacts,
# can be swapped out after training on a custom dataset in Tab 3.
# ----------------------------------------------------------------------
def load_default_model():
    try:
        model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
        name_path = os.path.join(MODELS_DIR, "best_model_name.txt")
        if not os.path.exists(name_path):
            # finalize.py originally saved this under outputs/tables/ instead
            # of models/ - fall back there so the app still works either way.
            name_path = os.path.join(BASE_DIR, "outputs", "tables", "best_model_name.txt")
        with open(name_path) as f:
            name = f.read().strip()
    except FileNotFoundError:
        return None
    return {
        "model": model, "vectorizer": vectorizer, "label_encoder": label_encoder,
        "name": name, "source": "Default (Kopi Kenangan, Logistic Regression with SMOTE)",
        "classes": list(label_encoder.classes_),
    }


if "active" not in st.session_state:
    st.session_state.active = load_default_model()
if "train_result" not in st.session_state:
    st.session_state.train_result = None

st.title("Knowledge-Driven IDSS for Malaysian Kopi F&B SME Product Managers")
st.caption(
    "Classify customer review sentiment and get an at-a-glance action recommendation."
)

with st.sidebar:
    st.header("Active model")
    active = st.session_state.active
    if active is None:
        st.error("No model loaded yet. Train one in the 'Train on Your Own Data' tab.")
    else:
        st.success(f"**{active['name']}**")
        st.caption(active["source"])
        st.write("Classes:", ", ".join(active["classes"]))
    st.divider()
    st.caption(
        "This app is not tied to the Kopi Kenangan dataset — use the "
        "'Train on Your Own Data' tab to fit fresh models on any review "
        "export and make that the active model."
    )

tab1, tab2, tab3, tab4 = st.tabs([
    ":material/stacked_bar_chart: Batch Analyze",
    ":material/rate_review: Analyze a Review",
    ":material/model_training: Train on Your Own Data",
    ":material/info: Model Info",
])

RECOMMENDATION = {
    "Negative": "Flag for immediate follow-up — reach out to the customer and check for a "
                "recurring operational issue.",
    "Neutral": "Monitor — track whether this sentiment trend grows across similar reviews.",
    "Positive": "No action needed — consider highlighting in testimonials/marketing.",
}

# ----------------------------------------------------------------------
# Single review — content unchanged, now the second tab (":material/
# rate_review: Analyze a Review") so Batch Analyze can lead.
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Analyze a single review")
    active = st.session_state.active
    if active is None:
        st.warning("Train or load a model first.")
    else:
        review = st.text_area("Paste a customer review", height=120,
                               placeholder="e.g. Aplikasi nya bagus, tapi sering error saat checkout...")
        use_id_stopwords = st.checkbox(
            "Text is Indonesian/Malay (apply stop-word removal)", value=True, key="t1_stop")
        if st.button("Analyze sentiment", type="primary") and review.strip():
            pred, proba = predict_sentiment(
                review, active["model"], active["vectorizer"], active["label_encoder"],
                use_indonesian_stopwords=use_id_stopwords,
            )
            rec = RECOMMENDATION.get(pred, "")
            st.markdown(sentiment_card_html(pred, proba, rec), unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Batch analyze — now the first tab (":material/stacked_bar_chart:
# Batch Analyze") since it's the more powerful workflow.
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Batch analyze a file of reviews")
    active = st.session_state.active
    if active is None:
        st.warning("Train or load a model first.")
    else:
        # ------------------------------------------------------------
        # AI insights brief — shown up front (before the uploader) so the
        # feature is visible immediately, but disabled with an explanatory
        # caption until a batch has actually been analyzed below. Reads
        # from session_state, which the "Run batch analysis" button (further
        # down) populates and then st.rerun()s to refresh this section in
        # the same interaction, rather than requiring an extra click.
        # ------------------------------------------------------------
        t2_result_preview = st.session_state.get("t2_result")
        ai_brief_section(
            "t2_ai_brief",
            (lambda: generate_batch_insights_brief(
                len(t2_result_preview["df_out"]),
                t2_result_preview["df_out"]["predicted_sentiment"].value_counts(),
                t2_result_preview["neg_terms_all"].head(3),
                t2_result_preview["pos_terms_all"].head(3),
                neg_categories=t2_result_preview["cat_counts_neg"],
            )) if t2_result_preview is not None else (lambda: None),
            enabled=t2_result_preview is not None,
            disabled_caption="Upload a file and run batch analysis below to enable this.",
        )
        st.divider()

        up = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"], key="batch_upload")
        if up is not None:
            df_in = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
            text_col = st.selectbox("Which column contains the review text?", df_in.columns, key="t2_col")
            use_id_stopwords2 = st.checkbox(
                "Text is Indonesian/Malay (apply stop-word removal)", value=True, key="t2_stop")
            if st.button("Run batch analysis", type="primary"):
                with st.spinner("Scoring reviews..."):
                    preds, confidences, cleaned = [], [], []
                    for text in df_in[text_col].astype(str):
                        pred, proba = predict_sentiment(
                            text, active["model"], active["vectorizer"], active["label_encoder"],
                            use_indonesian_stopwords=use_id_stopwords2,
                        )
                        preds.append(pred)
                        confidences.append(max(proba.values()) if proba else None)
                        cleaned.append(clean_text(text, use_id_stopwords2))
                    df_out = df_in.copy()
                    df_out["predicted_sentiment"] = preds
                    df_out["confidence"] = confidences
                    df_out["_clean_text"] = cleaned

                # ----------------------------------------------------
                # Term extraction computed once here so it can feed both
                # the AI insights brief below and the Theme Highlights
                # section further down, without running twice.
                # ----------------------------------------------------
                available_classes = set(df_out["predicted_sentiment"].unique())
                empty_terms = pd.DataFrame(columns=["term", "count", "pct_of_class", "examples"])
                neg_terms_all = (
                    top_terms_by_sentiment(df_out[text_col], df_out["_clean_text"],
                                            df_out["predicted_sentiment"], target="Negative", top_n=10)
                    if "Negative" in available_classes else empty_terms
                )
                pos_terms_all = (
                    top_terms_by_sentiment(df_out[text_col], df_out["_clean_text"],
                                            df_out["predicted_sentiment"], target="Positive", top_n=10)
                    if "Positive" in available_classes else empty_terms
                )

                # Fixed-category breakdown (Service / Product Quality / Price),
                # computed once here alongside the term extraction above so it
                # can feed the category chart, the AI brief, and the Q&A chat
                # context without recomputing. Keyword-rule based (see
                # app_pipeline.categorize_reviews) — local and instant, no
                # extra API calls even for large batches.
                cat_counts_all = category_counts(df_out["_clean_text"])
                cat_counts_neg = (
                    category_counts(df_out["_clean_text"], df_out["predicted_sentiment"], "Negative")
                    if "Negative" in available_classes else cat_counts_all.iloc[0:0]
                )

                # Persist to session_state instead of only holding these as
                # local variables. Everything below used to live inside this
                # same "if st.button(...)" block, which meant it only ever
                # rendered during the exact script run where the button was
                # clicked — clicking anything nested inside it (like the AI
                # insights button) triggers a NEW rerun where this button is
                # False again, collapsing the whole results section back to
                # the pre-run upload screen. Storing the results here and
                # rendering them unconditionally below (gated on session_state
                # instead of the button's return value) fixes that.
                st.session_state["t2_result"] = {
                    "df_out": df_out,
                    "text_col": text_col,
                    "available_classes": available_classes,
                    "neg_terms_all": neg_terms_all,
                    "pos_terms_all": pos_terms_all,
                    "cat_counts_all": cat_counts_all,
                    "cat_counts_neg": cat_counts_neg,
                }
                # Rerun immediately so the AI insights section up top (which
                # reads session_state at the very start of the script) picks
                # up these results and switches from disabled to functional
                # in this same interaction, instead of staying disabled
                # until some unrelated later rerun.
                st.rerun()

            # ----------------------------------------------------------------
            # Results display — reads from session_state so it stays on
            # screen across reruns triggered by widgets below it (e.g. the
            # "Generate AI insights" button), instead of disappearing.
            # ----------------------------------------------------------------
            result = st.session_state.get("t2_result")
            if result is not None:
                df_out = result["df_out"]
                text_col_r = result["text_col"]
                available_classes = result["available_classes"]
                neg_terms_all = result["neg_terms_all"]
                pos_terms_all = result["pos_terms_all"]
                cat_counts_all = result["cat_counts_all"]
                cat_counts_neg = result["cat_counts_neg"]
                counts = df_out["predicted_sentiment"].value_counts()
                total = len(df_out)

                st.success(f"Scored {len(df_out)} reviews.")

                # ----------------------------------------------------
                # KPI summary row: total + per-class count/share, so a PM
                # gets the headline picture before scrolling into detail.
                # ----------------------------------------------------
                kpi_cols = st.columns(4)
                with kpi_cols[0]:
                    st.markdown(kpi_tile_html("Total reviews", f"{total}", "scored this batch", NEUTRAL_ACCENT),
                                unsafe_allow_html=True)
                for col, label in zip(kpi_cols[1:], ["Negative", "Neutral", "Positive"]):
                    n = int(counts.get(label, 0))
                    pct = (100 * n / total) if total else 0
                    accent = PALETTE[label]["accent"]
                    with col:
                        st.markdown(kpi_tile_html(label, f"{n}", f"{pct:.0f}% of batch", accent),
                                    unsafe_allow_html=True)

                st.write("")
                qa_chat_section(
                    "t2_qa",
                    build_qa_context(total, counts, neg_terms_all.head(3), pos_terms_all.head(3),
                                      cat_counts_neg),
                )

                st.write("")
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write("Sentiment breakdown")
                    st.bar_chart(counts)
                with col2:
                    st.write("Class breakdown")
                    st.dataframe(counts.rename("count"))

                if cat_counts_all.sum() > 0:
                    st.write("Review categories mentioned (Service / Product Quality / Price)")
                    st.caption(
                        "Keyword-rule classified from the cleaned review text — a review can "
                        "mention more than one category, so these counts don't need to add up "
                        "to the total review count. All reviews vs. Negative-only side by side, "
                        "so a category that's disproportionately tied to complaints (a tall "
                        "Negative bar relative to its All-reviews bar) stands out at a glance."
                    )
                    cat_compare = pd.DataFrame({
                        "All reviews": cat_counts_all,
                        "Negative reviews": cat_counts_neg.reindex(cat_counts_all.index),
                    }).fillna(0).astype(int)
                    st.bar_chart(cat_compare, stack=False)

                negatives = df_out[df_out["predicted_sentiment"] == "Negative"]
                if len(negatives):
                    st.write(f"⚠️ {len(negatives)} reviews flagged Negative — priority follow-up list:")
                    st.dataframe(negatives[[text_col_r, "predicted_sentiment", "confidence"]])

                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("Download full results (CSV)", csv_bytes,
                                    file_name="sentiment_results.csv", mime="text/csv")

                # --------------------------------------------------------
                # Theme highlights: distinctive recurring words/phrases per
                # sentiment class, so a PM gets concrete issues to fix and
                # strengths to promote instead of just a pile of reviews.
                # --------------------------------------------------------
                if {"Negative", "Positive"} & available_classes:
                    st.divider()
                    st.subheader("Theme highlights from this batch")
                    st.caption(
                        "Recurring words/phrases that show up disproportionately more "
                        "in one sentiment class than the rest of the batch — a quick "
                        "way to spot concrete issues to fix and strengths to promote, "
                        "beyond reading individual reviews one by one."
                    )
                    col_neg, col_pos = st.columns(2)

                    with col_neg:
                        st.markdown(
                            f'<span style="color:{PALETTE["Negative"]["accent"]}; font-weight:600;">'
                            f'🔧 Top issues to improve</span> <span style="color:#8A7C6B;">'
                            f'(from Negative reviews)</span>',
                            unsafe_allow_html=True,
                        )
                        if "Negative" not in available_classes:
                            st.info("No Negative reviews in this batch.")
                        elif neg_terms_all.empty:
                            st.info("Not enough Negative reviews to extract reliable themes.")
                        else:
                            for _, row in neg_terms_all.iterrows():
                                with st.expander(
                                    f"\"{row['term']}\" — {row['count']} reviews "
                                    f"({row['pct_of_class']}% of Negative)"
                                ):
                                    for ex in row["examples"]:
                                        st.write("•", ex)

                    with col_pos:
                        st.markdown(
                            f'<span style="color:{PALETTE["Positive"]["accent"]}; font-weight:600;">'
                            f'🌟 Top strengths to scale</span> <span style="color:#8A7C6B;">'
                            f'(from Positive reviews)</span>',
                            unsafe_allow_html=True,
                        )
                        if "Positive" not in available_classes:
                            st.info("No Positive reviews in this batch.")
                        elif pos_terms_all.empty:
                            st.info("Not enough Positive reviews to extract reliable themes.")
                        else:
                            for _, row in pos_terms_all.iterrows():
                                with st.expander(
                                    f"\"{row['term']}\" — {row['count']} reviews "
                                    f"({row['pct_of_class']}% of Positive)"
                                ):
                                    for ex in row["examples"]:
                                        st.write("•", ex)

# ----------------------------------------------------------------------
# TAB 3 — train on your own dataset (generalization)
# ----------------------------------------------------------------------
with tab3:
    st.subheader("Train fresh models on your own dataset")
    st.caption(
        "Upload any CSV/XLSX with a text column, plus either a numeric rating "
        "column (e.g. 1-5 stars) or an existing sentiment-label column. This "
        "runs the same six-classifier pipeline used to build the default Kopi "
        "Kenangan model, with SMOTE and GridSearchCV tuning available as "
        "optional toggles below (the default model itself uses SMOTE without "
        "tuning, which scored best in the 20-result comparison)."
    )

    # AI insights brief — shown up front, disabled until models have been
    # trained below (see tab2 for the same pattern and the st.rerun() note).
    t3_result_preview = st.session_state.get("train_result")
    ai_brief_section(
        "t3_ai_brief",
        (lambda: generate_training_insights_brief(
            pd.DataFrame([
                {"Model": name, "Accuracy": round(r["test_accuracy"], 4),
                 "F1 (macro)": round(r["test_f1_macro"], 4),
                 "F1 (weighted)": round(r["test_f1_weighted"], 4),
                 "Train time (s)": round(r["train_time_sec"], 2)}
                for name, r in t3_result_preview["results"].items()
            ]).sort_values("F1 (weighted)", ascending=False),
            t3_result_preview["best_model_name"], t3_result_preview["n_train"], t3_result_preview["n_test"],
        )) if t3_result_preview is not None else (lambda: None),
        enabled=t3_result_preview is not None,
        disabled_caption="Upload data and train models below to enable this.",
    )
    st.divider()

    with st.container(border=True):
        st.markdown("**Step 1 — Upload & configure columns**")
        up2 = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"], key="train_upload")

        labels = None
        text_col = None
        if up2 is not None:
            df_raw = pd.read_csv(up2) if up2.name.endswith(".csv") else pd.read_excel(up2)
            st.write(f"Loaded {len(df_raw)} rows.")
            st.dataframe(df_raw.head(5))

            text_col = st.selectbox("Text column", df_raw.columns, key="t3_text")
            label_mode = st.radio(
                "How is sentiment indicated in this dataset?",
                ["Derive from a numeric star rating (1-5)", "Use an existing sentiment/label column"],
            )

            if label_mode.startswith("Derive"):
                rating_col = st.selectbox("Rating column", df_raw.columns, key="t3_rating")
                neg_max = st.number_input("Ratings <= this value = Negative", 1, 5, 2)
                neu_val = st.number_input("Rating equal to this value = Neutral (0 to disable)", 0, 5, 3)
                try:
                    ratings = pd.to_numeric(df_raw[rating_col], errors="coerce")
                    labels = ratings.apply(lambda s: score_to_3class(s, neg_max, neu_val) if pd.notnull(s) else None)
                except Exception as e:
                    st.error(f"Could not read rating column: {e}")
            else:
                label_col = st.selectbox("Label column", df_raw.columns, key="t3_label")
                labels = df_raw[label_col].astype(str)

    if up2 is not None:
        with st.container(border=True):
            st.markdown("**Step 2 — Training options**")
            use_id_stopwords3 = st.checkbox(
                "Text is Indonesian/Malay (apply stop-word removal)", value=True, key="t3_stop")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                use_smote = st.checkbox("Use SMOTE for class imbalance", value=True)
            with col_b:
                use_tuning = st.checkbox("Use GridSearchCV tuning", value=True)
            with col_c:
                models_to_run = st.multiselect(
                    "Models to train", list(MODEL_REGISTRY.keys()), default=list(MODEL_REGISTRY.keys()))

            if st.button("Train models", type="primary") and labels is not None:
                work_df = df_raw.copy()
                work_df["_label"] = labels
                work_df = work_df.dropna(subset=["_label", text_col])
                work_df = prepare_dataset(work_df, text_col, use_id_stopwords3)
                work_df = work_df.dropna(subset=["_label"])

                if work_df["_label"].nunique() < 2:
                    st.error("Need at least 2 distinct sentiment classes to train. Check your column choices.")
                elif not models_to_run:
                    st.error("Select at least one model to train.")
                else:
                    progress = st.progress(0.0, text="Starting...")

                    def cb(i, total, name):
                        progress.progress((i) / total, text=f"Training {name} ({i+1}/{total})...")

                    try:
                        with st.spinner("Training — this can take up to a minute depending on dataset size..."):
                            result = train_and_evaluate(
                                work_df["clean_text"], work_df["_label"],
                                use_smote=use_smote, use_tuning=use_tuning,
                                models_to_run=models_to_run, progress_callback=cb,
                            )
                        progress.progress(1.0, text="Done!")
                        st.session_state.train_result = result
                        # Refresh immediately so the AI insights section up
                        # top switches from disabled to functional right away.
                        st.rerun()
                    except ValueError as e:
                        progress.empty()
                        st.error(f"Couldn't train on this dataset: {e}")
                    except Exception as e:
                        progress.empty()
                        st.error(
                            f"Something went wrong while training: {e}\n\n"
                            "Try a different text/label column, a larger dataset, or "
                            "fewer selected models."
                        )

    result = st.session_state.train_result
    if result is not None:
        with st.container(border=True):
            st.markdown("**Step 3 — Results**")
            st.success(f"Best model: **{result['best_model_name']}**  "
                       f"(trained on {result['n_train']} rows, tested on {result['n_test']})")
            rows = []
            for name, r in result["results"].items():
                rows.append({
                    "Model": name, "Accuracy": round(r["test_accuracy"], 4),
                    "F1 (macro)": round(r["test_f1_macro"], 4),
                    "F1 (weighted)": round(r["test_f1_weighted"], 4),
                    "Train time (s)": round(r["train_time_sec"], 2),
                })
            comp_df = pd.DataFrame(rows).sort_values("F1 (weighted)", ascending=False)
            st.dataframe(comp_df, use_container_width=True)
            st.bar_chart(comp_df.set_index("Model")[["Accuracy", "F1 (weighted)"]])

            st.markdown("#### 🤖 Smart model suggestion")
            st.info(generate_model_suggestion(result["results"]))
            st.write("")

            chosen = st.selectbox(
                "Set which trained model as the active model for Tabs 1 & 2",
                comp_df["Model"].tolist(), index=0, key="t3_choose",
            )
            if st.button("Set as active model"):
                st.session_state.active = {
                    "model": result["fitted_models"][chosen],
                    "vectorizer": result["vectorizer"],
                    "label_encoder": result["label_encoder"],
                    "name": chosen,
                    "source": f"Custom-trained on uploaded file ({up2.name})",
                    "classes": result["classes"],
                }
                st.success(f"{chosen} is now the active model — try it in Tab 1 or Tab 2.")

            buf_model = io.BytesIO()
            joblib.dump(result["fitted_models"][comp_df.iloc[0]["Model"]], buf_model)
            st.download_button(
                "Download best trained model (.pkl)", buf_model.getvalue(),
                file_name="custom_best_model.pkl",
            )

# ----------------------------------------------------------------------
# TAB 4 — model info
# ----------------------------------------------------------------------
def load_default_results():
    """Load the full 6-model comparison behind the default Kopi Kenangan
    model, so the same generate_model_suggestion() text used for custom
    datasets can also explain the default model's selection here."""
    for candidate in (
        os.path.join(MODELS_DIR, "model_results_all.json"),
        os.path.join(BASE_DIR, "outputs", "tables", "model_results_all.json"),
    ):
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.load(f)
    return None


with tab4:
    st.subheader("About the active model")
    active = st.session_state.active
    if active is None:
        st.warning("No model loaded.")
    else:
        st.write("**Name:**", active["name"])
        st.write("**Source:**", active["source"])
        st.write("**Sentiment classes:**", ", ".join(active["classes"]))
        # Metrics come from outputs/tables/model_results_all.json (written by
        # notebooks/extended_20_results/stage6_export_deployed.py), so this
        # text stays correct whenever the experiments are re-run.
        _default_results = load_default_results() or {}
        _best_default_name = (
            max(_default_results, key=lambda n: _default_results[n]["test_f1_weighted"])
            if _default_results else "Logistic Regression"
        )
        _best_metrics = _default_results.get(_best_default_name)
        _metrics_text = (
            f": {_best_metrics['test_accuracy'] * 100:.1f}% accuracy, "
            f"{_best_metrics['test_f1_weighted']:.3f} weighted F1"
            if _best_metrics else ""
        )
        st.write(
            "The default model was trained on 1,197 cleaned Kopi Kenangan app "
            f"reviews using TF-IDF features and a {_best_default_name} classifier "
            "on a SMOTE-balanced training set (default hyperparameters; in the "
            "dissertation's 20-result comparison this configuration "
            "outperformed the GridSearchCV-tuned version and both RF+XGBoost "
            f"ensemble combinations{_metrics_text})."
        )
        st.write(
            "To use this app for a different SME's data, go to **Train on Your "
            "Own Data**, upload a review export, and set the resulting best "
            "model as active — no code changes required."
        )

        if active["source"].startswith("Default"):
            default_results = load_default_results()
            if default_results:
                st.markdown("#### 🤖 Smart model suggestion")
                suggestion = generate_model_suggestion(default_results)
                # All 6 default-model results were trained with SMOTE (no tuning -
                # Chapter 4), so any bare classifier name generate_model_suggestion()
                # mentions here should read "<Name> with SMOTE" for consistency with
                # the thesis. This is done only at this call site, not inside
                # generate_model_suggestion() itself, because that function is shared
                # with Tab 3's "Train on Your Own Data" results, where SMOTE may or
                # may not have actually been used depending on the user's toggle.
                suggestion = re.sub(
                    r"Logistic Regression(?! with SMOTE)", "Logistic Regression with SMOTE", suggestion
                )
                st.info(suggestion)
