"""AI insights brief and grounded Q&A through Hugging Face.

Same model and prompts as the Streamlit app. The model only ever sees the
aggregated batch statistics, never raw review text. HF_TOKEN comes from the
Vercel project's environment variables and never reaches the browser.
"""
import os

HF_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


class AIError(RuntimeError):
    """A failure safe to show to the visitor."""


def token() -> str:
    return os.environ.get("HF_TOKEN", "").strip()


def enabled() -> bool:
    return bool(token())


def _chat(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    if not enabled():
        raise AIError("The AI features aren't switched on for this site.")
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=HF_MODEL_ID, provider="auto", token=token(), timeout=45)
    try:
        completion = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.4,
        )
    except Exception as exc:
        raise AIError(
            "Hugging Face didn't return an answer just now. The free inference tier may be "
            "busy or out of monthly credit — try again in a minute."
        ) from exc
    text = (completion.choices[0].message.content or "").strip()
    if not text:
        raise AIError("The model returned an empty response. Try again in a few seconds.")
    return text


def _pct(stats: dict) -> dict:
    total = stats["total"]
    return {k: round(100 * stats["counts"].get(k, 0) / total, 1) if total else 0
            for k in ("Negative", "Neutral", "Positive")}


def _terms(terms) -> str:
    return ", ".join(f"\"{t['term']}\" ({t['count']} reviews)" for t in terms) or "none identified"


def _cats(cats: dict) -> str:
    return ", ".join(f"{c} ({n} reviews)" for c, n in cats.items() if n > 0) or "none identified"


def batch_brief(stats: dict) -> str:
    c, pct = stats["counts"], _pct(stats)
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
        f"Batch of {stats['total']} customer reviews just analyzed.\n"
        f"Sentiment breakdown: Negative {c.get('Negative', 0)} ({pct['Negative']}%), "
        f"Neutral {c.get('Neutral', 0)} ({pct['Neutral']}%), "
        f"Positive {c.get('Positive', 0)} ({pct['Positive']}%).\n"
        f"Top recurring negative themes: {_terms(stats['negTerms'])}.\n"
        f"Top recurring positive themes: {_terms(stats['posTerms'])}.\n"
        f"Negative reviews by business category (keyword-rule classified — Service, Product "
        f"Quality, Price): {_cats(stats['negCategories'])}.\n"
        "Write the structured insights brief now."
    )
    return _chat(system_prompt, user_prompt, max_tokens=320)


def qa_context(stats: dict) -> str:
    c, pct = stats["counts"], _pct(stats)
    return (
        f"Total reviews analyzed: {stats['total']}.\n"
        f"Sentiment breakdown: Negative {c.get('Negative', 0)} ({pct['Negative']}%), "
        f"Neutral {c.get('Neutral', 0)} ({pct['Neutral']}%), "
        f"Positive {c.get('Positive', 0)} ({pct['Positive']}%).\n"
        f"Top recurring negative themes: {_terms(stats['negTerms'])}.\n"
        f"Top recurring positive themes: {_terms(stats['posTerms'])}.\n"
        f"Negative reviews by business category (Service, Product Quality, Price; "
        f"keyword-rule classified, a review can match more than one category): "
        f"{_cats(stats['negCategories'])}."
    )


def answer(stats: dict, question: str) -> str:
    system_prompt = (
        "You are a business analyst assistant for a Malaysian Kopi F&B SME product manager. "
        "You may ONLY use the DATA block below to answer — never invent numbers, review "
        "content, or details not present in it. If the question cannot be answered from this "
        "data, say so plainly (e.g. \"I don't have that information in the current batch "
        "data\") instead of guessing. Keep answers under 100 words, plain English, practical.\n\n"
        f"DATA:\n{qa_context(stats)}"
    )
    return _chat(system_prompt, question, max_tokens=220)
