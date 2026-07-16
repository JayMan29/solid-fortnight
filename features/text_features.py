"""
Two independent text-derived signals:

1. Financial sentiment via FinBERT (local model, no API key needed).
2. Structured commercial-event extraction via the Claude API (needs
   ANTHROPIC_API_KEY). This replaces generic "positive/negative" scoring
   with materiality-aware event typing, per the schema in the project brief.
"""
from __future__ import annotations
import json
import os
import requests
import pandas as pd

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

EVENT_SCHEMA_PROMPT = """Analyze the article only using information explicitly stated in the text.
Identify:
1. The publicly traded company and ticker (if determinable).
2. Event type: one of [new_product, customer_contract, partnership,
   factory_expansion, regulatory_approval, patent_or_research_milestone,
   earnings_beat, guidance_increase, executive_departure, product_delay,
   lawsuit, capital_raise, share_dilution, acquisition, government_funding,
   other].
3. Whether this represents a concept, prototype, pilot, signed contract,
   shipment, deployed product, or recognized revenue.
4. Any dollar value, unit count, customer, deadline, production target,
   or revenue impact mentioned.
5. Positive and negative implications.
6. Whether the claim is independently confirmed or only attributed to
   company management.
7. A materiality score from -5 (very negative / severe) to +5
   (very positive / highly material), where vague forward-looking language
   ("exploring", "could", "plans to", "aims to") scores near 0.
8. Confidence from 0 to 1.

Return ONLY valid JSON matching this schema, no other text:
{"ticker": str, "event_type": str, "stage": str, "has_dollar_value": bool,
 "has_named_customer": bool, "has_deployment_date": bool,
 "is_management_claim_only": bool, "materiality_score": float,
 "confidence": float}

Article:
"""


def score_financial_sentiment(articles: pd.DataFrame) -> pd.DataFrame:
    """Adds sentiment_positive/negative/neutral/net columns using FinBERT."""
    from transformers import pipeline  # heavy import, done lazily

    result = articles.copy()
    if result.empty:
        for col in ["sentiment_positive", "sentiment_negative", "sentiment_neutral", "sentiment_net"]:
            result[col] = []
        return result

    sentiment_pipeline = pipeline(
        task="text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
    )
    texts = (result["title"].fillna("") + ". " + result["description"].fillna("")).str.slice(0, 1500).tolist()
    predictions = sentiment_pipeline(texts, batch_size=16, top_k=None)

    rows = []
    for prediction_set in predictions:
        score_map = {item["label"].lower(): float(item["score"]) for item in prediction_set}
        positive = score_map.get("positive", 0.0)
        negative = score_map.get("negative", 0.0)
        neutral = score_map.get("neutral", 0.0)
        rows.append({
            "sentiment_positive": positive,
            "sentiment_negative": negative,
            "sentiment_neutral": neutral,
            "sentiment_net": positive - negative,
        })
    return pd.concat([result.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def extract_event(article_text: str, api_key: str | None = None) -> dict:
    """
    Calls the Claude API to extract a structured commercial event from
    a single article's title + description. Returns a dict matching
    EVENT_SCHEMA_PROMPT's schema, or a low-confidence default on failure.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    default = {
        "ticker": None, "event_type": "other", "stage": "unknown",
        "has_dollar_value": False, "has_named_customer": False,
        "has_deployment_date": False, "is_management_claim_only": True,
        "materiality_score": 0.0, "confidence": 0.0,
    }
    if not api_key or not article_text.strip():
        return default

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 300,
        "messages": [{"role": "user", "content": EVENT_SCHEMA_PROMPT + article_text}],
    }
    try:
        response = requests.post(CLAUDE_API_URL, headers=headers, json=body, timeout=30)
        response.raise_for_status()
        content_blocks = response.json().get("content", [])
        text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
        cleaned = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        return {**default, **parsed}
    except Exception:
        return default


def aggregate_events(articles: pd.DataFrame, api_key: str | None = None, max_articles: int = 20) -> dict[str, float]:
    """
    Runs event extraction over the most recent N articles (to control
    API cost) and rolls up materiality into a single commercial-event score.
    """
    if articles.empty:
        return {"commercial_event_score_90d": 0.0, "material_event_count_90d": 0.0}

    recent = articles.sort_values("published_at", ascending=False).head(max_articles)
    scores = []
    material_count = 0
    for _, row in recent.iterrows():
        text = f"{row.get('title', '')}. {row.get('description', '')}"
        event = extract_event(text, api_key=api_key)
        weighted = event["materiality_score"] * event["confidence"]
        scores.append(weighted)
        if abs(event["materiality_score"]) >= 3 and event["confidence"] >= 0.6:
            material_count += 1

    return {
        "commercial_event_score_90d": float(sum(scores)),
        "material_event_count_90d": float(material_count),
    }
