"""
News retrieval and near-duplicate removal.

Uses the Currents API (currentsapi.services) -- free tier allows
production/scheduled use (unlike NewsAPI.org's dev-only free tier).
Docs: https://currentsapi.services/en/docs/search
"""
from __future__ import annotations
from datetime import date
from typing import Any
from urllib.parse import urlparse
import pandas as pd
import requests
from rapidfuzz.fuzz import ratio

CURRENTS_SEARCH_URL = "https://api.currentsapi.services/v1/search"


def _domain_from_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.replace("www.", "") or None
    except ValueError:
        return None


def fetch_company_news(
    search_terms: list[str],
    ticker: str,
    start_date: date,
    end_date: date,
    api_key: str,
    page_size: int = 100,
) -> pd.DataFrame:
    """
    `search_terms` should be the company name + known aliases (see
    universe.get_search_terms) -- NOT the raw ticker, since tickers
    like "ON", "ALL", "AI" collide with common words. Currents' free
    tier doesn't reliably support boolean OR syntax, so we run one
    request per alias and merge + dedupe the results.
    """
    if not api_key:
        raise RuntimeError("NEWS_API_KEY is not set")

    headers = {"Authorization": api_key}
    all_rows: list[dict[str, Any]] = []

    for term in search_terms:
        params = {
            "keywords": term,
            "language": "en",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "page_size": page_size,
        }
        response = requests.get(CURRENTS_SEARCH_URL, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "ok":
            continue
        for article in payload.get("news", []):
            all_rows.append({
                "ticker": ticker,
                "published_at": pd.to_datetime(article.get("published"), utc=True, errors="coerce"),
                "source": _domain_from_url(article.get("url")),
                "title": article.get("title") or "",
                "description": article.get("description") or "",
                "url": article.get("url"),
            })

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset="url").reset_index(drop=True)


def remove_duplicate_articles(articles: pd.DataFrame, similarity_threshold: int = 88) -> pd.DataFrame:
    """Collapse syndicated copies of the same story to a single event."""
    if articles.empty:
        return articles.copy()
    articles = articles.sort_values("published_at").reset_index(drop=True)
    retained_indices: list[int] = []
    retained_titles: list[str] = []
    for index, row in articles.iterrows():
        title = str(row["title"]).lower().strip()
        is_duplicate = any(ratio(title, t) >= similarity_threshold for t in retained_titles)
        if not is_duplicate:
            retained_indices.append(index)
            retained_titles.append(title)
    return articles.loc[retained_indices].reset_index(drop=True)


SOURCE_WEIGHTS = {
    "reuters.com": 1.2,
    "bloomberg.com": 1.2,
    "wsj.com": 1.2,
    "sec.gov": 1.5,
    "techcrunch.com": 0.8,
    "businessinsider.com": 0.6,
}
DEFAULT_SOURCE_WEIGHT = 0.4


def source_weight(source_name: str | None) -> float:
    if not source_name:
        return DEFAULT_SOURCE_WEIGHT
    return SOURCE_WEIGHTS.get(source_name.strip().lower(), DEFAULT_SOURCE_WEIGHT)
