"""
News retrieval and near-duplicate removal.

Uses NewsAPI (newsapi.org) by default. Swap in Polygon's news endpoint
if you have that key instead -- the shape of `fetch_company_news`'s
return value is what the rest of the pipeline depends on, so keep the
column names (`published_at`, `source`, `title`, `description`, `url`)
stable if you change providers.
"""
from __future__ import annotations
from datetime import date
from typing import Any
import pandas as pd
import requests
from rapidfuzz.fuzz import ratio

NEWS_API_URL = "https://newsapi.org/v2/everything"


def fetch_company_news(
    query: str,
    ticker: str,
    start_date: date,
    end_date: date,
    api_key: str,
    page_size: int = 100,
) -> pd.DataFrame:
    """
    `query` should already be an alias-safe boolean query, e.g.
    universe.get_search_query(ticker) -- NOT the raw ticker, since
    tickers like "ON", "ALL", "AI" collide with common words.
    """
    if not api_key:
        raise RuntimeError("NEWS_API_KEY is not set")
    params = {
        "q": query,
        "from": start_date.isoformat(),
        "to": end_date.isoformat(),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    response = requests.get(NEWS_API_URL, params=params, timeout=30)
    response.raise_for_status()
    articles: list[dict[str, Any]] = response.json().get("articles", [])
    rows = []
    for article in articles:
        rows.append({
            "ticker": ticker,
            "published_at": pd.to_datetime(article.get("publishedAt"), utc=True, errors="coerce"),
            "source": (article.get("source") or {}).get("name"),
            "title": article.get("title") or "",
            "description": article.get("description") or "",
            "url": article.get("url"),
        })
    return pd.DataFrame(rows)


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
    "reuters": 1.2,
    "bloomberg": 1.2,
    "the wall street journal": 1.2,
    "sec.gov": 1.5,
    "techcrunch": 0.8,
    "business insider": 0.6,
}
DEFAULT_SOURCE_WEIGHT = 0.4


def source_weight(source_name: str | None) -> float:
    if not source_name:
        return DEFAULT_SOURCE_WEIGHT
    return SOURCE_WEIGHTS.get(source_name.strip().lower(), DEFAULT_SOURCE_WEIGHT)
