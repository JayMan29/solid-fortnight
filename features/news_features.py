"""
Attention features: raw counts are weak signals on their own, so we
normalize against each company's own trailing baseline (acceleration
and z-score) and weight by source credibility.
"""
from __future__ import annotations
from datetime import date
import numpy as np
import pandas as pd
from data.news import source_weight


def calculate_news_features(articles: pd.DataFrame, as_of_date: date) -> dict[str, float]:
    if articles.empty:
        return {
            "news_count_30d": 0.0,
            "news_count_90d": 0.0,
            "news_acceleration": 1.0,
            "news_attention_zscore": 0.0,
            "unique_sources_30d": 0.0,
            "weighted_mentions_30d": 0.0,
        }

    cutoff = pd.Timestamp(as_of_date, tz="UTC")
    articles = articles.loc[articles["published_at"] <= cutoff].copy()

    start_30d = cutoff - pd.Timedelta(days=30)
    start_90d = cutoff - pd.Timedelta(days=90)
    start_12m = cutoff - pd.Timedelta(days=365)

    recent_30 = articles.loc[articles["published_at"] > start_30d]
    recent_90 = articles.loc[articles["published_at"] > start_90d]
    prior_60 = articles.loc[
        (articles["published_at"] > start_90d) & (articles["published_at"] <= start_30d)
    ]
    trailing_12m = articles.loc[articles["published_at"] > start_12m]

    count_30 = len(recent_30)
    prior_monthly_avg = len(prior_60) / 2
    acceleration = (count_30 + 1) / (prior_monthly_avg + 1)

    # Monthly bucket counts over the trailing year, for a z-score of current attention
    if not trailing_12m.empty:
        monthly_counts = (
            trailing_12m.set_index("published_at").resample("30D").size()
        )
        mu, sigma = monthly_counts.mean(), monthly_counts.std()
        z = float((count_30 - mu) / sigma) if sigma and sigma > 0 else 0.0
    else:
        z = 0.0

    weighted_mentions = float(recent_30["source"].apply(source_weight).sum())

    return {
        "news_count_30d": float(count_30),
        "news_count_90d": float(len(recent_90)),
        "news_acceleration": float(acceleration),
        "news_attention_zscore": z,
        "unique_sources_30d": float(recent_30["source"].nunique()),
        "weighted_mentions_30d": weighted_mentions,
    }
