"""
Usage:
    python main.py demo        # runs entirely on synthetic data, no network needed
    python main.py rank        # builds real features for config.SETTINGS.default_universe
                                # and prints today's ranking (needs API keys + network)
    python main.py backtest    # walk-forward train/test + Baseline vs +News vs +NLP comparison
                                # (needs historical data already collected into a CSV)

Run `python main.py demo` first to see the whole pipeline work end to end
before pointing it at real data.
"""
from __future__ import annotations
import sys
from datetime import date, timedelta
import numpy as np
import pandas as pd

from config import SETTINGS
from universe import UNIVERSE, get_search_terms
from models.scoring_model import compute_composite_score, explain_score_change


def make_synthetic_dataset(n_months: int = 24, seed: int = 7) -> pd.DataFrame:
    """
    Generates a plausible fake panel (one row per ticker per month) so the
    scoring model, training pipeline, and backtest can all be exercised
    without hitting any external API. Do not use conclusions from this
    data for real decisions -- it's a plumbing test, not a market signal.
    """
    rng = np.random.default_rng(seed)
    tickers = list(SETTINGS.default_universe)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_months, freq="MS")

    rows = []
    for ticker in tickers:
        # give each ticker a persistent "quality" so scores aren't pure noise
        base_quality = rng.normal(0, 1)
        for d in dates:
            rows.append({
                "date": d,
                "ticker": ticker,
                "sector": "robotics_ai",
                "revenue_growth_yoy": rng.normal(0.15 + 0.05 * base_quality, 0.08),
                "gross_margin": np.clip(rng.normal(0.45 + 0.05 * base_quality, 0.05), 0, 0.9),
                "free_cash_flow_margin": rng.normal(0.10 + 0.03 * base_quality, 0.06),
                "net_debt": rng.normal(500_000_000, 300_000_000),
                "ev_to_revenue": np.clip(rng.normal(8 - base_quality, 3), 1, 40),
                "momentum_6m": rng.normal(0.05 + 0.02 * base_quality, 0.12),
                "relative_momentum_12m": rng.normal(0.02 * base_quality, 0.10),
                "volatility_6m": np.clip(rng.normal(0.35, 0.08), 0.1, 1.0),
                "news_count_30d": max(0, rng.poisson(20 + 5 * base_quality)),
                "news_acceleration": np.clip(rng.normal(1.0 + 0.1 * base_quality, 0.3), 0.1, 5),
                "news_attention_zscore": rng.normal(0.1 * base_quality, 1.0),
                "unique_sources_30d": max(1, rng.poisson(8)),
                "sentiment_net": np.clip(rng.normal(0.05 * base_quality, 0.2), -1, 1),
                "commercial_event_score_90d": rng.normal(2 * base_quality, 3),
                "rnd_growth_yoy": rng.normal(0.12 + 0.04 * base_quality, 0.08),
                "share_dilution_yoy": rng.normal(0.02, 0.03),
            })
    df = pd.DataFrame(rows)

    # synthetic forward 12m relative return, loosely correlated with "quality"
    # so the walk-forward model has *something* learnable to find.
    quality_by_ticker = df.groupby("ticker")["revenue_growth_yoy"].transform("mean")
    noise = rng.normal(0, 0.15, size=len(df))
    df["forward_excess_return_12m"] = 0.6 * quality_by_ticker + noise
    df["outperformed_next_12m"] = (df["forward_excess_return_12m"] > df["forward_excess_return_12m"].median()).astype(int)
    return df


def run_demo() -> None:
    print("Building synthetic panel dataset (24 months x universe)...")
    dataset = make_synthetic_dataset()

    print("\nComputing composite (transparent, weighted) scores for the latest month...")
    latest_date = dataset["date"].max()
    latest = dataset[dataset["date"] == latest_date].copy()
    scored = compute_composite_score(latest)
    ranked = scored.sort_values("composite_score", ascending=False)

    print(f"\nTop-ranked tickers as of {latest_date.date()}:")
    print(ranked[["ticker", "composite_score"]].head(10).to_string(index=False))

    # explain the #1 ticker's move vs the prior month, if we have one
    top_ticker = ranked.iloc[0]["ticker"]
    prior_date_options = sorted(dataset["date"].unique())
    if len(prior_date_options) >= 2:
        prior_date = prior_date_options[-2]
        prior_row_df = dataset[(dataset["date"] == prior_date) & (dataset["ticker"] == top_ticker)]
        if not prior_row_df.empty:
            prior_scored = compute_composite_score(
                pd.concat([dataset[dataset["date"] == prior_date], latest])
            )
            prior_row = prior_scored[(prior_scored["date"] == prior_date) & (prior_scored["ticker"] == top_ticker)].iloc[0]
            current_row = prior_scored[(prior_scored["date"] == latest_date) & (prior_scored["ticker"] == top_ticker)].iloc[0]
            print(f"\nWhy {top_ticker} is ranked #1 (change vs prior month):")
            print(explain_score_change(current_row, prior_row))

    print("\nRunning walk-forward ML training on the synthetic panel...")
    from models.train_model import train_walk_forward_model
    training_end = dataset["date"].quantile(0.6)
    testing_start = dataset["date"].quantile(0.65)
    try:
        pipeline, test_results = train_walk_forward_model(
            dataset, training_end=str(training_end.date()), testing_start=str(testing_start.date())
        )
        print("\nSample predictions on held-out synthetic data:")
        print(
            test_results[["date", "ticker", "prediction_probability", "outperformed_next_12m"]]
            .sort_values("prediction_probability", ascending=False)
            .head(10)
            .to_string(index=False)
        )
    except ValueError as e:
        print(f"(Skipped ML step: {e})")

    print(
        "\nDemo complete. This ran entirely on synthetic data to prove the "
        "pipeline works end to end -- it is NOT a real ranking. Run "
        "`python main.py rank` with API keys set and network access for "
        "real data."
    )


def run_rank() -> None:
    """Builds the full feature set for the real universe using live APIs."""
    from data.prices import fetch_price_history
    from data.fundamentals import fetch_company_facts, get_cik_for_ticker
    from data.news import fetch_company_news, remove_duplicate_articles
    from features.price_features import calculate_price_features
    from features.fundamental_features import calculate_fundamental_features
    from features.news_features import calculate_news_features
    from features.text_features import score_financial_sentiment, aggregate_events

    today = date.today()
    lookback_start = today - timedelta(days=SETTINGS.news_baseline_window_days)
    rows = []

    for ticker in SETTINGS.default_universe:
        print(f"Processing {ticker}...")
        try:
            prices = fetch_price_history(ticker, today - timedelta(days=400), today)
            price_feats = calculate_price_features(prices["close"], pd.Timestamp(today))
        except Exception as e:
            print(f"  price fetch failed: {e}")
            price_feats = {}

        try:
            facts = fetch_company_facts(ticker, SETTINGS.sec_user_agent)
            fund_feats = calculate_fundamental_features(facts, today)
        except Exception as e:
            print(f"  fundamentals fetch failed: {e}")
            fund_feats = {}

        try:
            search_terms = get_search_terms(ticker)
            articles = fetch_company_news(search_terms, ticker, lookback_start, today, SETTINGS.news_api_key)
            articles = remove_duplicate_articles(articles)
            news_feats = calculate_news_features(articles, today)
            articles = score_financial_sentiment(articles)
            sentiment_net = float(articles["sentiment_net"].tail(30).mean()) if not articles.empty else 0.0
            event_feats = aggregate_events(articles)
        except Exception as e:
            print(f"  news fetch failed: {e}")
            news_feats, sentiment_net, event_feats = {}, 0.0, {}

        rows.append({
            "date": pd.Timestamp(today), "ticker": ticker, "sector": "robotics_ai",
            **price_feats, **fund_feats, **news_feats,
            "sentiment_net": sentiment_net, **event_feats,
        })

    df = pd.DataFrame(rows)
    scored = compute_composite_score(df)
    ranked = scored.sort_values("composite_score", ascending=False)
    print("\nFinal ranking:")
    print(ranked[["ticker", "composite_score"]].to_string(index=False))
    ranked.to_csv("output/ranking.csv", index=False)
    print("\nSaved to output/ranking.csv")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command == "demo":
        run_demo()
    elif command == "rank":
        import os
        os.makedirs("output", exist_ok=True)
        run_rank()
    else:
        print(__doc__)
