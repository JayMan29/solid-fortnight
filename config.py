"""
Central configuration for the stock research pipeline.

All API keys are read from environment variables so nothing secret
ever lives in source control. Set these before running main.py:

    export SEC_USER_AGENT="Your Name your_email@example.com"
    export NEWS_API_KEY="..."          # https://newsapi.org
    export POLYGON_API_KEY="..."       # optional, https://polygon.io

NOTE ON NETWORK ACCESS:
This project makes live HTTP calls to SEC EDGAR, NewsAPI (or Polygon),
and a price data source. It will NOT run inside network-sandboxed
environments that only allow package registries. Run it locally or in
a normal cloud VM with outbound internet access.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    # --- API credentials ---
    sec_user_agent: str = os.environ.get(
        "SEC_USER_AGENT", "Research Script contact@example.com"
    )
    news_api_key: str = os.environ.get("NEWS_API_KEY", "")
    polygon_api_key: str = os.environ.get("POLYGON_API_KEY", "")

    # --- Universe ---
    # Keep this modest at first. Wide universes multiply API calls.
    default_universe: tuple[str, ...] = (
        "NVDA", "TSLA", "AVGO", "QCOM", "AMBA", "TER", "ROK", "ABB",
        "ISRG", "STM", "TXN", "HON", "TDY", "NOC", "AME",
    )

    # --- Feature windows (days) ---
    news_recent_window_days: int = 30
    news_baseline_window_days: int = 180
    momentum_window_days: int = 252
    volatility_window_days: int = 126

    # --- Scoring weights (must be revisited via backtest, not trusted blindly) ---
    weights: dict = field(default_factory=lambda: {
        "fundamental_growth": 0.20,
        "business_quality": 0.15,
        "price_momentum": 0.15,
        "news_acceleration": 0.10,
        "news_sentiment": 0.10,
        "commercial_event_quality": 0.15,
        "technology_position": 0.10,
        "financial_strength": 0.05,
        "overvaluation_penalty": -0.15,
        "risk_penalty": -0.10,
    })

    # --- Output ---
    output_dir: str = "output"


SETTINGS = Settings()
