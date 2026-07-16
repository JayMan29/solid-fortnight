"""
The transparent, human-auditable scoring model. This runs before (and
alongside) any ML model, since a ranking you can explain is more useful
for a discretionary investor than a black-box probability.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import SETTINGS


def zscore_within_group(series: pd.Series, group: pd.Series) -> pd.Series:
    """Standardize each value against its own sector/month group, not the whole universe."""
    grouped = series.groupby(group)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return (series - mean) / std


def compute_composite_score(features: pd.DataFrame, sector_col: str = "sector") -> pd.DataFrame:
    """
    `features` must contain one row per (ticker, date) with raw feature
    columns. Returns the same frame with added z-scored components and
    a final `composite_score` column.

    Expected raw columns (missing ones are treated as neutral/0 after scaling):
      revenue_growth_yoy, gross_margin, free_cash_flow_margin,   -> growth/quality
      momentum_6m, relative_momentum_12m,                        -> momentum
      news_acceleration, news_attention_zscore,                  -> news volume
      sentiment_net,                                             -> news sentiment
      commercial_event_score_90d,                                -> event quality
      rnd_growth_yoy,                                             -> tech position proxy
      net_debt, volatility_6m, share_dilution_yoy,                -> risk
      ev_to_revenue                                               -> valuation
    """
    df = features.copy()
    if sector_col not in df.columns:
        df[sector_col] = "ALL"  # fallback: one big group

    component_map = {
        "fundamental_growth": ["revenue_growth_yoy"],
        "business_quality": ["gross_margin", "free_cash_flow_margin"],
        "price_momentum": ["momentum_6m", "relative_momentum_12m"],
        "news_acceleration": ["news_acceleration", "news_attention_zscore"],
        "news_sentiment": ["sentiment_net"],
        "commercial_event_quality": ["commercial_event_score_90d"],
        "technology_position": ["rnd_growth_yoy"],
        "financial_strength": ["net_debt"],  # inverted below
        "overvaluation_penalty": ["ev_to_revenue"],
        "risk_penalty": ["volatility_6m", "share_dilution_yoy"],
    }

    for component, cols in component_map.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            df[f"z_{component}"] = 0.0
            continue
        z_cols = []
        for c in present:
            z = zscore_within_group(df[c], df[sector_col]).fillna(0.0)
            if c in ("net_debt", "ev_to_revenue", "volatility_6m", "share_dilution_yoy"):
                z = -z  # higher debt/valuation/vol/dilution is worse
            z_cols.append(z)
        df[f"z_{component}"] = pd.concat(z_cols, axis=1).mean(axis=1)

    weights = SETTINGS.weights
    df["composite_score"] = sum(
        df[f"z_{component}"] * weight for component, weight in weights.items()
    )
    return df


def explain_score_change(
    current_row: pd.Series,
    previous_row: pd.Series | None,
    top_n: int = 3,
) -> str:
    """Human-readable explanation of why a score moved, for the dashboard."""
    if previous_row is None:
        return "First observation for this ticker; no prior score to compare."
    component_cols = [c for c in current_row.index if c.startswith("z_")]
    deltas = (current_row[component_cols] - previous_row[component_cols]).sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    lines = []
    for name, delta in deltas.head(top_n).items():
        direction = "up" if delta > 0 else "down"
        lines.append(f"  - {name.replace('z_', '').replace('_', ' ')} moved {direction} ({delta:+.2f})")
    return "\n".join(lines) if lines else "  - No material change in scored components."
