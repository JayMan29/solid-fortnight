"""
Turns a monthly ranking into a simple equal-weight top-decile portfolio,
rebalanced monthly. This is intentionally simple -- the point of the
backtest is to check whether the *ranking* has any predictive value at
all before worrying about portfolio optimization.
"""
from __future__ import annotations
import pandas as pd


def build_top_decile_portfolio(
    ranked_scores: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    score_col: str = "composite_score",
    top_fraction: float = 0.10,
) -> pd.DataFrame:
    """
    Returns a DataFrame of (date, ticker, weight) for each rebalance date,
    equal-weighting the top `top_fraction` of the universe by score.
    """
    holdings = []
    for rebalance_date, group in ranked_scores.groupby(date_col):
        n_holdings = max(1, int(round(len(group) * top_fraction)))
        top = group.sort_values(score_col, ascending=False).head(n_holdings)
        weight = 1.0 / len(top)
        for _, row in top.iterrows():
            holdings.append({date_col: rebalance_date, ticker_col: row[ticker_col], "weight": weight})
    return pd.DataFrame(holdings)
