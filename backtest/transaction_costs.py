"""
A backtest that ignores transaction costs is close to useless for a
strategy that rebalances monthly. This applies a flat cost per unit of
turnover (spread + commission + slippage, combined into one number you
should calibrate to your actual broker).
"""
from __future__ import annotations
import pandas as pd

DEFAULT_COST_BPS = 10  # 0.10% per unit of one-way turnover; adjust to your broker


def calculate_turnover(holdings: pd.DataFrame, date_col: str = "date",
                        ticker_col: str = "ticker") -> pd.DataFrame:
    """One-way turnover between consecutive rebalance dates, per date."""
    dates = sorted(holdings[date_col].unique())
    rows = []
    prev_weights: dict[str, float] = {}
    for d in dates:
        current = holdings.loc[holdings[date_col] == d].set_index(ticker_col)["weight"].to_dict()
        all_tickers = set(current) | set(prev_weights)
        turnover = sum(abs(current.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in all_tickers) / 2
        rows.append({date_col: d, "turnover": turnover})
        prev_weights = current
    return pd.DataFrame(rows)


def apply_transaction_costs(
    gross_returns: pd.DataFrame,
    holdings: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
    date_col: str = "date",
) -> pd.DataFrame:
    turnover = calculate_turnover(holdings, date_col=date_col)
    merged = gross_returns.merge(turnover, on=date_col, how="left")
    merged["turnover"] = merged["turnover"].fillna(0.0)
    merged["cost"] = merged["turnover"] * (cost_bps / 10_000)
    merged["net_return"] = merged["gross_return"] - merged["cost"]
    return merged
