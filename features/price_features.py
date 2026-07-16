"""
Momentum and risk features derived purely from OHLCV price history.
All functions take an `as_of_date` and only look backward from it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _return_over_window(prices: pd.Series, as_of_date: pd.Timestamp, days: int) -> float:
    window = prices.loc[:as_of_date]
    if len(window) < 2:
        return np.nan
    start_idx = max(0, len(window) - days)
    start_price = window.iloc[start_idx]
    end_price = window.iloc[-1]
    if start_price == 0 or pd.isna(start_price):
        return np.nan
    return float(end_price / start_price - 1.0)


def calculate_price_features(
    close_prices: pd.Series,
    as_of_date: pd.Timestamp,
    sector_close_prices: pd.Series | None = None,
) -> dict[str, float]:
    momentum_12m = _return_over_window(close_prices, as_of_date, 252)
    momentum_6m = _return_over_window(close_prices, as_of_date, 126)
    momentum_3m = _return_over_window(close_prices, as_of_date, 63)
    momentum_1m = _return_over_window(close_prices, as_of_date, 21)
    # 12-month momentum EXCLUDING the most recent month (standard academic construction)
    momentum_12m_ex1m = np.nan
    if not (np.isnan(momentum_12m) or np.isnan(momentum_1m)):
        momentum_12m_ex1m = (1 + momentum_12m) / (1 + momentum_1m) - 1

    window = close_prices.loc[:as_of_date]
    daily_returns = window.pct_change().dropna()
    volatility_6m = float(daily_returns.tail(126).std() * np.sqrt(252)) if len(daily_returns) else np.nan

    trailing_high = window.tail(252).max() if len(window) else np.nan
    distance_from_52w_high = (
        float(window.iloc[-1] / trailing_high - 1.0)
        if trailing_high and not pd.isna(trailing_high)
        else np.nan
    )

    relative_momentum_12m = np.nan
    if sector_close_prices is not None:
        sector_return = _return_over_window(sector_close_prices, as_of_date, 252)
        if not (np.isnan(momentum_12m) or np.isnan(sector_return)):
            relative_momentum_12m = momentum_12m - sector_return

    return {
        "momentum_12m_ex1m": momentum_12m_ex1m,
        "momentum_6m": momentum_6m,
        "momentum_3m": momentum_3m,
        "volatility_6m": volatility_6m,
        "distance_from_52w_high": distance_from_52w_high,
        "relative_momentum_12m": relative_momentum_12m,
    }
