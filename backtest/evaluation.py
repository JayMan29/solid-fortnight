"""
Standard performance stats plus a helper for the "does adding a data
source actually help" comparison table (Baseline vs +News vs +NLP)
from the project brief.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    if returns.std() == 0 or returns.empty:
        return np.nan
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def max_drawdown(cumulative_returns: pd.Series) -> float:
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1.0
    return float(drawdown.min())


def evaluate_strategy(
    net_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict[str, float]:
    excess = net_returns - benchmark_returns
    cumulative = (1 + net_returns).cumprod()
    return {
        "annualized_return": float((1 + net_returns.mean()) ** 12 - 1),
        "annualized_excess_return": float(excess.mean() * 12),
        "sharpe": sharpe_ratio(net_returns),
        "max_drawdown": max_drawdown(cumulative),
        "hit_rate_vs_benchmark": float((excess > 0).mean()),
    }


def compare_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    `results` maps a model name (e.g. "baseline", "+news_volume", "+nlp_events")
    to its evaluate_strategy() output. Prints a comparison table like the
    one in the project brief -- use this to decide whether a data source
    earned its complexity, rather than assuming it did.
    """
    return pd.DataFrame(results).T.round(4)
