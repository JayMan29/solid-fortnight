"""
Daily price data. Uses stooq's free CSV endpoint (no key required) as the
primary source, since it's simple and reliable for backtesting. Falls back
to yfinance if installed and stooq fails.
"""
from __future__ import annotations
from datetime import date
import io
import pandas as pd
import requests

STOOQ_URL = "https://stooq.com/q/d/l/"


def fetch_price_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by date with columns:
    open, high, low, close, volume
    """
    try:
        return _fetch_from_stooq(ticker, start, end)
    except Exception as stooq_error:
        try:
            return _fetch_from_yfinance(ticker, start, end)
        except Exception as yf_error:
            raise RuntimeError(
                f"Could not fetch prices for {ticker}. "
                f"stooq error: {stooq_error}; yfinance error: {yf_error}"
            )


def _fetch_from_stooq(ticker: str, start: date, end: date) -> pd.DataFrame:
    # Stooq uses lowercase tickers with a .us suffix for US equities.
    symbol = f"{ticker.lower()}.us"
    params = {"s": symbol, "d1": start.strftime("%Y%m%d"),
              "d2": end.strftime("%Y%m%d"), "i": "d"}
    response = requests.get(STOOQ_URL, params=params, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    if df.empty or "Date" not in df.columns:
        raise ValueError(f"No stooq data returned for {ticker}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]]


def _fetch_from_yfinance(ticker: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf  # optional dependency
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No yfinance data returned for {ticker}")
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]]
