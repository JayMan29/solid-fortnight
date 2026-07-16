"""
Fundamentals from SEC EDGAR's free XBRL "company facts" API.

Docs: https://www.sec.gov/edgar/sec-api-documentation
Requires a descriptive User-Agent header (SEC blocks generic ones).

IMPORTANT (point-in-time correctness): every fact returned includes a
`filed` date. When building historical features, you must only use
facts whose `filed` date is on or before your as-of date -- otherwise
you leak future information into the past (look-ahead bias).
"""
from __future__ import annotations
from datetime import date
import requests
import pandas as pd

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


def get_cik_for_ticker(ticker: str, user_agent: str) -> int:
    headers = {"User-Agent": user_agent}
    response = requests.get(TICKER_MAP_URL, headers=headers, timeout=30)
    response.raise_for_status()
    mapping = response.json()
    for entry in mapping.values():
        if entry["ticker"].upper() == ticker.upper():
            return int(entry["cik_str"])
    raise KeyError(f"Ticker {ticker} not found in SEC ticker map")


def fetch_company_facts(ticker: str, user_agent: str) -> dict:
    cik = get_cik_for_ticker(ticker, user_agent)
    url = COMPANY_FACTS_URL.format(cik=cik)
    headers = {"User-Agent": user_agent}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_quarterly_metric(
    company_facts: dict,
    concept: str,
    taxonomy: str = "us-gaap",
    as_of_date: date | None = None,
) -> pd.DataFrame:
    """
    Pull a single XBRL concept (e.g. 'Revenues', 'GrossProfit',
    'NetCashProvidedByUsedInOperatingActivities') as a point-in-time
    respecting DataFrame with columns: end, val, filed, form.
    """
    try:
        units = company_facts["facts"][taxonomy][concept]["units"]
    except KeyError:
        return pd.DataFrame(columns=["end", "val", "filed", "form"])

    rows = []
    for unit_values in units.values():
        for entry in unit_values:
            rows.append({
                "end": entry.get("end"),
                "val": entry.get("val"),
                "filed": entry.get("filed"),
                "form": entry.get("form"),
                "fp": entry.get("fp"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    if as_of_date is not None:
        df = df[df["filed"] <= pd.Timestamp(as_of_date)]
    return df.sort_values("end")
