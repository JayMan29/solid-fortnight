"""
Recent filings index per company (form type, filing date, accession number).
Useful as a high-reliability event source to cross-check news-derived events
against: an 8-K "Entry into a Material Definitive Agreement" is a much
stronger signal than a press release alone.
"""
from __future__ import annotations
import requests
import pandas as pd

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


def fetch_recent_filings(cik: int, user_agent: str) -> pd.DataFrame:
    headers = {"User-Agent": user_agent}
    response = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()
    df = pd.DataFrame(recent)
    df["filingDate"] = pd.to_datetime(df["filingDate"])
    return df[["form", "filingDate", "accessionNumber", "primaryDocument"]].sort_values(
        "filingDate", ascending=False
    )


MATERIAL_8K_ITEMS = {
    "1.01": "material_agreement",
    "2.01": "acquisition_or_disposition",
    "2.02": "earnings_results",
    "5.02": "executive_change",
    "7.01": "reg_fd_disclosure",
}
